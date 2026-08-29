#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <hdf5.h>
#include <memory>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(ENABLE_NCCL)
#include <nccl.h>
#endif
#include <unistd.h>

#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/HaloExchanger.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "io/bp5/Bp5HistoryWriter.hpp"
#include "io/history/HistoryWriter.hpp"
#include "io/history/LegacyHistoryWriter.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/NumericalConfigurationValidation.hpp"
#include "utils/SstPath.hpp"
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"

#include "driver/Model.hpp"
#include "io/IOServer.hpp"

#if defined(ENABLE_NCCL)
void init_nccl(ncclComm_t* comm, int rank, int size, MPI_Comm mpi_comm) {
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
    ncclCommInitRank(comm, size, id, rank);
}
#endif

int world_rank_of(MPI_Comm comm) {
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
}

int get_io_tasks(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--io-tasks" && i + 1 < argc) {
            try {
                return std::stoi(argv[i + 1]);
            } 
            catch (const std::exception&) {
                if (world_rank_of(MPI_COMM_WORLD) == 0) {
                    std::cerr << "[Main] ERROR: --io-tasks expects an integer, got \""
                              << argv[i + 1] << "\"." << std::endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    return 0;
}

struct CommandLine {
    std::string config_file_path;
    bool help_requested = false;
};

CommandLine parse_command_line(int argc, char *argv[]) {
    CommandLine cli;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { cli.help_requested = true; continue; }
        if (arg == "--io-tasks") { i++; continue; }
        // An empty argv element would index past the end and, worse, blank the
        // config path.
        if (!arg.empty() && arg[0] != '-') cli.config_file_path = arg;
    }
    return cli;
}

bool is_launcher_rank_zero() {
    const char* rank_vars[] = {"OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK",
                               "MV2_COMM_WORLD_RANK", "SLURM_PROCID"};
    for (const char* var : rank_vars) {
        if (const char* value = std::getenv(var)) return std::string(value) == "0";
    }
    return true;
}

std::string project_root() {
    return std::string(VVM_ROOT_DIR);
}

void print_usage(const char* argv0) {
    const std::string exe = (argv0 && *argv0) ? std::string(argv0) : std::string("./build/vvm");
    const std::string root = project_root();
    const std::string prefix = root.empty() ? std::string("$VVM_ROOT") : root;
    const std::string case_dir = prefix + "/rundata/input_configs/default_cases";
    const std::string example = case_dir + "/advection_u.json";

    std::cout <<
"VVMex -- Vector Vorticity cloud-resolving Model (Kokkos + MPI)\n"
"\n"
"vvm takes one required argument: the path to a run configuration file (JSON).\n"
"There is no built-in default configuration, which is why a bare `" << exe << "`\n"
"has nothing to run.\n"
"\n"
"USAGE\n"
"  mpirun -np <ranks> " << exe << " <config.json> [--io-tasks <n>]\n"
"\n"
"  Recommended instead of mpirun: ./submit.py, which derives the environment from\n"
"  the CMake preset and places ranks, GPUs and threads for you.\n"
"\n"
"    ./submit.py --local --preset <preset> -c <config.json> --compute <ranks>\n"
"    ./submit.py                 # no arguments: asks for each value in turn\n"
"\n"
"YOUR FIRST RUN\n"
"  1. export VVM_ROOT=<repository root>\n"
"     Relative paths inside a configuration resolve against it, not the shell's\n"
"     working directory.\n"
"  2. Pick a case:\n"
"       ls " << case_dir << "\n"
"  3. In that file, point output.output_dir at a directory you can write, and\n"
"     check initial_conditions.source_file / netcdf_reader.source_file exist.\n"
"  4. Run it:\n"
"       ./submit.py --local --preset <preset> -c " << example << " --compute 1\n"
"     or, bypassing the wrapper:\n"
"       mpirun -np 1 " << exe << " " << example << "\n"
"\n"
"ARGUMENTS\n"
"  <config.json>   First non-option argument. Required.\n"
"  --io-tasks <n>  Reserve n of the MPI ranks as dedicated I/O servers. Valid only\n"
"                  when output.engine is \"SST\"; the remaining ranks run the model.\n"
"  -h, --help      Print this message and exit.\n"
"\n"
"WHERE TO READ MORE (under " << (root.empty() ? std::string("the repository root") : root) << ")\n"
"  docs/quick-start.md                    build the code, configure and run a case\n"
"  docs/user-guides/configuration.md      every configuration key and its default\n"
"  docs/user-guides/job-submission.md     ranks, GPUs, OpenMP threads, SLURM\n"
"  docs/user-guides/output.md             HDF5 / SST / BP5 engines, precision, restart\n"
"  docs/developer-guides/architecture.md  how the model is put together\n"
              << std::flush;
}

int run_vvm(int argc, char *argv[], int world_rank, int world_size) {
    const std::string config_file_path = parse_command_line(argc, argv).config_file_path;

    if (!std::ifstream(config_file_path).good()) {
        if (world_rank == 0) {
            std::cerr << "[Main] ERROR: cannot open configuration file: "
                      << config_file_path << "\n"
                      << "  Run `" << argv[0] << " --help` for how to pick one."
                      << std::endl;
        }
        return 1;
    }

    VVM::Utils::ConfigurationManager config(config_file_path);

    // clean existing SST file to prevent errors
    {
        const std::string engine = config.get_value<std::string>("output.engine", "HDF5");
        if (engine == "SST") {
            int cleanup_failed = 0;
            if (world_rank == 0) {
                try {
                    const std::string output_dir = config.get_value<std::string>("output.output_dir", std::string());
                    const std::string prefix = config.get_value<std::string>("output.output_filename_prefix", std::string());

                    const VVM::Utils::SstCleanupResult cleanup = VVM::Utils::remove_stale_sst_path(output_dir, prefix);

                    switch (cleanup.outcome) {
                        case VVM::Utils::SstCleanupOutcome::Removed:
                            std::cout << "[Main] Removing stale SST path: "
                                      << cleanup.path.string() << " ("
                                      << cleanup.removed_entries << " entries)" << std::endl;
                            break;
                        case VVM::Utils::SstCleanupOutcome::NotPresent:
                            // Nothing from a previous run. Say nothing.
                            break;
                        case VVM::Utils::SstCleanupOutcome::Rejected:
                        case VVM::Utils::SstCleanupOutcome::Failed:
                            std::cerr << "[Main] ERROR: stale SST cleanup: "
                                      << cleanup.message << "\n"
                                      << "  Check output.output_dir and "
                                         "output.output_filename_prefix in the configuration."
                                      << std::endl;
                            cleanup_failed = 1;
                            break;
                    }
                } 
                catch (const std::exception& e) {
                    std::cerr << "[Main] ERROR: stale SST cleanup: " << e.what() << std::endl;
                    cleanup_failed = 1;
                }
            }
            MPI_Bcast(&cleanup_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (cleanup_failed != 0) {
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Split Compute and IO jobs
    int num_io_tasks = get_io_tasks(argc, argv);

    {
        std::string engine = config.get_value<std::string>("output.engine", "HDF5");
        if (engine != "SST" && num_io_tasks > 0) {
            if (world_rank == 0) {
                std::cerr << "[Main] ERROR: output.engine=\"" << engine
                          << "\" but job was submitted with --io-tasks " << num_io_tasks << ".\n"
                          << "  IO server ranks are only used with SST engine.\n"
                          << "  Re-submit without --io-tasks (or with --io-tasks 0)." << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int num_sim_tasks = world_size - num_io_tasks;
    if (num_sim_tasks <= 0) {
        if (world_rank == 0) std::cerr << "Error: Not enough ranks for simulation!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    VVM::Utils::validate_numerical_configuration(config, num_sim_tasks);

    int color = (world_rank < num_sim_tasks) ? 0 : 1;
    MPI_Comm split_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &split_comm);
    int split_rank, split_size;
    MPI_Comm_rank(split_comm, &split_rank);
    MPI_Comm_size(split_comm, &split_size);

    int threads_per_rank = 1;
    if (const char* env_p = std::getenv("OMP_NUM_THREADS")) {
        try {
            threads_per_rank = std::stoi(env_p);
        } catch (const std::exception&) {
            threads_per_rank = 0;  // rejected below with the offending value
        }
        if (threads_per_rank < 1) {
            if (world_rank == 0) {
                std::cerr << "[Main] WARNING: OMP_NUM_THREADS=\"" << env_p
                          << "\" is not a positive integer; using 1." << std::endl;
            }
            threads_per_rank = 1;
        }
    }
    omp_set_num_threads(threads_per_rank);

    // Only compute ranks touch the GPU. 
    const bool is_compute_rank = (color == 0);

    int num_gpus = 0;
    if (is_compute_rank) {
#if defined(KOKKOS_ENABLE_CUDA)
        cudaError_t cuda_err = cudaGetDeviceCount(&num_gpus);
        if (cuda_err != cudaSuccess || num_gpus <= 0) {
            std::cerr << "[Rank " << world_rank
                      << "] ERROR: no visible CUDA device. CUDA_VISIBLE_DEVICES="
                      << (std::getenv("CUDA_VISIBLE_DEVICES")
                              ? std::getenv("CUDA_VISIBLE_DEVICES")
                              : "<unset>")
                      << " cudaGetDeviceCount error="
                      << cudaGetErrorString(cuda_err)
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
#endif

        Kokkos::InitializationSettings args;
#if defined(KOKKOS_ENABLE_CUDA)
        // core_run.sh constrains each process to one intended GPU.
        args.set_device_id(0);
#endif
        Kokkos::initialize(args);
    }

    if (world_rank == 0) {
        std::cout << "[System] CPU Threads per Rank set to: "
                  << threads_per_rank << std::endl;
    }
    std::cout << "[KokkosInit]"
              << " world_rank=" << world_rank
              << " role=" << (is_compute_rank ? "compute" : "io")
              << " CUDA_VISIBLE_DEVICES="
              << (std::getenv("CUDA_VISIBLE_DEVICES")
                      ? std::getenv("CUDA_VISIBLE_DEVICES")
                      : "<unset>")
              << " visible_num_gpus=" << num_gpus
              << (is_compute_rank ? " kokkos_device_id=0" : " kokkos=skipped(host-only)")
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (!is_compute_rank) {
        VVM::IO::run_io_server(split_comm, config);
        MPI_Comm_free(&split_comm);
        return 0;
    }

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm;
    init_nccl(&nccl_comm, split_rank, split_size, split_comm);
#endif

    {
        auto& timing = VVM::Utils::TimingManager::get_instance();

        timing.configure(
            config.get_value<bool>("performance.timing.enable", true),
            config.get_value<bool>("performance.timing.fence_gpu", false),
            config.get_value<int>("performance.timing.warmup_steps", 0)
        );

        const int timing_print_interval_steps =
            config.get_value<int>("performance.timing.print_interval_steps", 0);

        const bool timing_reset_after_interval_print =
            config.get_value<bool>("performance.timing.reset_after_interval_print", false);

        timing.start_timer("total_vvm");

        timing.start_timer("initialize");

        if (split_rank == 0) std::cout << "VVM Model Simulation Started." << std::endl;

        // if (rank == 0) config.print_config(); // Print loaded configuration


#if defined(ENABLE_NCCL)
        cudaStream_t stream = Kokkos::Cuda().cuda_stream();
#endif

        // Create a VVM model instance and run the simulation
        VVM::Core::Grid grid(config, split_comm);
        VVM::Core::Parameters parameters(config, grid);
        grid.print_info();

#if defined(ENABLE_NCCL)
        VVM::Core::State state(config, parameters, grid, nccl_comm, stream);
        VVM::Core::HaloExchanger halo_exchanger(config, grid, nccl_comm, stream);
#else
        VVM::Core::State state(config, parameters, grid);
        VVM::Core::HaloExchanger halo_exchanger(grid);
#endif
        VVM::Driver::Model model(config, parameters, grid, state, halo_exchanger);
        model.init();

        timing.stop_timer("initialize");

        std::unique_ptr<VVM::IO::HistoryWriter> history_writer;
        const std::string output_engine =
            config.get_value<std::string>("output.engine", "HDF5");
        if (output_engine == "BP5") {
            history_writer = std::make_unique<VVM::IO::BP5::Bp5HistoryWriter>(
                config, grid, parameters, state, split_comm);
        } else {
            history_writer = std::make_unique<VVM::IO::LegacyHistoryWriter>(
                config, grid, parameters, state, split_comm);
        }

        const bool restart_enabled = config.get_value<bool>("restart.enable", false);
        const bool output_initial_step =
            config.get_value<bool>("output.output_initial_step", true);
        if (output_initial_step) {
            VVM::Utils::Timer timer("io");
            history_writer->write(state.get_step(), state.get_time());
        } else if (split_rank == 0) {
            std::cout << "[Output] Skipping initial full output." << std::endl;
        }
        // output_manager->write_static_topo_file();

        // Simulation loop parameters
        double total_time = config.get_value<double>("simulation.total_time_s");
        double dt = parameters.get_value_host(parameters.dt);
        double output_interval = config.get_value<double>("simulation.output_interval_s");
        double next_output_time = output_interval;
        if (restart_enabled) {
            next_output_time = (std::floor(state.get_time() / output_interval) + 1.0) * output_interval;
        }

        // Simulation loop
        while (state.get_time() < total_time) {
            model.run_step(dt);

            state.increment_step();
            state.advance_time(dt);

            if (split_rank == 0) std::cout << state.get_time() << std::endl;

             // Output data at specified intervals
            if (state.get_time() >= next_output_time) {
                {
                    VVM::Utils::Timer timer("io");
                    history_writer->write(state.get_step(), state.get_time());
                }
                next_output_time += output_interval;
            }
            if (timing_print_interval_steps > 0 &&
                state.get_step() % static_cast<size_t>(timing_print_interval_steps) == 0) {
                timing.print_timings(split_comm, timing_reset_after_interval_print);
            }
        }
        history_writer->close();

        timing.stop_timer("total_vvm");
        timing.print_timings(split_comm, false);

        model.finalize();
        Kokkos::fence();
    }
#if defined(ENABLE_NCCL)
    ncclCommDestroy(nccl_comm);
#endif
    MPI_Comm_free(&split_comm);
    Kokkos::finalize();
    return 0;
}

int main(int argc, char *argv[]) {
    const CommandLine cli = parse_command_line(argc, argv);
    if (cli.help_requested || cli.config_file_path.empty()) {
        if (is_launcher_rank_zero()) print_usage(argv[0]);
        return 0;
    }

    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (H5set_free_list_limits(0, 0, 0, 0, 0, 0) < 0) {
        if (world_rank == 0)
            std::cerr << "[Main] ERROR: unable to configure HDF5 free-list limits." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int status = 0;
    try {
        status = run_vvm(argc, argv, world_rank, world_size);
    }
    catch (const std::exception& e) {
        std::cerr << "[Rank " << world_rank << "] FATAL: " << e.what() << std::endl;
        std::cerr.flush();
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    catch (...) {
        std::cerr << "[Rank " << world_rank << "] FATAL: unknown exception" << std::endl;
        std::cerr.flush();
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return status;
}
