// output.precision across the SST relay: the compute rank streams fields at the
// configured element type and the I/O server must relay that type into HDF5
// rather than assuming VVM::Real. Two ranks: one compute, one I/O server.
#include "OutputPrecisionCheck.hpp"

#include "io/IOServer.hpp"
#include "io/OutputManager.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 4 || world_size != 2) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "usage: mpirun -n 2 test_sst_precision BASE_CONFIG WORK_ROOT "
                         "unset|native|float32|float64\n");
        }
        MPI_Finalize();
        return 2;
    }

    const std::string precision = argv[3];
    const std::filesystem::path case_dir =
        std::filesystem::path(argv[2]) / ("sst_precision_" + precision);
    const std::filesystem::path config_path =
        case_dir.parent_path() / ("sst_precision_" + precision + ".json");

    // Last rank is the I/O server, as main.cpp splits them.
    const bool is_io_rank = (rank == world_size - 1);

    try {
        if (rank == 0) {
            std::filesystem::remove_all(case_dir);
            std::filesystem::create_directories(case_dir);
            std::ofstream output(config_path);
            output << VVMTest::make_config(argv[1], case_dir, "SST", precision).dump(2)
                   << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);

        VVM::Utils::ConfigurationManager config(config_path.string());

        MPI_Comm split_comm;
        MPI_Comm_split(MPI_COMM_WORLD, is_io_rank ? 1 : 0, rank, &split_comm);

        if (is_io_rank) {
            VVM::IO::run_io_server(split_comm, config);
        } else {
            Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
            {
                VVM::Core::Grid grid(config, split_comm);
                VVM::Core::Parameters parameters(config, grid);
                VVMTest::fill_coordinates(parameters, grid);
#if defined(ENABLE_NCCL)
                int split_rank = 0;
                int split_size = 1;
                MPI_Comm_rank(split_comm, &split_rank);
                MPI_Comm_size(split_comm, &split_size);
                ncclUniqueId id;
                if (split_rank == 0) ncclGetUniqueId(&id);
                MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, split_comm);
                ncclComm_t nccl_comm;
                ncclCommInitRank(&nccl_comm, split_size, id, split_rank);
                cudaStream_t stream = Kokkos::Cuda().cuda_stream();
                VVM::Core::State state(config, parameters, grid, nccl_comm, stream);
#else
                VVM::Core::State state(config, parameters, grid);
#endif
                VVMTest::fill_fields(state, grid);

                {
                    // Scoped so the writer closes the stream here, which is what
                    // ends the I/O server's read loop.
                    VVM::IO::OutputManager manager(
                        config, grid, parameters, state, split_comm);
                    manager.write(0, VVM::real(0.0));
                }
#if defined(ENABLE_NCCL)
                ncclCommDestroy(nccl_comm);
#endif
            }
            Kokkos::finalize();
        }

        MPI_Comm_free(&split_comm);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) VVMTest::inspect(case_dir / "history_000000.h5", precision);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] exception: %s\n", rank, e.what());
        ++VVMTest::failures;
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }

    int global_failures = 0;
    MPI_Allreduce(&VVMTest::failures, &global_failures, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (rank == 0 && global_failures == 0) {
        std::fprintf(stdout, "test_sst_precision(%s): all checks passed\n",
                     precision.c_str());
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
