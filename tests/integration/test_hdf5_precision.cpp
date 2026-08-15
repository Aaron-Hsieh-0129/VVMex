// output.precision applied to the legacy HDF5 writer: field datasets carry the
// configured element type, clocks and coordinates stay VVM::Real, and the
// values survive the conversion.
#include "OutputPrecisionCheck.hpp"

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
    int comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (argc != 4) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "usage: test_hdf5_precision BASE_CONFIG WORK_ROOT "
                         "unset|native|float32|float64\n");
        }
        MPI_Finalize();
        return 2;
    }

    const std::string precision = argv[3];
    const std::filesystem::path case_dir =
        std::filesystem::path(argv[2]) / ("hdf5_precision_" + precision);
    const std::filesystem::path config_path =
        case_dir.parent_path() / ("hdf5_precision_" + precision + ".json");

    try {
        if (rank == 0) {
            std::filesystem::remove_all(case_dir);
            std::filesystem::create_directories(case_dir.parent_path());
            std::ofstream output(config_path);
            output << VVMTest::make_config(argv[1], case_dir, "HDF5", precision).dump(2)
                   << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);

        Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
        {
            VVM::Utils::ConfigurationManager config(config_path.string());
            VVM::Core::Grid grid(config, MPI_COMM_WORLD);
            VVM::Core::Parameters parameters(config, grid);
            VVMTest::fill_coordinates(parameters, grid);
#if defined(ENABLE_NCCL)
            ncclUniqueId id;
            if (rank == 0) ncclGetUniqueId(&id);
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
            ncclComm_t nccl_comm;
            ncclCommInitRank(&nccl_comm, comm_size, id, rank);
            cudaStream_t stream = Kokkos::Cuda().cuda_stream();
            VVM::Core::State state(config, parameters, grid, nccl_comm, stream);
#else
            VVM::Core::State state(config, parameters, grid);
#endif
            VVMTest::fill_fields(state, grid);

            const VVM::Real interval =
                config.get_value<VVM::Real>("simulation.output_interval_s");
            VVM::IO::OutputManager manager(
                config, grid, parameters, state, MPI_COMM_WORLD);
            // Two steps, because the conversion buffers and the variable
            // definitions are created once and reused by every later write.
            manager.write(0, VVM::real(0.0));
            manager.write(1, interval);
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                VVMTest::inspect(case_dir / "history_000000.h5", precision);
                VVMTest::inspect(case_dir / "history_000001.h5", precision);
            }
            MPI_Barrier(MPI_COMM_WORLD);
#if defined(ENABLE_NCCL)
            ncclCommDestroy(nccl_comm);
#endif
        }
        Kokkos::finalize();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] exception: %s\n", rank, e.what());
        ++VVMTest::failures;
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }

    int global_failures = 0;
    MPI_Allreduce(&VVMTest::failures, &global_failures, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (rank == 0 && global_failures == 0) {
        std::fprintf(stdout, "test_hdf5_precision(%s): all checks passed\n",
                     precision.c_str());
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
