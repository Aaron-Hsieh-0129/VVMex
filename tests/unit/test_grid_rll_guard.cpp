#include "core/Grid.hpp"
#include "utils/ConfigurationManager.hpp"

#include <cstdio>
#include <exception>
#include <string>

#include <Kokkos_Core.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
        if (rank == 0) {
            std::fprintf(stderr, "usage: %s <config.json>\n", argv[0]);
        }

        MPI_Finalize();
        return 2;
    }

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));

    int failures = 0;

    try {
        const VVM::Utils::ConfigurationManager config(argv[1]);

        try {
            const VVM::Core::Grid grid(config, MPI_COMM_WORLD);
            (void)grid;

            ++failures;

            if (rank == 0) {
                std::fprintf(
                    stderr,
                    "FAIL: Grid accepted RLL before runtime integration was enabled\n");
            }
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();

            if (message.find("RLL Grid runtime construction is not yet enabled") ==
                std::string::npos) {
                ++failures;

                if (rank == 0) {
                    std::fprintf(
                        stderr,
                        "FAIL: Grid rejected RLL for an unexpected reason: %s\n",
                        error.what());
                }
            }
        }
    } catch (const std::exception& error) {
        ++failures;

        if (rank == 0) {
            std::fprintf(
                stderr,
                "FAIL: unexpected exception: %s\n",
                error.what());
        }
    }

    Kokkos::finalize();

    int global_failures = 0;

    MPI_Allreduce(
        &failures,
        &global_failures,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD);

    if (rank == 0 && global_failures == 0) {
        std::puts("test_grid_rll_guard: PASS");
    }

    MPI_Finalize();

    return global_failures == 0 ? 0 : 1;
}
