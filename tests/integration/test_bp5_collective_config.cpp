#include "io/bp5/Bp5CollectiveValidation.hpp"

#include <mpi.h>

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int failures = 0;
    try {
        VVM::IO::BP5::require_collective_match(
            "identical", MPI_COMM_WORLD, "test configuration");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rank %d: identical configuration failed: %s\n", rank, e.what());
        ++failures;
    }
    bool rejected = false;
    try {
        VVM::IO::BP5::require_collective_match(
            rank == 0 ? "root" : "different-" + std::to_string(rank),
            MPI_COMM_WORLD, "test configuration");
    } catch (const std::exception&) {
        rejected = true;
    }
    if (size > 1 && !rejected) {
        std::fprintf(stderr, "rank %d: inconsistent configuration was accepted\n", rank);
        ++failures;
    }
    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0 && global_failures == 0) {
        std::puts("test_bp5_collective_config: PASS");
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
