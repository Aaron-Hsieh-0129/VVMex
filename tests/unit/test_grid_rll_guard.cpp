#include "core/BoundaryConditionManager.hpp"
#include "core/Grid.hpp"
#include "core/geometry/GeometryKind.hpp"
#include "utils/ConfigurationManager.hpp"

#include <cstdio>
#include <exception>
#include <string>

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace {

int failures = 0;
int mpi_rank = 0;

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "Rank %d FAIL: %s\n", mpi_rank, message);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));

    {
        try {
            if (argc != 2) {
                throw std::invalid_argument("Usage: test_grid_rll_guard <configuration.json>");
            }

            const VVM::Utils::ConfigurationManager config(argv[1]);
            const VVM::Core::Grid grid(config, MPI_COMM_WORLD);
            const auto& horizontal = grid.horizontal_specification();

            check(
                grid.geometry().kind() == VVM::Core::Geometry::GeometryKind::RegularLatLon,
                "Grid must construct regular latitude-longitude geometry");

            check(
                horizontal.topology.q1 == VVM::Core::HorizontalEdgeTopology::Periodic,
                "RLL q1 topology must be periodic");

            check(
                horizontal.topology.q2 == VVM::Core::HorizontalEdgeTopology::Bounded,
                "RLL q2 topology must be bounded");

            bool model_guard_threw = false;

            try {
                const VVM::Core::BoundaryConditionManager boundary_conditions(grid);
                (void)boundary_conditions;
            } catch (const std::runtime_error& error) {
                model_guard_threw =
                    std::string(error.what()).find(
                        "full model execution is not enabled yet") != std::string::npos;
            }

            check(
                model_guard_threw,
                "BoundaryConditionManager must prevent incomplete RLL full-model execution");
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(stderr, "Rank %d unexpected exception: %s\n", mpi_rank, error.what());
        }
    }

    Kokkos::finalize();

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (mpi_rank == 0 && global_failures == 0) {
        std::puts("test_grid_rll_guard: PASS");
    }

    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
