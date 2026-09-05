#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include "utils/ConfigurationManager.hpp"

#include <cstdio>
#include <exception>

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace {

using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::Boundary::HorizontalBoundaryStencils;

int failures = 0;
int mpi_rank = 0;

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "Rank %d FAIL: %s\n", mpi_rank, message);
}

void test_constant_q2_halos_2d(const Grid& grid) {
    const int halo = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Field<2> field("boundary_test_2d", {ny, nx});
    auto data = field.get_mutable_device_data();

    Kokkos::parallel_for(
        "initialize_boundary_test_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            data(j, i) =
                VVM::real(1000.0) * static_cast<VVM::Real>(j) +
                static_cast<VVM::Real>(i);
        });

    HorizontalBoundaryStencils stencils(grid);
    stencils.fill_constant_q2_halos(field);

    const auto host = field.get_host_data();

    const bool is_south_boundary =
        grid.get_local_physical_start_y() == 0;

    const bool is_north_boundary =
        grid.get_local_physical_end_y() ==
        grid.get_global_points_y() - 1;

    if (is_south_boundary) {
        for (int j = 0; j < halo; ++j) {
            for (int i = 0; i < nx; ++i) {
                check(
                    host(j, i) == host(halo, i),
                    "2-D south q2 halo must equal the first physical row");
            }
        }
    }

    if (is_north_boundary) {
        const int last_physical_j = ny - halo - 1;

        for (int j = ny - halo; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                check(
                    host(j, i) == host(last_physical_j, i),
                    "2-D north q2 halo must equal the last physical row");
            }
        }
    }

    for (int j = halo; j < ny - halo; ++j) {
        for (int i = 0; i < nx; ++i) {
            const VVM::Real expected =
                VVM::real(1000.0) * static_cast<VVM::Real>(j) +
                static_cast<VVM::Real>(i);

            check(
                host(j, i) == expected,
                "2-D physical rows must remain unchanged");
        }
    }
}

void test_constant_q2_halos_3d(const Grid& grid) {
    const int halo = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Field<3> field("boundary_test_3d", {nz, ny, nx});
    auto data = field.get_mutable_device_data();

    Kokkos::parallel_for(
        "initialize_boundary_test_3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            data(k, j, i) =
                VVM::real(1000000.0) * static_cast<VVM::Real>(k) +
                VVM::real(1000.0) * static_cast<VVM::Real>(j) +
                static_cast<VVM::Real>(i);
        });

    HorizontalBoundaryStencils stencils(grid);
    stencils.fill_constant_q2_halos(field);

    const auto host = field.get_host_data();

    const bool is_south_boundary =
        grid.get_local_physical_start_y() == 0;

    const bool is_north_boundary =
        grid.get_local_physical_end_y() ==
        grid.get_global_points_y() - 1;

    if (is_south_boundary) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < halo; ++j) {
                for (int i = 0; i < nx; ++i) {
                    check(
                        host(k, j, i) == host(k, halo, i),
                        "3-D south q2 halo must equal the first physical row");
                }
            }
        }
    }

    if (is_north_boundary) {
        const int last_physical_j = ny - halo - 1;

        for (int k = 0; k < nz; ++k) {
            for (int j = ny - halo; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    check(
                        host(k, j, i) ==
                            host(k, last_physical_j, i),
                        "3-D north q2 halo must equal the last physical row");
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = halo; j < ny - halo; ++j) {
            for (int i = 0; i < nx; ++i) {
                const VVM::Real expected =
                    VVM::real(1000000.0) *
                        static_cast<VVM::Real>(k) +
                    VVM::real(1000.0) *
                        static_cast<VVM::Real>(j) +
                    static_cast<VVM::Real>(i);

                check(
                    host(k, j, i) == expected,
                    "3-D physical cells must remain unchanged");
            }
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 2) {
        if (mpi_rank == 0) {
            std::fprintf(stderr, "usage: %s <config.json>\n", argv[0]);
        }

        MPI_Finalize();
        return 2;
    }

    Kokkos::initialize(
        Kokkos::InitializationSettings().set_device_id(0));

    {
        try {
            const VVM::Utils::ConfigurationManager config(argv[1]);
            const Grid grid(config, MPI_COMM_WORLD);

            test_constant_q2_halos_2d(grid);
            test_constant_q2_halos_3d(grid);
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(
                stderr,
                "Rank %d unexpected exception: %s\n",
                mpi_rank,
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

    if (mpi_rank == 0) {
        if (global_failures == 0) {
            std::fprintf(
                stdout,
                "test_horizontal_boundary_stencils: PASS\n");
        } else {
            std::fprintf(
                stderr,
                "test_horizontal_boundary_stencils: %d failure(s)\n",
                global_failures);
        }
    }

    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
