#include "core/Grid.hpp"
#include "core/geometry/HorizontalGeometryFactory.hpp"
#include "utils/ConfigurationManager.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

#include <mpi.h>
#include <Kokkos_Core.hpp>

namespace {

using VVM::Core::Grid;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::HorizontalGeometryFactory;
using VVM::Core::Geometry::HorizontalGridSpec;
using VVM::Core::Geometry::HorizontalLocation;

int failures = 0;
int mpi_rank = 0;

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "Rank %d FAIL: %s\n", mpi_rank, message);
}

bool close(const VVM::Real actual, const VVM::Real expected) {
    return std::abs(actual - expected) < VVM::real(1.0e-5);
}

void test_host_metadata(const Grid& grid) {
    const auto& geometry = grid.geometry();
    const auto& layout = geometry.layout();

    check(geometry.kind() == GeometryKind::Cartesian, "Grid must construct Cartesian geometry");
    check(std::strcmp(geometry.name(), "cartesian") == 0, "Grid geometry must report the Cartesian name");
    check(close(geometry.dq1(), grid.get_dx()), "Geometry dq1 must equal Grid dx");
    check(close(geometry.dq2(), grid.get_dy()), "Geometry dq2 must equal Grid dy");

    check(layout.global_nx == grid.get_global_points_x(), "Geometry global_nx must match Grid");
    check(layout.global_ny == grid.get_global_points_y(), "Geometry global_ny must match Grid");
    check(layout.local_physical_nx == grid.get_local_physical_points_x(), "Geometry local_physical_nx must match Grid");
    check(layout.local_physical_ny == grid.get_local_physical_points_y(), "Geometry local_physical_ny must match Grid");
    check(layout.global_start_i == grid.get_local_physical_start_x(), "Geometry global_start_i must match Grid");
    check(layout.global_start_j == grid.get_local_physical_start_y(), "Geometry global_start_j must match Grid");
    check(layout.halo == grid.get_halo_cells(), "Geometry halo must match Grid");
    check(layout.panel_id == -1, "Cartesian geometry must use panel_id == -1");
    check(layout.local_total_nx() == grid.get_local_total_points_x(), "Geometry local_total_nx must match Grid");
    check(layout.local_total_ny() == grid.get_local_total_points_y(), "Geometry local_total_ny must match Grid");
}

void test_device_values(const Grid& grid) {
    const auto& geometry = grid.geometry();

    const auto t = geometry.device_view(HorizontalLocation::T);
    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);
    const auto z = geometry.device_view(HorizontalLocation::Z);

    Kokkos::View<VVM::Real*> results("grid_geometry_results", 20);

    const int h = grid.get_halo_cells();

    Kokkos::parallel_for(
        "EvaluateGridGeometry",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            results(0) = t.q1(h, h);
            results(1) = t.q2(h, h);
            results(2) = u.q1(h, h);
            results(3) = u.q2(h, h);
            results(4) = v.q1(h, h);
            results(5) = v.q2(h, h);
            results(6) = z.q1(h, h);
            results(7) = z.q2(h, h);

            results(8) = t.sqrt_g(h, h);
            results(9) = t.inv_sqrt_g(h, h);
            results(10) = t.g_cov.a11(h, h);
            results(11) = t.g_cov.a12(h, h);
            results(12) = t.g_cov.a22(h, h);
            results(13) = t.sqrt_g_g_contra.a11(h, h);
            results(14) = t.sqrt_g_g_contra.a12(h, h);
            results(15) = t.sqrt_g_g_contra.a22(h, h);

            results(16) = t.longitude(h, h);
            results(17) = t.latitude(h, h);
            results(18) = t.dq1;
            results(19) = t.dq2;
        });

    const auto host_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);

    const VVM::Real dx = grid.get_dx();
    const VVM::Real dy = grid.get_dy();
    const VVM::Real start_i = static_cast<VVM::Real>(grid.get_local_physical_start_x());
    const VVM::Real start_j = static_cast<VVM::Real>(grid.get_local_physical_start_y());

    check(close(host_results(0), start_i * dx), "T q1 must match the local global x coordinate");
    check(close(host_results(1), start_j * dy), "T q2 must match the local global y coordinate");
    check(close(host_results(2), (start_i + VVM::real(0.5)) * dx), "U q1 must have a half-cell x offset");
    check(close(host_results(3), start_j * dy), "U q2 must not have a y offset");
    check(close(host_results(4), start_i * dx), "V q1 must not have an x offset");
    check(close(host_results(5), (start_j + VVM::real(0.5)) * dy), "V q2 must have a half-cell y offset");
    check(close(host_results(6), (start_i + VVM::real(0.5)) * dx), "Z q1 must have a half-cell x offset");
    check(close(host_results(7), (start_j + VVM::real(0.5)) * dy), "Z q2 must have a half-cell y offset");

    check(close(host_results(8), VVM::real(1.0)), "Cartesian sqrt_g must equal one");
    check(close(host_results(9), VVM::real(1.0)), "Cartesian inv_sqrt_g must equal one");
    check(close(host_results(10), VVM::real(1.0)), "Cartesian g_cov.a11 must equal one");
    check(close(host_results(11), VVM::real(0.0)), "Cartesian g_cov.a12 must equal zero");
    check(close(host_results(12), VVM::real(1.0)), "Cartesian g_cov.a22 must equal one");
    check(close(host_results(13), VVM::real(1.0)), "Cartesian sqrt_g_g_contra.a11 must equal one");
    check(close(host_results(14), VVM::real(0.0)), "Cartesian sqrt_g_g_contra.a12 must equal zero");
    check(close(host_results(15), VVM::real(1.0)), "Cartesian sqrt_g_g_contra.a22 must equal one");

    check(close(host_results(16), VVM::real(0.0)), "Cartesian geometry longitude must remain zero");
    check(close(host_results(17), VVM::real(0.0)), "Cartesian geometry latitude must remain zero");
    check(close(host_results(18), dx), "Device geometry dq1 must equal Grid dx");
    check(close(host_results(19), dy), "Device geometry dq2 must equal Grid dy");
}

void expect_unimplemented_geometry(
    const GeometryKind kind,
    const Grid& grid,
    const char* message) {

    HorizontalGridSpec spec;
    spec.kind = kind;
    spec.dq1 = grid.get_dx();
    spec.dq2 = grid.get_dy();

    try {
        auto unused = HorizontalGeometryFactory::create(spec, grid.geometry().layout());
        (void)unused;
        check(false, message);
    } catch (const std::invalid_argument&) {
    }
}

void test_unimplemented_geometry_rejection(const Grid& grid) {
    expect_unimplemented_geometry(
        GeometryKind::RegularLatLon,
        grid,
        "Factory must reject regular latitude-longitude until it is implemented");

    expect_unimplemented_geometry(
        GeometryKind::CubedSphere,
        grid,
        "Factory must reject cubed sphere until it is implemented");
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

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));

    {
        try {
            const VVM::Utils::ConfigurationManager config(argv[1]);
            const Grid grid(config, MPI_COMM_WORLD);

            test_host_metadata(grid);
            test_device_values(grid);
            test_unimplemented_geometry_rejection(grid);
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(stderr, "Rank %d unexpected exception: %s\n", mpi_rank, error.what());
        }
    }

    Kokkos::finalize();

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        if (global_failures == 0) {
            std::fprintf(stdout, "test_grid_geometry: PASS\n");
        } else {
            std::fprintf(stderr, "test_grid_geometry: %d failure(s)\n", global_failures);
        }
    }

    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
