#include "core/Grid.hpp"
#include "core/geometry/HorizontalGeometryFactory.hpp"
#include "utils/ConfigurationManager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace {

using VVM::Core::Grid;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::HorizontalGeometryFactory;
using VVM::Core::Geometry::HorizontalGridSpec;
using VVM::Core::Geometry::HorizontalLocation;

int failures = 0;
int mpi_rank = 0;

constexpr VVM::Real relative_tolerance =
    sizeof(VVM::Real) == sizeof(double) ? VVM::real(1.0e-12) : VVM::real(5.0e-5);

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "Rank %d FAIL: %s\n", mpi_rank, message);
}

bool close(const VVM::Real actual, const VVM::Real expected) {
    const VVM::Real scale = std::max(VVM::real(1.0), std::abs(expected));
    return std::abs(actual - expected) <= relative_tolerance * scale;
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

void test_cartesian_device_values(const Grid& grid) {
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
    check(close(host_results(13), VVM::real(1.0)), "Cartesian J g^11 must equal one");
    check(close(host_results(14), VVM::real(0.0)), "Cartesian J g^12 must equal zero");
    check(close(host_results(15), VVM::real(1.0)), "Cartesian J g^22 must equal one");

    check(close(host_results(16), VVM::real(0.0)), "Cartesian longitude must remain zero");
    check(close(host_results(17), VVM::real(0.0)), "Cartesian latitude must remain zero");
    check(close(host_results(18), dx), "Device geometry dq1 must equal Grid dx");
    check(close(host_results(19), dy), "Device geometry dq2 must equal Grid dy");
}

HorizontalGridSpec make_regular_lat_lon_spec(const Grid& grid) {
    const VVM::Real pi = std::acos(VVM::real(-1.0));

    HorizontalGridSpec spec;
    spec.kind = GeometryKind::RegularLatLon;
    spec.dq1 = VVM::real(2.0) * pi / static_cast<VVM::Real>(grid.get_global_points_x());
    spec.dq2 = (pi / VVM::real(2.0)) / static_cast<VVM::Real>(grid.get_global_points_y());
    spec.regular_lat_lon.longitude_west_edge = VVM::real(0.0);
    spec.regular_lat_lon.latitude_south_edge = -pi / VVM::real(4.0);
    spec.regular_lat_lon.radius = VVM::real(6371220.0);

    return spec;
}

void test_regular_lat_lon_factory(const Grid& grid) {
    const HorizontalGridSpec spec = make_regular_lat_lon_spec(grid);
    const auto geometry = HorizontalGeometryFactory::create(spec, grid.geometry().layout());

    check(geometry != nullptr, "Factory must return a regular latitude-longitude geometry");
    check(geometry->kind() == GeometryKind::RegularLatLon, "Factory geometry must report RegularLatLon");
    check(std::strcmp(geometry->name(), "regular_latlon") == 0, "Factory geometry must report the RLL name");
    check(close(geometry->dq1(), spec.dq1), "Factory must forward the longitude increment");
    check(close(geometry->dq2(), spec.dq2), "Factory must forward the latitude increment");

    const auto t = geometry->device_view(HorizontalLocation::T);
    const auto u = geometry->device_view(HorizontalLocation::U);
    const auto v = geometry->device_view(HorizontalLocation::V);

    Kokkos::View<VVM::Real*> results("regular_lat_lon_factory_results", 6);

    const int h = grid.get_halo_cells();

    Kokkos::parallel_for(
        "EvaluateRegularLatLonFactoryGeometry",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            results(0) = t.longitude(h, h);
            results(1) = t.latitude(h, h);
            results(2) = u.longitude(h, h);
            results(3) = v.latitude(h, h);
            results(4) = t.sqrt_g(h, h);
            results(5) = t.g_cov.a22(h, h);
        });

    const auto host_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);

    const VVM::Real start_i = static_cast<VVM::Real>(grid.get_local_physical_start_x());
    const VVM::Real start_j = static_cast<VVM::Real>(grid.get_local_physical_start_y());
    const VVM::Real expected_t_longitude =
        spec.regular_lat_lon.longitude_west_edge + (start_i + VVM::real(0.5)) * spec.dq1;
    const VVM::Real expected_t_latitude =
        spec.regular_lat_lon.latitude_south_edge + (start_j + VVM::real(0.5)) * spec.dq2;
    const VVM::Real expected_u_longitude =
        spec.regular_lat_lon.longitude_west_edge + (start_i + VVM::real(1.0)) * spec.dq1;
    const VVM::Real expected_v_latitude =
        spec.regular_lat_lon.latitude_south_edge + (start_j + VVM::real(1.0)) * spec.dq2;
    const VVM::Real radius_squared = spec.regular_lat_lon.radius * spec.regular_lat_lon.radius;

    check(close(host_results(0), expected_t_longitude), "Factory RLL T longitude must use the rank's global start");
    check(close(host_results(1), expected_t_latitude), "Factory RLL T latitude must use the rank's global start");
    check(close(host_results(2), expected_u_longitude), "Factory RLL U longitude must use edge staggering");
    check(close(host_results(3), expected_v_latitude), "Factory RLL V latitude must use edge staggering");
    check(
        close(host_results(4), radius_squared * std::cos(expected_t_latitude)),
        "Factory must forward the Earth radius into the RLL Jacobian");
    check(close(host_results(5), radius_squared), "Factory must forward the Earth radius into g22");
}

void expect_factory_failure(
    const HorizontalGridSpec& spec,
    const Grid& grid,
    const char* message) {

    try {
        auto unused = HorizontalGeometryFactory::create(spec, grid.geometry().layout());
        (void)unused;
        check(false, message);
    } catch (const std::invalid_argument&) {
    }
}

void test_factory_rejections(const Grid& grid) {
    HorizontalGridSpec incomplete_regular_lat_lon = make_regular_lat_lon_spec(grid);
    incomplete_regular_lat_lon.regular_lat_lon.radius = VVM::real(0.0);

    expect_factory_failure(
        incomplete_regular_lat_lon,
        grid,
        "Factory must reject regular latitude-longitude geometry without a positive radius");

    HorizontalGridSpec cubed_sphere;
    cubed_sphere.kind = GeometryKind::CubedSphere;
    cubed_sphere.dq1 = grid.get_dx();
    cubed_sphere.dq2 = grid.get_dy();

    expect_factory_failure(
        cubed_sphere,
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
            test_cartesian_device_values(grid);
            test_regular_lat_lon_factory(grid);
            test_factory_rejections(grid);
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
