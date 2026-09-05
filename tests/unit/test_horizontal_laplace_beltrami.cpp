#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/HorizontalLaplaceBeltrami.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>

#include <Kokkos_Core.hpp>

namespace {

using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::GeometryField2D;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometryDeviceView;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::Operators::HorizontalLaplaceBeltramiDeviceView;
using VVM::Dynamics::Operators::ScalarStencilAtT;
using VVM::Dynamics::Operators::make_horizontal_laplace_beltrami_device_view;

int failures = 0;

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

bool close(const VVM::Real actual, const VVM::Real expected, const VVM::Real tolerance = VVM::real(1.0e-5)) {
    return std::abs(actual - expected) <= tolerance;
}

HorizontalDomainLayout make_layout(const int nx, const int ny, const int halo = 2) {
    HorizontalDomainLayout layout;

    layout.global_nx = nx;
    layout.global_ny = ny;
    layout.local_physical_nx = nx;
    layout.local_physical_ny = ny;
    layout.global_start_i = 0;
    layout.global_start_j = 0;
    layout.halo = halo;
    layout.panel_id = -1;

    return layout;
}

void set_constant_geometry(
    HorizontalGeometryDeviceView& geometry,
    const VVM::Real dq1,
    const VVM::Real dq2,
    const VVM::Real sqrt_g,
    const VVM::Real g11,
    const VVM::Real g12,
    const VVM::Real g22) {

    geometry.dq1 = dq1;
    geometry.dq2 = dq2;
    geometry.sqrt_g = GeometryField2D::constant_value(sqrt_g);
    geometry.inv_sqrt_g = GeometryField2D::constant_value(VVM::real(1.0) / sqrt_g);
    geometry.sqrt_g_g_contra.a11 = GeometryField2D::constant_value(sqrt_g * g11);
    geometry.sqrt_g_g_contra.a12 = GeometryField2D::constant_value(sqrt_g * g12);
    geometry.sqrt_g_g_contra.a22 = GeometryField2D::constant_value(sqrt_g * g22);
}

HorizontalLaplaceBeltramiDeviceView make_constant_metric_laplacian(
    const VVM::Real dq1,
    const VVM::Real dq2,
    const VVM::Real sqrt_g,
    const VVM::Real g11,
    const VVM::Real g12,
    const VVM::Real g22) {

    HorizontalGeometryDeviceView geometry;
    set_constant_geometry(geometry, dq1, dq2, sqrt_g, g11, g12, g22);

    HorizontalLaplaceBeltramiDeviceView laplacian;

    laplacian.gradient.u = geometry;
    laplacian.gradient.v = geometry;
    laplacian.divergence.t = geometry;
    laplacian.divergence.u = geometry;
    laplacian.divergence.v = geometry;

    return laplacian;
}

void test_constant_scalar() {
    const HorizontalDomainLayout layout = make_layout(12, 10);
    const CartesianGeometry geometry(layout, VVM::real(500.0), VVM::real(750.0));
    const auto laplacian = make_horizontal_laplace_beltrami_device_view(geometry);

    ScalarStencilAtT scalar;

    scalar.center = VVM::real(4.0);
    scalar.west = VVM::real(4.0);
    scalar.east = VVM::real(4.0);
    scalar.south = VVM::real(4.0);
    scalar.north = VVM::real(4.0);
    scalar.southwest = VVM::real(4.0);
    scalar.southeast = VVM::real(4.0);
    scalar.northwest = VVM::real(4.0);
    scalar.northeast = VVM::real(4.0);

    Kokkos::View<VVM::Real> result("constant_scalar_laplacian");

    Kokkos::parallel_for(
        "EvaluateConstantScalarLaplacian",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result() = laplacian.calculate_at_t(layout.halo, layout.halo, scalar);
        });

    VVM::Real result_host = VVM::real(0.0);
    Kokkos::deep_copy(result_host, result);

    check(close(result_host, VVM::real(0.0)), "A constant scalar must have zero Laplace-Beltrami value");
}

void test_cross_metric_mixed_derivative() {
    const VVM::Real dq1 = VVM::real(1.5);
    const VVM::Real dq2 = VVM::real(2.0);
    const VVM::Real g12 = VVM::real(0.4);

    const auto laplacian = make_constant_metric_laplacian(
        dq1,
        dq2,
        VVM::real(3.0),
        VVM::real(2.0),
        g12,
        VVM::real(1.5));

    const VVM::Real corner_value = dq1 * dq2;

    ScalarStencilAtT scalar;

    scalar.center = VVM::real(0.0);
    scalar.west = VVM::real(0.0);
    scalar.east = VVM::real(0.0);
    scalar.south = VVM::real(0.0);
    scalar.north = VVM::real(0.0);
    scalar.southwest = corner_value;
    scalar.southeast = -corner_value;
    scalar.northwest = -corner_value;
    scalar.northeast = corner_value;

    Kokkos::View<VVM::Real> result("cross_metric_laplacian");

    Kokkos::parallel_for(
        "EvaluateCrossMetricLaplacian",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result() = laplacian.calculate_at_t(0, 0, scalar);
        });

    VVM::Real result_host = VVM::real(0.0);
    Kokkos::deep_copy(result_host, result);

    check(close(result_host, VVM::real(2.0) * g12), "The Laplace-Beltrami operator must include both mixed-derivative contributions");
}

void test_cartesian_five_point_equivalence() {
    const HorizontalDomainLayout layout = make_layout(18, 14);
    const VVM::Real dx = VVM::real(2.0);
    const VVM::Real dy = VVM::real(3.0);

    const CartesianGeometry geometry(layout, dx, dy);
    const auto laplacian = make_horizontal_laplace_beltrami_device_view(geometry);
    const auto t = geometry.device_view(HorizontalLocation::T);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> scalar("cartesian_laplacian_scalar", ny, nx);
    Kokkos::View<VVM::Real**> actual("cartesian_laplacian_actual", ny, nx);
    Kokkos::View<VVM::Real**> legacy("cartesian_laplacian_legacy", ny, nx);

    Kokkos::parallel_for(
        "InitializeCartesianLaplacianScalar",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real x = t.q1(j, i);
            const VVM::Real y = t.q2(j, i);

            scalar(j, i) =
                VVM::real(0.25) * x * x +
                VVM::real(0.125) * y * y +
                VVM::real(0.01) * x * y;
        });

    Kokkos::parallel_for(
        "CompareCartesianLaplacian",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            actual(j, i) = laplacian.calculate_at_t(scalar, j, i);

            legacy(j, i) =
                (scalar(j, i + 1) - VVM::real(2.0) * scalar(j, i) + scalar(j, i - 1)) / (dx * dx) +
                (scalar(j + 1, i) - VVM::real(2.0) * scalar(j, i) + scalar(j - 1, i)) / (dy * dy);
        });

    const auto actual_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
    const auto legacy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), legacy);

    VVM::Real maximum_difference = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            maximum_difference = std::max(maximum_difference, std::abs(actual_host(j, i) - legacy_host(j, i)));
        }
    }

    check(maximum_difference <= VVM::real(1.0e-5), "Cartesian Laplace-Beltrami must equal the legacy five-point Laplacian");
}

void test_quadratic_scalar() {
    const HorizontalDomainLayout layout = make_layout(16, 12);
    const CartesianGeometry geometry(layout, VVM::real(100.0), VVM::real(200.0));
    const auto laplacian = make_horizontal_laplace_beltrami_device_view(geometry);
    const auto t = geometry.device_view(HorizontalLocation::T);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> scalar("quadratic_scalar", ny, nx);
    Kokkos::View<VVM::Real**> result("quadratic_laplacian", ny, nx);

    Kokkos::parallel_for(
        "InitializeQuadraticScalar",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real x = t.q1(j, i);
            const VVM::Real y = t.q2(j, i);

            scalar(j, i) = x * x + VVM::real(3.0) * y * y;
        });

    Kokkos::parallel_for(
        "EvaluateQuadraticLaplacian",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            result(j, i) = laplacian.calculate_at_t(scalar, j, i);
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    VVM::Real maximum_error = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            maximum_error = std::max(maximum_error, std::abs(result_host(j, i) - VVM::real(8.0)));
        }
    }

    check(maximum_error <= VVM::real(1.0e-4), "The Cartesian Laplacian of x^2 + 3y^2 must equal eight");
}

VVM::Real periodic_laplacian_error(const int n) {
    const VVM::Real pi = std::acos(VVM::real(-1.0));
    const VVM::Real domain_length = VVM::real(2.0) * pi;
    const VVM::Real spacing = domain_length / static_cast<VVM::Real>(n);

    const HorizontalDomainLayout layout = make_layout(n, n);
    const CartesianGeometry geometry(layout, spacing, spacing);
    const auto laplacian = make_horizontal_laplace_beltrami_device_view(geometry);
    const auto t = geometry.device_view(HorizontalLocation::T);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> scalar("periodic_laplacian_scalar", ny, nx);
    Kokkos::View<VVM::Real**> result("periodic_laplacian_result", ny, nx);

    Kokkos::parallel_for(
        "InitializePeriodicLaplacianScalar",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            scalar(j, i) = Kokkos::sin(t.q1(j, i)) + Kokkos::cos(t.q2(j, i));
        });

    Kokkos::parallel_for(
        "EvaluatePeriodicLaplacian",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            result(j, i) = laplacian.calculate_at_t(scalar, j, i);
        });

    const auto scalar_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalar);
    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    VVM::Real squared_error = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            const VVM::Real expected = -scalar_host(j, i);
            const VVM::Real error = result_host(j, i) - expected;

            squared_error += error * error;
        }
    }

    return std::sqrt(squared_error / static_cast<VVM::Real>(n * n));
}

void test_second_order_convergence() {
    const VVM::Real coarse_error = periodic_laplacian_error(32);
    const VVM::Real fine_error = periodic_laplacian_error(64);
    const VVM::Real error_ratio = coarse_error / fine_error;

    check(coarse_error > fine_error, "Refining the Cartesian grid must reduce the Laplacian error");
    check(error_ratio > VVM::real(3.5), "Laplace-Beltrami must show second-order convergence");
}

void test_regular_latlon_diagonal_matches_center_impulse() {
    const int physical_nx = 24;
    const int physical_ny = 12;
    const HorizontalDomainLayout layout = make_layout(physical_nx, physical_ny);

    const VVM::Real pi = VVM::real(std::acos(-1.0));
    const VVM::Real dlongitude = VVM::real(2.0) * pi / static_cast<VVM::Real>(physical_nx);
    const VVM::Real dlatitude = (pi / VVM::real(2.0)) / static_cast<VVM::Real>(physical_ny);

    const RegularLatLonGeometry geometry(
        layout,
        dlongitude,
        dlatitude,
        -pi,
        -pi / VVM::real(4.0),
        VVM::real(6371220.0));

    const auto laplacian = make_horizontal_laplace_beltrami_device_view(geometry);

    const int ny = layout.local_total_ny();
    const int h = layout.halo;
    const int i = h;

    Kokkos::View<VVM::Real*> applied("regular_latlon_impulse_applied", ny);
    Kokkos::View<VVM::Real*> diagonal("regular_latlon_impulse_diagonal", ny);

    Kokkos::parallel_for(
        "CompareRegularLatLonLaplacianDiagonal",
        Kokkos::RangePolicy<>(h, ny - h),
        KOKKOS_LAMBDA(const int j) {
            ScalarStencilAtT impulse;
            impulse.center = VVM::real(1.0);

            applied(j) = laplacian.calculate_at_t(j, i, impulse);
            diagonal(j) = laplacian.diagonal_at_t(j, i);
        });

    const auto applied_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), applied);
    const auto diagonal_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonal);

    const VVM::Real relative_tolerance = VVM::real(64.0) * std::numeric_limits<VVM::Real>::epsilon();

    for (int j = h; j < ny - h; ++j) {
        const VVM::Real difference = std::abs(applied_host(j) - diagonal_host(j));
        const VVM::Real scale = std::max(std::abs(applied_host(j)), std::abs(diagonal_host(j)));

        check(
            scale == VVM::real(0.0) ? difference == VVM::real(0.0) : difference <= relative_tolerance * scale,
            "RLL diagonal must equal the Laplace-Beltrami response to a center impulse");
    }
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    {
        try {
            test_constant_scalar();
            test_cross_metric_mixed_derivative();
            test_cartesian_five_point_equivalence();
            test_quadratic_scalar();
            test_second_order_convergence();
            test_regular_latlon_diagonal_matches_center_impulse();
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
        }
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::fprintf(stdout, "test_horizontal_laplace_beltrami: PASS\n");
    } else {
        std::fprintf(stderr, "test_horizontal_laplace_beltrami: %d failure(s)\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
