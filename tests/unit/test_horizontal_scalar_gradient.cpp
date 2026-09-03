#include "core/geometry/CartesianGeometry.hpp"
#include "dynamics/operators/HorizontalScalarGradient.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>

#include <Kokkos_Core.hpp>

namespace {

using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::GeometryField2D;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometryDeviceView;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Dynamics::Operators::HorizontalScalarGradientDeviceView;
using VVM::Dynamics::Operators::ScalarStencilAtT;
using VVM::Dynamics::Operators::load_scalar_stencil_at_t;
using VVM::Dynamics::Operators::make_horizontal_scalar_gradient_device_view;

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

void set_constant_inverse_metric(
    HorizontalGeometryDeviceView& geometry,
    const VVM::Real g11,
    const VVM::Real g12,
    const VVM::Real g22) {

    geometry.inv_sqrt_g = GeometryField2D::constant_value(VVM::real(1.0));
    geometry.sqrt_g_g_contra.a11 = GeometryField2D::constant_value(g11);
    geometry.sqrt_g_g_contra.a12 = GeometryField2D::constant_value(g12);
    geometry.sqrt_g_g_contra.a22 = GeometryField2D::constant_value(g22);
}

ScalarStencilAtT make_affine_stencil(
    const VVM::Real slope_q1,
    const VVM::Real slope_q2,
    const VVM::Real dq1,
    const VVM::Real dq2) {

    ScalarStencilAtT scalar;

    scalar.center = VVM::real(0.0);
    scalar.west = -slope_q1 * dq1;
    scalar.east = slope_q1 * dq1;
    scalar.south = -slope_q2 * dq2;
    scalar.north = slope_q2 * dq2;
    scalar.southwest = -slope_q1 * dq1 - slope_q2 * dq2;
    scalar.southeast = slope_q1 * dq1 - slope_q2 * dq2;
    scalar.northwest = -slope_q1 * dq1 + slope_q2 * dq2;
    scalar.northeast = slope_q1 * dq1 + slope_q2 * dq2;

    return scalar;
}

void test_cross_metric_affine_gradient() {
    const VVM::Real dq1 = VVM::real(2.0);
    const VVM::Real dq2 = VVM::real(3.0);
    const VVM::Real slope_q1 = VVM::real(1.25);
    const VVM::Real slope_q2 = VVM::real(-0.75);

    HorizontalScalarGradientDeviceView gradient;

    gradient.u.dq1 = dq1;
    gradient.u.dq2 = dq2;
    gradient.v.dq1 = dq1;
    gradient.v.dq2 = dq2;

    set_constant_inverse_metric(
        gradient.u,
        VVM::real(2.0),
        VVM::real(0.5),
        VVM::real(1.5));

    set_constant_inverse_metric(
        gradient.v,
        VVM::real(2.0),
        VVM::real(0.5),
        VVM::real(1.5));

    const ScalarStencilAtT scalar = make_affine_stencil(slope_q1, slope_q2, dq1, dq2);

    Kokkos::View<VVM::Real*> result("cross_metric_gradient", 2);

    Kokkos::parallel_for(
        "EvaluateCrossMetricGradient",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            const auto value = gradient.at_positive_faces(0, 0, scalar);

            result(0) = value.q1_at_u;
            result(1) = value.q2_at_v;
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    const VVM::Real expected_q1 = VVM::real(2.0) * slope_q1 + VVM::real(0.5) * slope_q2;
    const VVM::Real expected_q2 = VVM::real(0.5) * slope_q1 + VVM::real(1.5) * slope_q2;

    check(close(result_host(0), expected_q1), "The U gradient component must include g11 and g12");
    check(close(result_host(1), expected_q2), "The V gradient component must include g12 and g22");
}

void test_constant_cartesian_scalar() {
    const HorizontalDomainLayout layout = make_layout(12, 10);
    const CartesianGeometry geometry(layout, VVM::real(500.0), VVM::real(750.0));
    const auto gradient = make_horizontal_scalar_gradient_device_view(geometry);

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

    Kokkos::View<VVM::Real*> result("constant_scalar_gradient", 2);

    Kokkos::parallel_for(
        "EvaluateConstantScalarGradient",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            const auto value = gradient.at_positive_faces(layout.halo, layout.halo, scalar);

            result(0) = value.q1_at_u;
            result(1) = value.q2_at_v;
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    check(close(result_host(0), VVM::real(0.0)), "A constant scalar must have zero q1 gradient");
    check(close(result_host(1), VVM::real(0.0)), "A constant scalar must have zero q2 gradient");
}

void test_cartesian_affine_scalar() {
    const HorizontalDomainLayout layout = make_layout(16, 12);
    const VVM::Real dx = VVM::real(100.0);
    const VVM::Real dy = VVM::real(200.0);
    const VVM::Real slope_x = VVM::real(1.25);
    const VVM::Real slope_y = VVM::real(-0.75);

    const CartesianGeometry geometry(layout, dx, dy);
    const auto gradient = make_horizontal_scalar_gradient_device_view(geometry);
    const auto t = geometry.device_view(HorizontalLocation::T);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> scalar("affine_scalar", ny, nx);
    Kokkos::View<VVM::Real**> gradient_q1("affine_gradient_q1", ny, nx);
    Kokkos::View<VVM::Real**> gradient_q2("affine_gradient_q2", ny, nx);

    Kokkos::parallel_for(
        "InitializeAffineScalar",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            scalar(j, i) = slope_x * t.q1(j, i) + slope_y * t.q2(j, i) + VVM::real(3.0);
        });

    Kokkos::parallel_for(
        "EvaluateAffineScalarGradient",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const ScalarStencilAtT stencil = load_scalar_stencil_at_t(scalar, j, i);
            const auto value = gradient.at_positive_faces(j, i, stencil);

            gradient_q1(j, i) = value.q1_at_u;
            gradient_q2(j, i) = value.q2_at_v;
        });

    const auto q1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gradient_q1);
    const auto q2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gradient_q2);

    VVM::Real maximum_q1_error = VVM::real(0.0);
    VVM::Real maximum_q2_error = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            maximum_q1_error = std::max(maximum_q1_error, std::abs(q1_host(j, i) - slope_x));
            maximum_q2_error = std::max(maximum_q2_error, std::abs(q2_host(j, i) - slope_y));
        }
    }

    check(maximum_q1_error <= VVM::real(1.0e-5), "Cartesian q1 gradient must reproduce an affine x derivative");
    check(maximum_q2_error <= VVM::real(1.0e-5), "Cartesian q2 gradient must reproduce an affine y derivative");
}

void test_three_dimensional_stencil_loader() {
    Kokkos::View<VVM::Real***> scalar("three_dimensional_scalar", 2, 3, 3);
    Kokkos::View<VVM::Real*> result("three_dimensional_stencil", 9);

    Kokkos::parallel_for(
        "InitializeThreeDimensionalScalar",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {2, 3, 3}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            scalar(k, j, i) =
                VVM::real(100.0) * static_cast<VVM::Real>(k) +
                VVM::real(10.0) * static_cast<VVM::Real>(j) +
                static_cast<VVM::Real>(i);
        });

    Kokkos::parallel_for(
        "LoadThreeDimensionalStencil",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            const ScalarStencilAtT stencil = load_scalar_stencil_at_t(scalar, 1, 1, 1);

            result(0) = stencil.center;
            result(1) = stencil.west;
            result(2) = stencil.east;
            result(3) = stencil.south;
            result(4) = stencil.north;
            result(5) = stencil.southwest;
            result(6) = stencil.southeast;
            result(7) = stencil.northwest;
            result(8) = stencil.northeast;
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    check(close(result_host(0), VVM::real(111.0)), "3-D stencil center must be loaded correctly");
    check(close(result_host(1), VVM::real(110.0)), "3-D stencil west value must be loaded correctly");
    check(close(result_host(2), VVM::real(112.0)), "3-D stencil east value must be loaded correctly");
    check(close(result_host(3), VVM::real(101.0)), "3-D stencil south value must be loaded correctly");
    check(close(result_host(4), VVM::real(121.0)), "3-D stencil north value must be loaded correctly");
    check(close(result_host(5), VVM::real(100.0)), "3-D stencil southwest value must be loaded correctly");
    check(close(result_host(6), VVM::real(102.0)), "3-D stencil southeast value must be loaded correctly");
    check(close(result_host(7), VVM::real(120.0)), "3-D stencil northwest value must be loaded correctly");
    check(close(result_host(8), VVM::real(122.0)), "3-D stencil northeast value must be loaded correctly");
}

VVM::Real periodic_gradient_error(const int n) {
    const VVM::Real pi = std::acos(VVM::real(-1.0));
    const VVM::Real domain_length = VVM::real(2.0) * pi;
    const VVM::Real spacing = domain_length / static_cast<VVM::Real>(n);

    const HorizontalDomainLayout layout = make_layout(n, n);
    const CartesianGeometry geometry(layout, spacing, spacing);
    const auto gradient = make_horizontal_scalar_gradient_device_view(geometry);
    const auto t = geometry.device_view(HorizontalLocation::T);
    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> scalar("periodic_gradient_scalar", ny, nx);
    Kokkos::View<VVM::Real**> gradient_q1("periodic_gradient_q1", ny, nx);
    Kokkos::View<VVM::Real**> gradient_q2("periodic_gradient_q2", ny, nx);

    Kokkos::parallel_for(
        "InitializePeriodicGradientScalar",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            scalar(j, i) = Kokkos::sin(t.q1(j, i)) + Kokkos::cos(t.q2(j, i));
        });

    Kokkos::parallel_for(
        "EvaluatePeriodicGradient",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const ScalarStencilAtT stencil = load_scalar_stencil_at_t(scalar, j, i);
            const auto value = gradient.at_positive_faces(j, i, stencil);

            gradient_q1(j, i) = value.q1_at_u;
            gradient_q2(j, i) = value.q2_at_v;
        });

    const auto q1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gradient_q1);
    const auto q2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gradient_q2);
    const auto u_q1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u.q1.one_dimensional);
    const auto v_q2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.q2.one_dimensional);

    VVM::Real squared_error = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            const VVM::Real expected_q1 = std::cos(u_q1_host(i));
            const VVM::Real expected_q2 = -std::sin(v_q2_host(j));
            const VVM::Real q1_error = q1_host(j, i) - expected_q1;
            const VVM::Real q2_error = q2_host(j, i) - expected_q2;

            squared_error += q1_error * q1_error + q2_error * q2_error;
        }
    }

    return std::sqrt(squared_error / static_cast<VVM::Real>(2 * n * n));
}

void test_second_order_convergence() {
    const VVM::Real coarse_error = periodic_gradient_error(32);
    const VVM::Real fine_error = periodic_gradient_error(64);
    const VVM::Real error_ratio = coarse_error / fine_error;

    check(coarse_error > fine_error, "Refining the Cartesian grid must reduce the gradient error");
    check(error_ratio > VVM::real(3.5), "The centered face gradient must show second-order convergence");
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    {
        try {
            test_cross_metric_affine_gradient();
            test_constant_cartesian_scalar();
            test_cartesian_affine_scalar();
            test_three_dimensional_stencil_loader();
            test_second_order_convergence();
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
        }
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::fprintf(stdout, "test_horizontal_scalar_gradient: PASS\n");
    } else {
        std::fprintf(stderr, "test_horizontal_scalar_gradient: %d failure(s)\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
