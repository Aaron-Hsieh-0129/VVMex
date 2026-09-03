#include "core/geometry/CartesianGeometry.hpp"
#include "dynamics/operators/HorizontalCurl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>

#include <Kokkos_Core.hpp>

namespace {

using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::GeometryField2D;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Dynamics::Operators::CovariantVectorStencilAtZ;
using VVM::Dynamics::Operators::HorizontalCurlDeviceView;
using VVM::Dynamics::Operators::load_covariant_vector_stencil_at_z;
using VVM::Dynamics::Operators::make_horizontal_curl_device_view;

int failures = 0;

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

bool close(
    const VVM::Real actual,
    const VVM::Real expected,
    const VVM::Real tolerance = VVM::real(1.0e-5)) {

    return std::abs(actual - expected) <= tolerance;
}

HorizontalDomainLayout make_layout(
    const int nx,
    const int ny,
    const int halo = 2) {

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

void test_explicit_jacobian_weighting() {
    HorizontalCurlDeviceView curl;

    curl.z.inv_sqrt_g =
        GeometryField2D::constant_value(VVM::real(0.25));
    curl.z.dq1 = VVM::real(2.0);
    curl.z.dq2 = VVM::real(4.0);

    CovariantVectorStencilAtZ vector;

    vector.q1_at_u_j_i = VVM::real(3.0);
    vector.q1_at_u_jp1_i = VVM::real(11.0);
    vector.q2_at_v_j_i = VVM::real(1.0);
    vector.q2_at_v_j_ip1 = VVM::real(9.0);

    Kokkos::View<VVM::Real> result("jacobian_weighted_curl");

    Kokkos::parallel_for(
        "EvaluateJacobianWeightedCurl",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result() = curl.calculate_at_z(0, 0, vector);
        });

    VVM::Real result_host = VVM::real(0.0);
    Kokkos::deep_copy(result_host, result);

    const VVM::Real expected =
        VVM::real(0.25) *
        ((VVM::real(9.0) - VVM::real(1.0)) / VVM::real(2.0) -
         (VVM::real(11.0) - VVM::real(3.0)) / VVM::real(4.0));

    check(
        close(result_host, expected),
        "Horizontal curl must apply the Z-point inverse Jacobian");
}

void test_constant_covariant_vector() {
    const HorizontalDomainLayout layout = make_layout(12, 10);
    const CartesianGeometry geometry(
        layout,
        VVM::real(500.0),
        VVM::real(750.0));

    const auto curl = make_horizontal_curl_device_view(geometry);

    CovariantVectorStencilAtZ vector;

    vector.q1_at_u_j_i = VVM::real(4.0);
    vector.q1_at_u_jp1_i = VVM::real(4.0);
    vector.q2_at_v_j_i = VVM::real(-3.0);
    vector.q2_at_v_j_ip1 = VVM::real(-3.0);

    Kokkos::View<VVM::Real> result("constant_vector_curl");

    Kokkos::parallel_for(
        "EvaluateConstantVectorCurl",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result() =
                curl.calculate_at_z(layout.halo, layout.halo, vector);
        });

    VVM::Real result_host = VVM::real(0.0);
    Kokkos::deep_copy(result_host, result);

    check(
        close(result_host, VVM::real(0.0)),
        "A constant Cartesian vector must have zero curl");
}

void test_cartesian_equivalence() {
    const HorizontalDomainLayout layout = make_layout(18, 14);
    const VVM::Real dx = VVM::real(2.0);
    const VVM::Real dy = VVM::real(3.0);

    const CartesianGeometry geometry(layout, dx, dy);
    const auto curl = make_horizontal_curl_device_view(geometry);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> q1_at_u(
        "cartesian_covariant_q1_at_u", ny, nx);
    Kokkos::View<VVM::Real**> q2_at_v(
        "cartesian_covariant_q2_at_v", ny, nx);
    Kokkos::View<VVM::Real**> actual(
        "cartesian_curl", ny, nx);
    Kokkos::View<VVM::Real**> cartesian_reference(
        "cartesian_curl_reference", ny, nx);

    Kokkos::parallel_for(
        "InitializeCartesianCurlVector",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real i_value =
                static_cast<VVM::Real>(i - h);
            const VVM::Real j_value =
                static_cast<VVM::Real>(j - h);

            q1_at_u(j, i) =
                VVM::real(0.5) * i_value -
                VVM::real(0.25) * j_value * j_value;

            q2_at_v(j, i) =
                VVM::real(0.125) * i_value * i_value +
                VVM::real(0.75) * j_value;
        });

    Kokkos::parallel_for(
        "CompareCartesianCurl",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {h, h},
            {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            actual(j, i) =
                curl.calculate_at_z(q1_at_u, q2_at_v, j, i);

            cartesian_reference(j, i) =
                (q2_at_v(j, i + 1) - q2_at_v(j, i)) / dx -
                (q1_at_u(j + 1, i) - q1_at_u(j, i)) / dy;
        });

    const auto actual_host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            actual);

    const auto reference_host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            cartesian_reference);

    VVM::Real maximum_difference = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            maximum_difference = std::max(
                maximum_difference,
                std::abs(
                    actual_host(j, i) -
                    reference_host(j, i)));
        }
    }

    check(
        maximum_difference <= VVM::real(1.0e-6),
        "Generalized curl must equal the Cartesian dv/dx minus du/dy formula");
}

void test_solid_body_rotation() {
    const HorizontalDomainLayout layout = make_layout(16, 12);
    const VVM::Real dx = VVM::real(100.0);
    const VVM::Real dy = VVM::real(200.0);
    const VVM::Real angular_velocity = VVM::real(0.002);

    const CartesianGeometry geometry(layout, dx, dy);
    const auto curl = make_horizontal_curl_device_view(geometry);
    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> q1_at_u(
        "solid_body_q1_at_u", ny, nx);
    Kokkos::View<VVM::Real**> q2_at_v(
        "solid_body_q2_at_v", ny, nx);
    Kokkos::View<VVM::Real**> result(
        "solid_body_curl", ny, nx);

    Kokkos::parallel_for(
        "InitializeSolidBodyRotation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            q1_at_u(j, i) =
                -angular_velocity * u.q2(j, i);

            q2_at_v(j, i) =
                angular_velocity * v.q1(j, i);
        });

    Kokkos::parallel_for(
        "EvaluateSolidBodyRotationCurl",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {h, h},
            {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            result(j, i) =
                curl.calculate_at_z(q1_at_u, q2_at_v, j, i);
        });

    const auto result_host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            result);

    const VVM::Real expected =
        VVM::real(2.0) * angular_velocity;

    VVM::Real maximum_error = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            maximum_error = std::max(
                maximum_error,
                std::abs(result_host(j, i) - expected));
        }
    }

    check(
        maximum_error <= VVM::real(1.0e-6),
        "Solid-body rotation must have vertical curl equal to twice its angular velocity");
}

void test_three_dimensional_stencil_loader() {
    Kokkos::View<VVM::Real***> q1_at_u(
        "three_dimensional_q1_at_u", 2, 3, 3);
    Kokkos::View<VVM::Real***> q2_at_v(
        "three_dimensional_q2_at_v", 2, 3, 3);
    Kokkos::View<VVM::Real*> result(
        "three_dimensional_curl_stencil", 4);

    Kokkos::parallel_for(
        "InitializeThreeDimensionalCurlVector",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {2, 3, 3}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            q1_at_u(k, j, i) =
                VVM::real(100.0) * static_cast<VVM::Real>(k) +
                VVM::real(10.0) * static_cast<VVM::Real>(j) +
                static_cast<VVM::Real>(i);

            q2_at_v(k, j, i) =
                VVM::real(1000.0) * static_cast<VVM::Real>(k) +
                VVM::real(100.0) * static_cast<VVM::Real>(j) +
                VVM::real(10.0) * static_cast<VVM::Real>(i);
        });

    Kokkos::parallel_for(
        "LoadThreeDimensionalCurlStencil",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            const CovariantVectorStencilAtZ stencil =
                load_covariant_vector_stencil_at_z(
                    q1_at_u,
                    q2_at_v,
                    1,
                    1,
                    1);

            result(0) = stencil.q1_at_u_j_i;
            result(1) = stencil.q1_at_u_jp1_i;
            result(2) = stencil.q2_at_v_j_i;
            result(3) = stencil.q2_at_v_j_ip1;
        });

    const auto result_host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            result);

    check(
        close(result_host(0), VVM::real(111.0)),
        "The 3-D loader must load q1 at U(j,i)");

    check(
        close(result_host(1), VVM::real(121.0)),
        "The 3-D loader must load q1 at U(j+1,i)");

    check(
        close(result_host(2), VVM::real(1110.0)),
        "The 3-D loader must load q2 at V(j,i)");

    check(
        close(result_host(3), VVM::real(1120.0)),
        "The 3-D loader must load q2 at V(j,i+1)");
}

VVM::Real periodic_curl_error(const int n) {
    const VVM::Real pi = std::acos(VVM::real(-1.0));
    const VVM::Real domain_length =
        VVM::real(2.0) * pi;
    const VVM::Real spacing =
        domain_length / static_cast<VVM::Real>(n);

    const HorizontalDomainLayout layout =
        make_layout(n, n);

    const CartesianGeometry geometry(
        layout,
        spacing,
        spacing);

    const auto curl =
        make_horizontal_curl_device_view(geometry);

    const auto u =
        geometry.device_view(HorizontalLocation::U);
    const auto v =
        geometry.device_view(HorizontalLocation::V);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> q1_at_u(
        "periodic_curl_q1_at_u", ny, nx);
    Kokkos::View<VVM::Real**> q2_at_v(
        "periodic_curl_q2_at_v", ny, nx);
    Kokkos::View<VVM::Real**> result(
        "periodic_curl_result", ny, nx);

    Kokkos::parallel_for(
        "InitializePeriodicCurlVector",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            q1_at_u(j, i) =
                Kokkos::sin(u.q2(j, i));

            q2_at_v(j, i) =
                Kokkos::cos(v.q1(j, i));
        });

    Kokkos::parallel_for(
        "EvaluatePeriodicCurl",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {h, h},
            {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            result(j, i) =
                curl.calculate_at_z(q1_at_u, q2_at_v, j, i);
        });

    const auto result_host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            result);

    VVM::Real squared_error = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            const VVM::Real x =
                (static_cast<VVM::Real>(i - h) +
                 VVM::real(0.5)) * spacing;

            const VVM::Real y =
                (static_cast<VVM::Real>(j - h) +
                 VVM::real(0.5)) * spacing;

            const VVM::Real expected =
                -std::sin(x) - std::cos(y);

            const VVM::Real error =
                result_host(j, i) - expected;

            squared_error += error * error;
        }
    }

    return std::sqrt(
        squared_error /
        static_cast<VVM::Real>(n * n));
}

void test_second_order_convergence() {
    const VVM::Real coarse_error =
        periodic_curl_error(32);
    const VVM::Real fine_error =
        periodic_curl_error(64);
    const VVM::Real error_ratio =
        coarse_error / fine_error;

    check(
        coarse_error > fine_error,
        "Refining the Cartesian grid must reduce the curl error");

    check(
        error_ratio > VVM::real(3.5),
        "The centered U/V-to-Z curl must show second-order convergence");
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    {
        try {
            test_explicit_jacobian_weighting();
            test_constant_covariant_vector();
            test_cartesian_equivalence();
            test_solid_body_rotation();
            test_three_dimensional_stencil_loader();
            test_second_order_convergence();
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(
                stderr,
                "Unexpected exception: %s\n",
                error.what());
        }
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::fprintf(
            stdout,
            "test_horizontal_curl: PASS\n");
    } else {
        std::fprintf(
            stderr,
            "test_horizontal_curl: %d failure(s)\n",
            failures);
    }

    return failures == 0 ? 0 : 1;
}
