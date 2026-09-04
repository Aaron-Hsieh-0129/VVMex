#include "core/geometry/CartesianGeometry.hpp"
#include "dynamics/operators/VerticalVorticity.hpp"

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
using VVM::Dynamics::Operators::VerticalVorticityDeviceView;
using VVM::Dynamics::Operators::make_vertical_vorticity_device_view;

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

void test_cartesian_equivalence() {
    const HorizontalDomainLayout layout = make_layout(18, 14);
    const VVM::Real dx = VVM::real(2.0);
    const VVM::Real dy = VVM::real(3.0);

    const CartesianGeometry geometry(layout, dx, dy);
    const auto vorticity = make_vertical_vorticity_device_view(geometry);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> contravariant_q1_at_u(
        "cartesian_vorticity_q1_at_u", ny, nx);

    Kokkos::View<VVM::Real**> contravariant_q2_at_v(
        "cartesian_vorticity_q2_at_v", ny, nx);

    Kokkos::View<VVM::Real**> actual(
        "cartesian_vertical_vorticity", ny, nx);

    Kokkos::View<VVM::Real**> cartesian_reference(
        "cartesian_vertical_vorticity_reference", ny, nx);

    Kokkos::parallel_for(
        "InitializeCartesianVorticityWind",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real i_value =
                static_cast<VVM::Real>(i - h);

            const VVM::Real j_value =
                static_cast<VVM::Real>(j - h);

            contravariant_q1_at_u(j, i) =
                VVM::real(0.5) * i_value -
                VVM::real(0.25) * j_value * j_value;

            contravariant_q2_at_v(j, i) =
                VVM::real(0.125) * i_value * i_value +
                VVM::real(0.75) * j_value;
        });

    Kokkos::parallel_for(
        "CompareCartesianVerticalVorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {h, h},
            {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            actual(j, i) =
                vorticity.calculate_at_z(
                    contravariant_q1_at_u,
                    contravariant_q2_at_v,
                    j,
                    i);

            cartesian_reference(j, i) =
                (contravariant_q2_at_v(j, i + 1) -
                 contravariant_q2_at_v(j, i)) / dx -
                (contravariant_q1_at_u(j + 1, i) -
                 contravariant_q1_at_u(j, i)) / dy;
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
        "Cartesian vertical vorticity must equal dv/dx minus du/dy");
}

void test_nonorthogonal_linear_vector() {
    const int physical_nx = 12;
    const int physical_ny = 10;
    const int halo = 2;
    const int nx = physical_nx + 2 * halo;
    const int ny = physical_ny + 2 * halo;

    const VVM::Real dq1 = VVM::real(2.0);
    const VVM::Real dq2 = VVM::real(3.0);

    const VVM::Real g11 = VVM::real(2.0);
    const VVM::Real g12 = VVM::real(0.5);
    const VVM::Real g22 = VVM::real(3.0);

    const VVM::Real jacobian =
        std::sqrt(g11 * g22 - g12 * g12);

    const VVM::Real slope_q1 = VVM::real(1.25);
    const VVM::Real slope_q2 = VVM::real(-0.75);

    VerticalVorticityDeviceView vorticity;

    vorticity.vector_lowering.u.g_cov.a11 =
        GeometryField2D::constant_value(g11);

    vorticity.vector_lowering.u.g_cov.a12 =
        GeometryField2D::constant_value(g12);

    vorticity.vector_lowering.v.g_cov.a12 =
        GeometryField2D::constant_value(g12);

    vorticity.vector_lowering.v.g_cov.a22 =
        GeometryField2D::constant_value(g22);

    vorticity.curl.z.inv_sqrt_g =
        GeometryField2D::constant_value(
            VVM::real(1.0) / jacobian);

    vorticity.curl.z.dq1 = dq1;
    vorticity.curl.z.dq2 = dq2;

    Kokkos::View<VVM::Real**> contravariant_q1_at_u(
        "linear_nonorthogonal_q1_at_u", ny, nx);

    Kokkos::View<VVM::Real**> contravariant_q2_at_v(
        "linear_nonorthogonal_q2_at_v", ny, nx);

    Kokkos::View<VVM::Real**> result(
        "linear_nonorthogonal_vorticity", ny, nx);

    Kokkos::parallel_for(
        "InitializeLinearNonorthogonalWind",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real q1_at_u =
                (static_cast<VVM::Real>(i - halo) +
                 VVM::real(0.5)) * dq1;

            const VVM::Real q2_at_v =
                (static_cast<VVM::Real>(j - halo) +
                 VVM::real(0.5)) * dq2;

            contravariant_q1_at_u(j, i) =
                slope_q1 * q1_at_u;

            contravariant_q2_at_v(j, i) =
                slope_q2 * q2_at_v;
        });

    Kokkos::parallel_for(
        "EvaluateLinearNonorthogonalVorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {halo, halo},
            {ny - halo, nx - halo}),
        KOKKOS_LAMBDA(const int j, const int i) {
            result(j, i) =
                vorticity.calculate_at_z(
                    contravariant_q1_at_u,
                    contravariant_q2_at_v,
                    j,
                    i);
        });

    const auto result_host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            result);

    const VVM::Real expected =
        (g12 / jacobian) *
        (slope_q1 - slope_q2);

    VVM::Real maximum_error = VVM::real(0.0);

    for (int j = halo; j < ny - halo; ++j) {
        for (int i = halo; i < nx - halo; ++i) {
            maximum_error = std::max(
                maximum_error,
                std::abs(result_host(j, i) - expected));
        }
    }

    check(
        maximum_error <= VVM::real(1.0e-5),
        "Vertical vorticity must include the nonorthogonal g12 contributions");
}

void test_three_dimensional_overload() {
    const HorizontalDomainLayout layout = make_layout(12, 10);
    const VVM::Real dx = VVM::real(100.0);
    const VVM::Real dy = VVM::real(200.0);
    const VVM::Real angular_velocity = VVM::real(0.002);

    const CartesianGeometry geometry(layout, dx, dy);
    const auto vorticity = make_vertical_vorticity_device_view(geometry);
    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real***> contravariant_q1_at_u(
        "three_dimensional_vorticity_q1_at_u", 2, ny, nx);

    Kokkos::View<VVM::Real***> contravariant_q2_at_v(
        "three_dimensional_vorticity_q2_at_v", 2, ny, nx);

    Kokkos::View<VVM::Real> result(
        "three_dimensional_vertical_vorticity");

    Kokkos::parallel_for(
        "InitializeThreeDimensionalVorticityWind",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {2, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const VVM::Real level_factor =
                static_cast<VVM::Real>(k + 1);

            contravariant_q1_at_u(k, j, i) =
                -level_factor *
                angular_velocity *
                u.q2(j, i);

            contravariant_q2_at_v(k, j, i) =
                level_factor *
                angular_velocity *
                v.q1(j, i);
        });

    Kokkos::parallel_for(
        "EvaluateThreeDimensionalVerticalVorticity",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result() =
                vorticity.calculate_at_z(
                    contravariant_q1_at_u,
                    contravariant_q2_at_v,
                    1,
                    h,
                    h);
        });

    VVM::Real result_host = VVM::real(0.0);
    Kokkos::deep_copy(result_host, result);

    const VVM::Real expected =
        VVM::real(4.0) * angular_velocity;

    check(
        close(result_host, expected),
        "The 3-D overload must calculate vertical vorticity at the requested level");
}

VVM::Real periodic_nonorthogonal_error(const int n) {
    const VVM::Real pi = std::acos(VVM::real(-1.0));
    const VVM::Real spacing =
        VVM::real(2.0) * pi /
        static_cast<VVM::Real>(n);

    const int halo = 2;
    const int nx = n + 2 * halo;
    const int ny = n + 2 * halo;

    const VVM::Real g11 = VVM::real(1.5);
    const VVM::Real g12 = VVM::real(0.25);
    const VVM::Real g22 = VVM::real(1.25);

    const VVM::Real jacobian =
        std::sqrt(g11 * g22 - g12 * g12);

    VerticalVorticityDeviceView vorticity;

    vorticity.vector_lowering.u.g_cov.a11 =
        GeometryField2D::constant_value(g11);

    vorticity.vector_lowering.u.g_cov.a12 =
        GeometryField2D::constant_value(g12);

    vorticity.vector_lowering.v.g_cov.a12 =
        GeometryField2D::constant_value(g12);

    vorticity.vector_lowering.v.g_cov.a22 =
        GeometryField2D::constant_value(g22);

    vorticity.curl.z.inv_sqrt_g =
        GeometryField2D::constant_value(
            VVM::real(1.0) / jacobian);

    vorticity.curl.z.dq1 = spacing;
    vorticity.curl.z.dq2 = spacing;

    Kokkos::View<VVM::Real**> contravariant_q1_at_u(
        "periodic_nonorthogonal_q1_at_u", ny, nx);

    Kokkos::View<VVM::Real**> contravariant_q2_at_v(
        "periodic_nonorthogonal_q2_at_v", ny, nx);

    Kokkos::View<VVM::Real**> result(
        "periodic_nonorthogonal_vorticity", ny, nx);

    Kokkos::parallel_for(
        "InitializePeriodicNonorthogonalWind",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real q1_at_u =
                (static_cast<VVM::Real>(i - halo) +
                 VVM::real(0.5)) * spacing;

            const VVM::Real q2_at_u =
                static_cast<VVM::Real>(j - halo) * spacing;

            const VVM::Real q1_at_v =
                static_cast<VVM::Real>(i - halo) * spacing;

            const VVM::Real q2_at_v =
                (static_cast<VVM::Real>(j - halo) +
                 VVM::real(0.5)) * spacing;

            contravariant_q1_at_u(j, i) =
                Kokkos::sin(q1_at_u) +
                Kokkos::cos(q2_at_u);

            contravariant_q2_at_v(j, i) =
                Kokkos::cos(q1_at_v) +
                Kokkos::sin(q2_at_v);
        });

    Kokkos::parallel_for(
        "EvaluatePeriodicNonorthogonalVorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {halo, halo},
            {ny - halo, nx - halo}),
        KOKKOS_LAMBDA(const int j, const int i) {
            result(j, i) =
                vorticity.calculate_at_z(
                    contravariant_q1_at_u,
                    contravariant_q2_at_v,
                    j,
                    i);
        });

    const auto result_host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            result);

    VVM::Real squared_error = VVM::real(0.0);

    for (int j = halo; j < ny - halo; ++j) {
        for (int i = halo; i < nx - halo; ++i) {
            const VVM::Real q1_at_z =
                (static_cast<VVM::Real>(i - halo) +
                 VVM::real(0.5)) * spacing;

            const VVM::Real q2_at_z =
                (static_cast<VVM::Real>(j - halo) +
                 VVM::real(0.5)) * spacing;

            const VVM::Real expected =
                (
                    g12 * std::cos(q1_at_z) -
                    g22 * std::sin(q1_at_z) +
                    g11 * std::sin(q2_at_z) -
                    g12 * std::cos(q2_at_z)
                ) / jacobian;

            const VVM::Real error =
                result_host(j, i) - expected;

            squared_error += error * error;
        }
    }

    return std::sqrt(
        squared_error /
        static_cast<VVM::Real>(n * n));
}

void test_second_order_nonorthogonal_convergence() {
    const VVM::Real coarse_error =
        periodic_nonorthogonal_error(32);

    const VVM::Real fine_error =
        periodic_nonorthogonal_error(64);

    const VVM::Real error_ratio =
        coarse_error / fine_error;

    check(
        coarse_error > fine_error,
        "Refining the grid must reduce the composed vertical-vorticity error");

    check(
        error_ratio > VVM::real(3.5),
        "The composed nonorthogonal vertical-vorticity operator must show second-order convergence");
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    {
        try {
            test_cartesian_equivalence();
            test_nonorthogonal_linear_vector();
            test_three_dimensional_overload();
            test_second_order_nonorthogonal_convergence();
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
            "test_vertical_vorticity: PASS\n");
    } else {
        std::fprintf(
            stderr,
            "test_vertical_vorticity: %d failure(s)\n",
            failures);
    }

    return failures == 0 ? 0 : 1;
}
