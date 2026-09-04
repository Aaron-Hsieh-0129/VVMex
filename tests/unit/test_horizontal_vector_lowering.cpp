#include "core/geometry/CartesianGeometry.hpp"
#include "dynamics/operators/HorizontalVectorLowering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>

#include <Kokkos_Core.hpp>

namespace {

using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::GeometryField2D;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Dynamics::Operators::ContravariantVectorStencilAtU;
using VVM::Dynamics::Operators::ContravariantVectorStencilAtV;
using VVM::Dynamics::Operators::HorizontalVectorLoweringDeviceView;
using VVM::Dynamics::Operators::load_contravariant_vector_stencil_at_u;
using VVM::Dynamics::Operators::load_contravariant_vector_stencil_at_v;
using VVM::Dynamics::Operators::make_horizontal_vector_lowering_device_view;

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

void test_cartesian_identity() {
    const HorizontalDomainLayout layout = make_layout(12, 10);
    const CartesianGeometry geometry(layout, VVM::real(500.0), VVM::real(750.0));
    const auto lowering = make_horizontal_vector_lowering_device_view(geometry);

    ContravariantVectorStencilAtU vector_at_u;
    vector_at_u.q1_at_u_j_i = VVM::real(4.5);
    vector_at_u.q2_at_v_j_i = VVM::real(1.0);
    vector_at_u.q2_at_v_j_ip1 = VVM::real(2.0);
    vector_at_u.q2_at_v_jm1_i = VVM::real(3.0);
    vector_at_u.q2_at_v_jm1_ip1 = VVM::real(4.0);

    ContravariantVectorStencilAtV vector_at_v;
    vector_at_v.q2_at_v_j_i = VVM::real(-2.25);
    vector_at_v.q1_at_u_j_i = VVM::real(5.0);
    vector_at_v.q1_at_u_jp1_i = VVM::real(6.0);
    vector_at_v.q1_at_u_j_im1 = VVM::real(7.0);
    vector_at_v.q1_at_u_jp1_im1 = VVM::real(8.0);

    Kokkos::View<VVM::Real*> result("cartesian_lowering_result", 2);

    Kokkos::parallel_for(
        "EvaluateCartesianVectorLowering",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result(0) = lowering.calculate_covariant_q1_at_u(
                layout.halo,
                layout.halo,
                vector_at_u);

            result(1) = lowering.calculate_covariant_q2_at_v(
                layout.halo,
                layout.halo,
                vector_at_v);
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    check(close(result_host(0), VVM::real(4.5)), "Cartesian lowering must preserve the q1 component");
    check(close(result_host(1), VVM::real(-2.25)), "Cartesian lowering must preserve the q2 component");
}

void test_nonorthogonal_metric() {
    HorizontalVectorLoweringDeviceView lowering;

    lowering.u.g_cov.a11 = GeometryField2D::constant_value(VVM::real(2.0));
    lowering.u.g_cov.a12 = GeometryField2D::constant_value(VVM::real(0.5));

    lowering.v.g_cov.a12 = GeometryField2D::constant_value(VVM::real(0.5));
    lowering.v.g_cov.a22 = GeometryField2D::constant_value(VVM::real(3.0));

    ContravariantVectorStencilAtU vector_at_u;
    vector_at_u.q1_at_u_j_i = VVM::real(4.0);
    vector_at_u.q2_at_v_j_i = VVM::real(2.0);
    vector_at_u.q2_at_v_j_ip1 = VVM::real(4.0);
    vector_at_u.q2_at_v_jm1_i = VVM::real(6.0);
    vector_at_u.q2_at_v_jm1_ip1 = VVM::real(8.0);

    ContravariantVectorStencilAtV vector_at_v;
    vector_at_v.q2_at_v_j_i = VVM::real(-2.0);
    vector_at_v.q1_at_u_j_i = VVM::real(1.0);
    vector_at_v.q1_at_u_jp1_i = VVM::real(3.0);
    vector_at_v.q1_at_u_j_im1 = VVM::real(5.0);
    vector_at_v.q1_at_u_jp1_im1 = VVM::real(7.0);

    Kokkos::View<VVM::Real*> result("nonorthogonal_lowering_result", 2);

    Kokkos::parallel_for(
        "EvaluateNonorthogonalVectorLowering",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result(0) = lowering.calculate_covariant_q1_at_u(0, 0, vector_at_u);
            result(1) = lowering.calculate_covariant_q2_at_v(0, 0, vector_at_v);
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    const VVM::Real expected_q1 =
        VVM::real(2.0) * VVM::real(4.0) +
        VVM::real(0.5) * VVM::real(5.0);

    const VVM::Real expected_q2 =
        VVM::real(3.0) * VVM::real(-2.0) +
        VVM::real(0.5) * VVM::real(4.0);

    check(close(result_host(0), expected_q1), "Covariant q1 must contain the interpolated g12 q2 contribution");
    check(close(result_host(1), expected_q2), "Covariant q2 must contain the interpolated g12 q1 contribution");
}

void test_metric_weighting_before_interpolation() {
    constexpr int nx = 4;
    constexpr int ny = 4;

    Kokkos::View<VVM::Real**> q1_at_u("weighted_q1_at_u", ny, nx);
    Kokkos::View<VVM::Real**> q2_at_v("weighted_q2_at_v", ny, nx);
    Kokkos::View<VVM::Real**> g12_at_u("weighted_g12_at_u", ny, nx);
    Kokkos::View<VVM::Real**> g12_at_v("weighted_g12_at_v", ny, nx);

    Kokkos::parallel_for(
        "InitializeWeightedLoweringFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            q1_at_u(j, i) =
                VVM::real(1.0) +
                VVM::real(0.5) * static_cast<VVM::Real>(i) +
                VVM::real(0.25) * static_cast<VVM::Real>(j);

            q2_at_v(j, i) =
                VVM::real(2.0) +
                VVM::real(0.75) * static_cast<VVM::Real>(i) -
                VVM::real(0.125) * static_cast<VVM::Real>(j);

            g12_at_u(j, i) =
                VVM::real(0.1) *
                static_cast<VVM::Real>(1 + 2 * j + i);

            g12_at_v(j, i) =
                VVM::real(0.05) *
                static_cast<VVM::Real>(2 + j + 3 * i);
        });

    HorizontalVectorLoweringDeviceView lowering;

    lowering.u.g_cov.a11 = GeometryField2D::constant_value(VVM::real(2.0));
    lowering.u.g_cov.a12 = GeometryField2D::full_2d(g12_at_u);

    lowering.v.g_cov.a12 = GeometryField2D::full_2d(g12_at_v);
    lowering.v.g_cov.a22 = GeometryField2D::constant_value(VVM::real(3.0));

    Kokkos::View<VVM::Real*> result("weighted_lowering_result", 4);

    Kokkos::parallel_for(
        "EvaluateWeightedVectorLowering",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            constexpr int j = 1;
            constexpr int i = 1;

            result(0) = lowering.calculate_covariant_q1_at_u(q1_at_u, q2_at_v, j, i);
            result(1) =
                VVM::real(2.0) * q1_at_u(j, i) +
                VVM::real(0.25) * (
                    g12_at_v(j, i) * q2_at_v(j, i) +
                    g12_at_v(j, i + 1) * q2_at_v(j, i + 1) +
                    g12_at_v(j - 1, i) * q2_at_v(j - 1, i) +
                    g12_at_v(j - 1, i + 1) * q2_at_v(j - 1, i + 1));

            result(2) = lowering.calculate_covariant_q2_at_v(q1_at_u, q2_at_v, j, i);
            result(3) =
                VVM::real(3.0) * q2_at_v(j, i) +
                VVM::real(0.25) * (
                    g12_at_u(j, i) * q1_at_u(j, i) +
                    g12_at_u(j + 1, i) * q1_at_u(j + 1, i) +
                    g12_at_u(j, i - 1) * q1_at_u(j, i - 1) +
                    g12_at_u(j + 1, i - 1) * q1_at_u(j + 1, i - 1));
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    check(close(result_host(0), result_host(1)), "The q1 cross term must weight values before interpolation");
    check(close(result_host(2), result_host(3)), "The q2 cross term must weight values before interpolation");
}

void test_three_dimensional_stencil_loaders() {
    Kokkos::View<VVM::Real***> q1_at_u("three_dimensional_q1_at_u", 2, 3, 3);
    Kokkos::View<VVM::Real***> q2_at_v("three_dimensional_q2_at_v", 2, 3, 3);
    Kokkos::View<VVM::Real*> result("three_dimensional_lowering_stencils", 10);

    Kokkos::parallel_for(
        "InitializeThreeDimensionalLoweringFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {2, 3, 3}),
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
        "LoadThreeDimensionalLoweringStencils",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            const ContravariantVectorStencilAtU stencil_at_u =
                load_contravariant_vector_stencil_at_u(q1_at_u, q2_at_v, 1, 1, 1);

            const ContravariantVectorStencilAtV stencil_at_v =
                load_contravariant_vector_stencil_at_v(q1_at_u, q2_at_v, 1, 1, 1);

            result(0) = stencil_at_u.q1_at_u_j_i;
            result(1) = stencil_at_u.q2_at_v_j_i;
            result(2) = stencil_at_u.q2_at_v_j_ip1;
            result(3) = stencil_at_u.q2_at_v_jm1_i;
            result(4) = stencil_at_u.q2_at_v_jm1_ip1;

            result(5) = stencil_at_v.q2_at_v_j_i;
            result(6) = stencil_at_v.q1_at_u_j_i;
            result(7) = stencil_at_v.q1_at_u_jp1_i;
            result(8) = stencil_at_v.q1_at_u_j_im1;
            result(9) = stencil_at_v.q1_at_u_jp1_im1;
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    check(close(result_host(0), VVM::real(111.0)), "The U stencil must load q1 at U(j,i)");
    check(close(result_host(1), VVM::real(1110.0)), "The U stencil must load q2 at V(j,i)");
    check(close(result_host(2), VVM::real(1120.0)), "The U stencil must load q2 at V(j,i+1)");
    check(close(result_host(3), VVM::real(1010.0)), "The U stencil must load q2 at V(j-1,i)");
    check(close(result_host(4), VVM::real(1020.0)), "The U stencil must load q2 at V(j-1,i+1)");

    check(close(result_host(5), VVM::real(1110.0)), "The V stencil must load q2 at V(j,i)");
    check(close(result_host(6), VVM::real(111.0)), "The V stencil must load q1 at U(j,i)");
    check(close(result_host(7), VVM::real(121.0)), "The V stencil must load q1 at U(j+1,i)");
    check(close(result_host(8), VVM::real(110.0)), "The V stencil must load q1 at U(j,i-1)");
    check(close(result_host(9), VVM::real(120.0)), "The V stencil must load q1 at U(j+1,i-1)");
}

VVM::Real smooth_lowering_error(const int n) {
    const VVM::Real pi = std::acos(VVM::real(-1.0));
    const VVM::Real spacing = VVM::real(2.0) * pi / static_cast<VVM::Real>(n);
    const int halo = 2;
    const int nx = n + 2 * halo;
    const int ny = n + 2 * halo;

    Kokkos::View<VVM::Real**> q1_at_u("smooth_q1_at_u", ny, nx);
    Kokkos::View<VVM::Real**> q2_at_v("smooth_q2_at_v", ny, nx);
    Kokkos::View<VVM::Real**> g12_at_u("smooth_g12_at_u", ny, nx);
    Kokkos::View<VVM::Real**> g12_at_v("smooth_g12_at_v", ny, nx);
    Kokkos::View<VVM::Real**> covariant_q1_at_u("smooth_covariant_q1_at_u", ny, nx);
    Kokkos::View<VVM::Real**> covariant_q2_at_v("smooth_covariant_q2_at_v", ny, nx);

    Kokkos::parallel_for(
        "InitializeSmoothLoweringFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real x_at_u =
                (static_cast<VVM::Real>(i - halo) + VVM::real(0.5)) * spacing;
            const VVM::Real y_at_u =
                static_cast<VVM::Real>(j - halo) * spacing;

            const VVM::Real x_at_v =
                static_cast<VVM::Real>(i - halo) * spacing;
            const VVM::Real y_at_v =
                (static_cast<VVM::Real>(j - halo) + VVM::real(0.5)) * spacing;

            q1_at_u(j, i) =
                Kokkos::sin(x_at_u) +
                VVM::real(0.5) * Kokkos::cos(y_at_u);

            q2_at_v(j, i) =
                Kokkos::cos(x_at_v) -
                VVM::real(0.25) * Kokkos::sin(y_at_v);

            g12_at_u(j, i) =
                VVM::real(0.2) +
                VVM::real(0.05) * Kokkos::sin(x_at_u + y_at_u);

            g12_at_v(j, i) =
                VVM::real(0.2) +
                VVM::real(0.05) * Kokkos::sin(x_at_v + y_at_v);
        });

    HorizontalVectorLoweringDeviceView lowering;

    lowering.u.g_cov.a11 = GeometryField2D::constant_value(VVM::real(1.5));
    lowering.u.g_cov.a12 = GeometryField2D::full_2d(g12_at_u);

    lowering.v.g_cov.a12 = GeometryField2D::full_2d(g12_at_v);
    lowering.v.g_cov.a22 = GeometryField2D::constant_value(VVM::real(1.25));

    Kokkos::parallel_for(
        "EvaluateSmoothVectorLowering",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({halo, halo}, {ny - halo, nx - halo}),
        KOKKOS_LAMBDA(const int j, const int i) {
            covariant_q1_at_u(j, i) =
                lowering.calculate_covariant_q1_at_u(q1_at_u, q2_at_v, j, i);

            covariant_q2_at_v(j, i) =
                lowering.calculate_covariant_q2_at_v(q1_at_u, q2_at_v, j, i);
        });

    const auto covariant_q1_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), covariant_q1_at_u);

    const auto covariant_q2_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), covariant_q2_at_v);

    VVM::Real squared_error = VVM::real(0.0);

    for (int j = halo; j < ny - halo; ++j) {
        for (int i = halo; i < nx - halo; ++i) {
            const VVM::Real x_at_u =
                (static_cast<VVM::Real>(i - halo) + VVM::real(0.5)) * spacing;
            const VVM::Real y_at_u =
                static_cast<VVM::Real>(j - halo) * spacing;

            const VVM::Real x_at_v =
                static_cast<VVM::Real>(i - halo) * spacing;
            const VVM::Real y_at_v =
                (static_cast<VVM::Real>(j - halo) + VVM::real(0.5)) * spacing;

            const VVM::Real q1_exact =
                std::sin(x_at_u) +
                VVM::real(0.5) * std::cos(y_at_u);

            const VVM::Real q2_exact_at_u =
                std::cos(x_at_u) -
                VVM::real(0.25) * std::sin(y_at_u);

            const VVM::Real g12_exact_at_u =
                VVM::real(0.2) +
                VVM::real(0.05) * std::sin(x_at_u + y_at_u);

            const VVM::Real q1_exact_at_v =
                std::sin(x_at_v) +
                VVM::real(0.5) * std::cos(y_at_v);

            const VVM::Real q2_exact =
                std::cos(x_at_v) -
                VVM::real(0.25) * std::sin(y_at_v);

            const VVM::Real g12_exact_at_v =
                VVM::real(0.2) +
                VVM::real(0.05) * std::sin(x_at_v + y_at_v);

            const VVM::Real expected_q1 =
                VVM::real(1.5) * q1_exact +
                g12_exact_at_u * q2_exact_at_u;

            const VVM::Real expected_q2 =
                g12_exact_at_v * q1_exact_at_v +
                VVM::real(1.25) * q2_exact;

            const VVM::Real q1_error =
                covariant_q1_host(j, i) - expected_q1;

            const VVM::Real q2_error =
                covariant_q2_host(j, i) - expected_q2;

            squared_error += q1_error * q1_error + q2_error * q2_error;
        }
    }

    return std::sqrt(squared_error / static_cast<VVM::Real>(2 * n * n));
}

void test_second_order_cross_component_interpolation() {
    const VVM::Real coarse_error = smooth_lowering_error(32);
    const VVM::Real fine_error = smooth_lowering_error(64);
    const VVM::Real error_ratio = coarse_error / fine_error;

    check(coarse_error > fine_error, "Refining the grid must reduce the vector-lowering interpolation error");
    check(error_ratio > VVM::real(3.5), "The U/V cross-component interpolation must show second-order convergence");
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    {
        try {
            test_cartesian_identity();
            test_nonorthogonal_metric();
            test_metric_weighting_before_interpolation();
            test_three_dimensional_stencil_loaders();
            test_second_order_cross_component_interpolation();
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
        }
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::fprintf(stdout, "test_horizontal_vector_lowering: PASS\n");
    } else {
        std::fprintf(stderr, "test_horizontal_vector_lowering: %d failure(s)\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
