#include "core/geometry/CartesianGeometry.hpp"
#include "dynamics/operators/HorizontalFluxDivergence.hpp"

#include <cmath>
#include <cstdio>
#include <exception>

#include <Kokkos_Core.hpp>

namespace {

using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::GeometryField2D;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Dynamics::Operators::HorizontalFluxDivergenceDeviceView;
using VVM::Dynamics::Operators::make_horizontal_flux_divergence_device_view;

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

void test_explicit_metric_weighting() {
    HorizontalFluxDivergenceDeviceView divergence;

    divergence.t.inv_sqrt_g = GeometryField2D::constant_value(VVM::real(0.25));
    divergence.t.dq1 = VVM::real(2.0);
    divergence.t.dq2 = VVM::real(4.0);

    divergence.u.sqrt_g = GeometryField2D::constant_value(VVM::real(3.0));
    divergence.v.sqrt_g = GeometryField2D::constant_value(VVM::real(5.0));

    Kokkos::View<VVM::Real> result("metric_weighted_divergence");

    Kokkos::parallel_for(
        "EvaluateMetricWeightedDivergence",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result() = divergence.at_t(
                0,
                0,
                VVM::real(7.0),
                VVM::real(1.0),
                VVM::real(6.0),
                VVM::real(2.0));
        });

    VVM::Real host_result = VVM::real(0.0);
    Kokkos::deep_copy(host_result, result);

    const VVM::Real expected = VVM::real(0.25) * (
        VVM::real(3.0) * (VVM::real(7.0) - VVM::real(1.0)) / VVM::real(2.0) +
        VVM::real(5.0) * (VVM::real(6.0) - VVM::real(2.0)) / VVM::real(4.0));

    check(close(host_result, expected), "Flux divergence must apply face Jacobians and the inverse cell Jacobian");
}

void test_cartesian_equivalence() {
    const HorizontalDomainLayout layout = make_layout(18, 14);
    const VVM::Real dx = VVM::real(2.0);
    const VVM::Real dy = VVM::real(3.0);

    const CartesianGeometry geometry(layout, dx, dy);
    const auto divergence = make_horizontal_flux_divergence_device_view(geometry);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> flux_q1("cartesian_flux_q1", ny, nx);
    Kokkos::View<VVM::Real**> flux_q2("cartesian_flux_q2", ny, nx);
    Kokkos::View<VVM::Real**> actual("cartesian_divergence", ny, nx);
    Kokkos::View<VVM::Real**> legacy("legacy_cartesian_divergence", ny, nx);

    Kokkos::parallel_for(
        "InitializeCartesianFluxes",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real i_value = static_cast<VVM::Real>(i - h);
            const VVM::Real j_value = static_cast<VVM::Real>(j - h);

            flux_q1(j, i) = VVM::real(0.25) * i_value * i_value - VVM::real(0.5) * j_value;
            flux_q2(j, i) = VVM::real(0.125) * j_value * j_value + VVM::real(0.75) * i_value;
        });

    Kokkos::parallel_for(
        "CompareCartesianFluxDivergence",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            actual(j, i) = divergence.at_t(
                j,
                i,
                flux_q1(j, i),
                flux_q1(j, i - 1),
                flux_q2(j, i),
                flux_q2(j - 1, i));

            legacy(j, i) =
                (flux_q1(j, i) - flux_q1(j, i - 1)) / dx +
                (flux_q2(j, i) - flux_q2(j - 1, i)) / dy;
        });

    const auto actual_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
    const auto legacy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), legacy);

    VVM::Real maximum_difference = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            maximum_difference = std::max(maximum_difference, std::abs(actual_host(j, i) - legacy_host(j, i)));
        }
    }

    check(maximum_difference <= VVM::real(1.0e-6), "Cartesian generalized divergence must equal the legacy Cartesian formula");
}

void test_constant_flux() {
    const HorizontalDomainLayout layout = make_layout(12, 10);
    const CartesianGeometry geometry(layout, VVM::real(500.0), VVM::real(750.0));
    const auto divergence = make_horizontal_flux_divergence_device_view(geometry);

    Kokkos::View<VVM::Real> result("constant_flux_divergence");

    Kokkos::parallel_for(
        "EvaluateConstantFluxDivergence",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            result() = divergence.at_t(
                layout.halo,
                layout.halo,
                VVM::real(4.0),
                VVM::real(4.0),
                VVM::real(-3.0),
                VVM::real(-3.0));
        });

    VVM::Real host_result = VVM::real(0.0);
    Kokkos::deep_copy(host_result, result);

    check(close(host_result, VVM::real(0.0)), "A constant Cartesian flux must have zero divergence");
}

void test_linear_flux() {
    const HorizontalDomainLayout layout = make_layout(16, 12);
    const VVM::Real dx = VVM::real(100.0);
    const VVM::Real dy = VVM::real(200.0);
    const VVM::Real slope_q1 = VVM::real(1.25);
    const VVM::Real slope_q2 = VVM::real(-0.75);

    const CartesianGeometry geometry(layout, dx, dy);
    const auto divergence = make_horizontal_flux_divergence_device_view(geometry);
    const auto u = geometry.device_view(VVM::Core::Geometry::HorizontalLocation::U);
    const auto v = geometry.device_view(VVM::Core::Geometry::HorizontalLocation::V);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> flux_q1("linear_flux_q1", ny, nx);
    Kokkos::View<VVM::Real**> flux_q2("linear_flux_q2", ny, nx);
    Kokkos::View<VVM::Real**> result("linear_flux_divergence", ny, nx);

    Kokkos::parallel_for(
        "InitializeLinearFlux",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            flux_q1(j, i) = slope_q1 * u.q1(j, i) + VVM::real(2.0);
            flux_q2(j, i) = slope_q2 * v.q2(j, i) - VVM::real(3.0);
        });

    Kokkos::parallel_for(
        "EvaluateLinearFluxDivergence",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            result(j, i) = divergence.at_t(
                j,
                i,
                flux_q1(j, i),
                flux_q1(j, i - 1),
                flux_q2(j, i),
                flux_q2(j - 1, i));
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    const VVM::Real expected = slope_q1 + slope_q2;

    VVM::Real maximum_error = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            maximum_error = std::max(maximum_error, std::abs(result_host(j, i) - expected));
        }
    }

    check(maximum_error <= VVM::real(1.0e-5), "The divergence of a linear Cartesian flux must equal the sum of its slopes");
}

VVM::Real periodic_sine_error(const int n) {
    const VVM::Real pi = std::acos(VVM::real(-1.0));
    const VVM::Real domain_length = VVM::real(2.0) * pi;
    const VVM::Real spacing = domain_length / static_cast<VVM::Real>(n);

    const HorizontalDomainLayout layout = make_layout(n, n);
    const CartesianGeometry geometry(layout, spacing, spacing);
    const auto divergence = make_horizontal_flux_divergence_device_view(geometry);
    const auto t = geometry.device_view(VVM::Core::Geometry::HorizontalLocation::T);
    const auto u = geometry.device_view(VVM::Core::Geometry::HorizontalLocation::U);
    const auto v = geometry.device_view(VVM::Core::Geometry::HorizontalLocation::V);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> flux_q1("sine_flux_q1", ny, nx);
    Kokkos::View<VVM::Real**> flux_q2("sine_flux_q2", ny, nx);
    Kokkos::View<VVM::Real**> result("sine_flux_divergence", ny, nx);

    Kokkos::parallel_for(
        "InitializePeriodicSineFlux",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            flux_q1(j, i) = Kokkos::sin(u.q1(j, i));
            flux_q2(j, i) = Kokkos::cos(v.q2(j, i));
        });

    Kokkos::parallel_for(
        "EvaluatePeriodicSineDivergence",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            result(j, i) = divergence.at_t(
                j,
                i,
                flux_q1(j, i),
                flux_q1(j, i - 1),
                flux_q2(j, i),
                flux_q2(j - 1, i));
        });

    const auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    VVM::Real squared_error = VVM::real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            const VVM::Real x = static_cast<VVM::Real>(i - h) * spacing;
            const VVM::Real y = static_cast<VVM::Real>(j - h) * spacing;
            const VVM::Real expected = std::cos(x) - std::sin(y);
            const VVM::Real error = result_host(j, i) - expected;

            squared_error += error * error;
        }
    }

    return std::sqrt(squared_error / static_cast<VVM::Real>(n * n));
}

void test_periodic_global_flux_integral() {
    const int physical_nx = 48;
    const int physical_ny = 36;
    const VVM::Real pi = std::acos(VVM::real(-1.0));
    const VVM::Real domain_length = VVM::real(2.0) * pi;
    const VVM::Real dx = domain_length / static_cast<VVM::Real>(physical_nx);
    const VVM::Real dy = domain_length / static_cast<VVM::Real>(physical_ny);

    const HorizontalDomainLayout layout = make_layout(physical_nx, physical_ny);
    const CartesianGeometry geometry(layout, dx, dy);
    const auto divergence = make_horizontal_flux_divergence_device_view(geometry);
    const auto t = geometry.device_view(VVM::Core::Geometry::HorizontalLocation::T);
    const auto u = geometry.device_view(VVM::Core::Geometry::HorizontalLocation::U);
    const auto v = geometry.device_view(VVM::Core::Geometry::HorizontalLocation::V);

    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    Kokkos::View<VVM::Real**> flux_q1("periodic_integral_flux_q1", ny, nx);
    Kokkos::View<VVM::Real**> flux_q2("periodic_integral_flux_q2", ny, nx);
    Kokkos::View<VVM::Real**> divergence_at_t("periodic_integral_divergence", ny, nx);

    Kokkos::parallel_for(
        "InitializePeriodicIntegralFlux",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            flux_q1(j, i) =
                Kokkos::sin(VVM::real(2.0) * u.q1(j, i)) +
                VVM::real(0.37) * Kokkos::cos(VVM::real(3.0) * u.q2(j, i));

            flux_q2(j, i) =
                VVM::real(0.41) * Kokkos::sin(VVM::real(4.0) * v.q1(j, i)) -
                VVM::real(0.65) * Kokkos::cos(VVM::real(2.0) * v.q2(j, i));
        });

    Kokkos::parallel_for(
        "EvaluatePeriodicIntegralDivergence",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            divergence_at_t(j, i) = divergence.at_t(
                j,
                i,
                flux_q1(j, i),
                flux_q1(j, i - 1),
                flux_q2(j, i),
                flux_q2(j - 1, i));
        });

    Kokkos::View<VVM::Real> area_integral("periodic_divergence_area_integral");

    Kokkos::parallel_reduce(
        "IntegratePeriodicDivergence",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i, VVM::Real& sum) {
            const VVM::Real cell_area =
                t.sqrt_g(j, i) * t.dq1 * t.dq2;

            sum += divergence_at_t(j, i) * cell_area;
        },
        area_integral);

    VVM::Real area_integral_host = VVM::real(0.0);
    Kokkos::deep_copy(area_integral_host, area_integral);

    const VVM::Real integral_tolerance =
        sizeof(VVM::Real) == sizeof(double)
            ? VVM::real(1.0e-10)
            : VVM::real(1.0e-4);

    check(
        std::abs(area_integral_host) <= integral_tolerance,
        "The area integral of periodic flux divergence must be zero");
}

void test_second_order_convergence() {
    const VVM::Real coarse_error = periodic_sine_error(32);
    const VVM::Real fine_error = periodic_sine_error(64);
    const VVM::Real error_ratio = coarse_error / fine_error;

    check(coarse_error > fine_error, "Refining the Cartesian grid must reduce the manufactured-solution error");
    check(error_ratio > VVM::real(3.5), "Centered face divergence must show second-order convergence");
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    {
        try {
            test_explicit_metric_weighting();
            test_cartesian_equivalence();
            test_constant_flux();
            test_linear_flux();
            test_periodic_global_flux_integral();
            test_second_order_convergence();
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
        }
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::fprintf(stdout, "test_horizontal_flux_divergence: PASS\n");
    } else {
        std::fprintf(stderr, "test_horizontal_flux_divergence: %d failure(s)\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
