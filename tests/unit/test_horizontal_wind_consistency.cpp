#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/HorizontalCurl.hpp"
#include "dynamics/operators/HorizontalFluxDivergence.hpp"
#include "dynamics/operators/HorizontalLaplaceBeltrami.hpp"
#include "dynamics/operators/HorizontalVectorLowering.hpp"
#include "dynamics/operators/HorizontalWindReconstruction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace {

using VVM::Real;
using VVM::real;
using namespace VVM::Core::Geometry;
using namespace VVM::Dynamics::Operators;

using View2D = Kokkos::View<Real**>;
using Policy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

enum class TestMode {
    Constant,
    Rotational,
    Divergent,
    Mixed,
    Zonal
};

const char* mode_name(const TestMode mode) {
    switch (mode) {
        case TestMode::Constant:
            return "constant";
        case TestMode::Rotational:
            return "rotational";
        case TestMode::Divergent:
            return "divergent";
        case TestMode::Mixed:
            return "mixed";
        case TestMode::Zonal:
            return "zonal";
    }

    return "unknown";
}

HorizontalDomainLayout make_layout() {
    HorizontalDomainLayout layout;

    layout.global_nx = 12;
    layout.global_ny = 16;
    layout.local_physical_nx = layout.global_nx;
    layout.local_physical_ny = layout.global_ny;
    layout.global_start_i = 0;
    layout.global_start_j = 0;

    // Reconstruction and lowering are evaluated on successively smaller
    // regions, leaving every stencil needed by the physical cells valid.
    layout.halo = 3;

    return layout;
}

int run_case(const HorizontalGeometry& geometry, const TestMode mode, const double radius, const double south_edge) {
    const auto& layout = geometry.layout();
    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;

    const bool spherical = geometry.kind() == GeometryKind::RegularLatLon;
    const double dq1 = static_cast<double>(geometry.dq1());
    const double dq2 = static_cast<double>(geometry.dq2());
    const double length_scale = spherical ? radius : 1.0;
    const double zonal_speed = 30.0;

    double amplitude = spherical ? radius * radius * 1.0e-5 : 1.0;
    if (mode == TestMode::Zonal) {
        amplitude = spherical ? radius * zonal_speed : zonal_speed * dq2;
    }

    View2D psi("consistency_psi", ny, nx);
    View2D chi("consistency_chi", ny, nx);
    View2D q1("consistency_contravariant_q1", ny, nx);
    View2D q2("consistency_contravariant_q2", ny, nx);
    View2D covariant_q1("consistency_covariant_q1", ny, nx);
    View2D covariant_q2("consistency_covariant_q2", ny, nx);
    View2D physical_q1("consistency_physical_q1", ny, nx);
    View2D physical_q2("consistency_physical_q2", ny, nx);

    // Components: divergence, Laplacian(chi), curl, Laplacian(psi).
    Kokkos::View<Real***> diagnostics("consistency_diagnostics", 4, ny, nx);

    auto psi_host = Kokkos::create_mirror_view(psi);
    auto chi_host = Kokkos::create_mirror_view(chi);

    const bool use_psi = mode == TestMode::Rotational || mode == TestMode::Mixed;
    const bool use_chi = mode == TestMode::Divergent || mode == TestMode::Mixed;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x_z = static_cast<double>(i - h) + 1.0;
            const double y_z = static_cast<double>(j - h) + 1.0;
            const double x_t = static_cast<double>(i - h) + 0.5;
            const double y_t = static_cast<double>(j - h) + 0.5;

            const double psi_shape = 0.25 * x_z * x_z - 0.375 * y_z * y_z + 0.125 * x_z * y_z;
            const double chi_shape = -0.125 * x_t * x_t + 0.25 * y_t * y_t - 0.0625 * x_t * y_t;

            double psi_value = amplitude * (7.0 + (use_psi ? psi_shape : 0.0));
            double chi_value = amplitude * (-3.0 + (use_chi ? chi_shape : 0.0));

            if (mode == TestMode::Zonal) {
                if (spherical) {
                    const double latitude_z = south_edge + y_z * dq2;
                    psi_value = -radius * zonal_speed * std::sin(latitude_z);
                } else {
                    psi_value = -zonal_speed * y_z * dq2;
                }

                chi_value = 0.0;
            }

            psi_host(j, i) = static_cast<Real>(psi_value);
            chi_host(j, i) = static_cast<Real>(chi_value);
        }
    }

    Kokkos::deep_copy(psi, psi_host);
    Kokkos::deep_copy(chi, chi_host);

    const auto reconstruction = make_horizontal_wind_reconstruction_device_view(geometry);
    const auto lowering = make_horizontal_vector_lowering_device_view(geometry);
    const auto divergence = make_horizontal_flux_divergence_device_view(geometry);
    const auto curl = make_horizontal_curl_device_view(geometry);
    const auto laplacian = make_horizontal_laplace_beltrami_device_view(geometry);

    // These diagonal conversions are valid for the Cartesian and RLL
    // geometries tested here. They are not a cubed-sphere transformation.
    const auto physical_scale_q1 = geometry.device_view(HorizontalLocation::U).contravariant_to_physical.a11;
    const auto physical_scale_q2 = geometry.device_view(HorizontalLocation::V).contravariant_to_physical.a22;

    Kokkos::parallel_for("ConsistencyReconstructWind", Policy2D({1, 1}, {ny - 1, nx - 1}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const Real value_q1 = reconstruction.calculate_contravariant_q1_at_u(psi, chi, j, i);
            const Real value_q2 = reconstruction.calculate_contravariant_q2_at_v(psi, chi, j, i);

            q1(j, i) = value_q1;
            q2(j, i) = value_q2;

            physical_q1(j, i) = physical_scale_q1(j, i) * value_q1;
            physical_q2(j, i) = physical_scale_q2(j, i) * value_q2;
        });

    Kokkos::parallel_for("ConsistencyLowerWind", Policy2D({2, 2}, {ny - 2, nx - 2}),
        KOKKOS_LAMBDA(const int j, const int i) {
            covariant_q1(j, i) = lowering.calculate_covariant_q1_at_u(q1, q2, j, i);
            covariant_q2(j, i) = lowering.calculate_covariant_q2_at_v(q1, q2, j, i);
        });

    Kokkos::parallel_for("ConsistencyEvaluateOperators", Policy2D({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            diagnostics(0, j, i) = divergence.at_t(j, i, q1(j, i), q1(j, i - 1), q2(j, i), q2(j - 1, i));
            diagnostics(1, j, i) = laplacian.calculate_at_t(chi, j, i);
            diagnostics(2, j, i) = curl.calculate_at_z(covariant_q1, covariant_q2, j, i);
            diagnostics(3, j, i) = laplacian.calculate_at_z(psi, j, i);
        });

    const auto diagnostics_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagnostics);
    const auto physical_q1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), physical_q1);
    const auto physical_q2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), physical_q2);

    // Characteristic scales remain nonzero for the cancellation tests.
    // In particular, do not divide a zero-divergence error by its expected zero.
    const double inverse_spacing_sum = 1.0 / dq1 + 1.0 / dq2;
    const double inverse_spacing_squared_sum = 1.0 / (dq1 * dq1) + 1.0 / (dq2 * dq2);
    const double velocity_scale = amplitude * inverse_spacing_sum / length_scale;
    const double derivative_scale = amplitude * inverse_spacing_squared_sum / (length_scale * length_scale);

    const double tolerance = sizeof(Real) == sizeof(float) ? 5.0e-4 : 5.0e-11;

    std::array<double, 4> errors = {};
    bool finite = true;
    double mean_q1 = 0.0;
    double expected_mean_q1 = 0.0;

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            const double actual_divergence = static_cast<double>(diagnostics_host(0, j, i));
            const double expected_divergence = static_cast<double>(diagnostics_host(1, j, i));
            const double actual_curl = static_cast<double>(diagnostics_host(2, j, i));
            const double expected_curl = static_cast<double>(diagnostics_host(3, j, i));

            const double actual_q1 = static_cast<double>(physical_q1_host(j, i));
            const double actual_q2 = static_cast<double>(physical_q2_host(j, i));

            // Independent physical-component reference from the sampled
            // potentials. It does not use the geometry conversion matrices.
            const double dpsi_dq2 = (static_cast<double>(psi_host(j, i)) - static_cast<double>(psi_host(j - 1, i))) / dq2;
            const double dpsi_dq1 = (static_cast<double>(psi_host(j, i)) - static_cast<double>(psi_host(j, i - 1))) / dq1;
            const double dchi_dq1 = (static_cast<double>(chi_host(j, i + 1)) - static_cast<double>(chi_host(j, i))) / dq1;
            const double dchi_dq2 = (static_cast<double>(chi_host(j + 1, i)) - static_cast<double>(chi_host(j, i))) / dq2;

            double expected_q1 = -dpsi_dq2 + dchi_dq1;
            double expected_q2 = dpsi_dq1 + dchi_dq2;

            if (spherical) {
                const double latitude_u = south_edge + (static_cast<double>(j - h) + 0.5) * dq2;
                const double latitude_v = south_edge + (static_cast<double>(j - h) + 1.0) * dq2;

                expected_q1 = -dpsi_dq2 / radius + dchi_dq1 / (radius * std::cos(latitude_u));
                expected_q2 = dpsi_dq1 / (radius * std::cos(latitude_v)) + dchi_dq2 / radius;
            }

            if (mode == TestMode::Zonal) {
                // Analytic result of applying the centered difference to
                // psi = -R*U0*sin(phi), including its finite-spacing factor.
                if (spherical) {
                    const double latitude_u = south_edge + (static_cast<double>(j - h) + 0.5) * dq2;
                    const double half_dlatitude = 0.5 * dq2;
                    expected_q1 = zonal_speed * std::cos(latitude_u) * std::sin(half_dlatitude) / half_dlatitude;
                } else {
                    expected_q1 = zonal_speed;
                }

                expected_q2 = 0.0;
            }

            const std::array<double, 8> values = {
                actual_divergence, expected_divergence, actual_curl, expected_curl,
                actual_q1, expected_q1, actual_q2, expected_q2
            };

            for (const double value : values) {
                finite = finite && std::isfinite(value);
            }

            errors[0] = std::max(errors[0], std::abs(actual_divergence - expected_divergence) / derivative_scale);
            errors[1] = std::max(errors[1], std::abs(actual_curl - expected_curl) / derivative_scale);
            errors[2] = std::max(errors[2], std::abs(actual_q1 - expected_q1) / velocity_scale);
            errors[3] = std::max(errors[3], std::abs(actual_q2 - expected_q2) / velocity_scale);

            mean_q1 += actual_q1;
            expected_mean_q1 += expected_q1;
        }
    }

    const double cell_count = static_cast<double>(layout.local_physical_nx) * layout.local_physical_ny;
    mean_q1 /= cell_count;
    expected_mean_q1 /= cell_count;

    bool passed = finite;
    for (const double error : errors) {
        passed = passed && error <= tolerance;
    }

    if (mode == TestMode::Zonal) {
        // This is an arithmetic test diagnostic, not a proposed spherical
        // averaging rule. A mean subtraction would incorrectly erase flow.
        passed = passed && std::isfinite(mean_q1);
        passed = passed && mean_q1 > 10.0;
        passed = passed && std::abs(mean_q1 - expected_mean_q1) / zonal_speed <= tolerance;

        std::printf("%s zonal physical_mean=%.17e expected_mean=%.17e\n",
            geometry.name(), mean_q1, expected_mean_q1);
    }

    std::printf("%s mode=%s div_error=%.3e curl_error=%.3e physical_q1_error=%.3e physical_q2_error=%.3e %s\n",
        geometry.name(), mode_name(mode), errors[0], errors[1], errors[2], errors[3], passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}

int run_tests() {
    const auto layout = make_layout();

    const Real radius = real(6371220.0);
    const Real south_edge = real(-0.6);

    const CartesianGeometry cartesian(layout, real(2.0), real(3.0));
    const RegularLatLonGeometry regular_latlon(layout, real(0.1), real(0.075), real(0.0), south_edge, radius);

    const std::array<TestMode, 5> modes = {
        TestMode::Constant,
        TestMode::Rotational,
        TestMode::Divergent,
        TestMode::Mixed,
        TestMode::Zonal
    };

    int failures = 0;

    for (const TestMode mode : modes) {
        failures += run_case(cartesian, mode, 1.0, 0.0);
        failures += run_case(regular_latlon, mode, static_cast<double>(radius), static_cast<double>(south_edge));
    }

    return failures;
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int local_failures = 0;

    try {
        local_failures = run_tests();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test_horizontal_wind_consistency: %s\n", error.what());
        local_failures = 1;
    }

    int global_failures = 0;
    MPI_Allreduce(&local_failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::finalize();
    MPI_Finalize();

    return global_failures == 0 ? 0 : 1;
}
