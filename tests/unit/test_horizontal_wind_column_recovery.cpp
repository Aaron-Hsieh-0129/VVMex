#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/HorizontalVorticity.hpp"
#include "dynamics/operators/HorizontalWindReconstruction.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometry;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::Operators::make_horizontal_vorticity_device_view;
using VVM::Dynamics::Operators::make_horizontal_wind_reconstruction_device_view;

int failures = 0;

HorizontalDomainLayout make_layout() {
    HorizontalDomainLayout layout;
    layout.global_nx = 12;
    layout.global_ny = 10;
    layout.local_physical_nx = 12;
    layout.local_physical_ny = 10;
    layout.halo = 2;
    return layout;
}

void test_column(const HorizontalGeometry& geometry, bool spherical, Real radius, Real south_edge) {
    const auto layout = geometry.layout();
    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;
    constexpr int nz = 5;
    constexpr int top = nz - 1;

    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();
    const Real amplitude = spherical ? radius * radius * real(1e-6) : real(1.0);
    const Real tolerance = sizeof(Real) == sizeof(double) ? real(1e-10) : real(5e-4);

    const auto reconstruction = make_horizontal_wind_reconstruction_device_view(geometry);
    const auto vorticity = make_horizontal_vorticity_device_view(geometry);

    Kokkos::View<Real**> psi("column_psi", ny, nx);
    Kokkos::View<Real**> chi("column_chi", ny, nx);
    Kokkos::View<Real**> top_contravariant1("column_top_contravariant1", ny, nx);
    Kokkos::View<Real**> top_contravariant2("column_top_contravariant2", ny, nx);

    Kokkos::View<Real***> w("column_w", nz, ny, nx);
    Kokkos::View<Real***> omega1("column_omega1", nz, ny, nx);
    Kokkos::View<Real***> omega2("column_omega2", nz, ny, nx);
    Kokkos::View<Real***> covariant1("column_covariant1", nz, ny, nx);
    Kokkos::View<Real***> covariant2("column_covariant2", nz, ny, nx);
    Kokkos::View<Real*> spacing("column_spacing", nz - 1);

    auto psi_host = Kokkos::create_mirror_view(psi);
    auto chi_host = Kokkos::create_mirror_view(chi);
    auto w_host = Kokkos::create_mirror_view(w);
    auto omega1_host = Kokkos::create_mirror_view(omega1);
    auto omega2_host = Kokkos::create_mirror_view(omega2);
    auto spacing_host = Kokkos::create_mirror_view(spacing);

    const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h});
    const auto compact_policy = Kokkos::Experimental::require(
        policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    for (bool stretched : {false, true}) {
        const std::array<Real, nz> z = stretched
            ? std::array<Real, nz>{real(0), real(40), real(120), real(240), real(400)}
            : std::array<Real, nz>{real(0), real(100), real(200), real(300), real(400)};

        for (int k = 0; k < nz - 1; ++k) {
            spacing_host(k) = z[k + 1] - z[k];
        }
        Kokkos::deep_copy(spacing, spacing_host);

        for (int mode = 0; mode < 4; ++mode) {
            const Real psi_factor = (mode & 1) ? amplitude : real(0.0);
            const Real chi_factor = (mode & 2) ? amplitude : real(0.0);
            const Real shear_e = mode == 0 ? real(0.0) : real(2e-4);
            const Real shear_n = mode == 0 ? real(0.0) : real(-3e-4);
            const Real coordinate_scale = spherical ? radius : real(1.0);
            const Real a = mode == 0 ? real(0.0) : coordinate_scale * real(1e-4);
            const Real b = mode == 0 ? real(0.0) : coordinate_scale * real(-2e-4);

            for (int j = 0; j < ny; ++j) {
                const Real y = static_cast<Real>(layout.global_start_j + j - h);
                const Real phi_u = south_edge + (y + real(0.5)) * dq2;
                const Real phi_v = south_edge + (y + real(1.0)) * dq2;
                const Real h1_u = spherical ? radius * std::cos(phi_u) : real(1.0);
                const Real h1_v = spherical ? radius * std::cos(phi_v) : real(1.0);
                const Real h2 = spherical ? radius : real(1.0);

                for (int i = 0; i < nx; ++i) {
                    const Real x = static_cast<Real>(layout.global_start_i + i - h);
                    const Real q1 = (x + real(0.5)) * dq1;

                    psi_host(j, i) = real(7.0) * amplitude
                        + psi_factor * (real(0.25) * x - real(0.375) * y);
                    chi_host(j, i) = real(-3.0) * amplitude
                        + chi_factor * (real(-0.125) * x + real(0.25) * y);

                    for (int k = 0; k < nz; ++k) {
                        w_host(k, j, i) = a * q1 + b * phi_u;

                        // Independently manufactured contravariant vorticity.
                        // Physical wind shear is shear_e/shear_n.
                        omega1_host(k, j, i) = (b - h2 * shear_n) / (h1_v * h2);
                        omega2_host(k, j, i) = (h1_u * shear_e - a) / (h1_u * h2);
                    }
                }
            }

            Kokkos::deep_copy(psi, psi_host);
            Kokkos::deep_copy(chi, chi_host);
            Kokkos::deep_copy(w, w_host);
            Kokkos::deep_copy(omega1, omega1_host);
            Kokkos::deep_copy(omega2, omega2_host);

            Kokkos::parallel_for("ReconstructCovariantTopWind", compact_policy,
                KOKKOS_LAMBDA(const int j, const int i) {
                    covariant1(top, j, i) = reconstruction.calculate_covariant_q1_at_u(psi, chi, j, i);
                    covariant2(top, j, i) = reconstruction.calculate_covariant_q2_at_v(psi, chi, j, i);

                    top_contravariant1(j, i) = reconstruction.calculate_contravariant_q1_at_u(psi, chi, j, i);
                    top_contravariant2(j, i) = reconstruction.calculate_contravariant_q2_at_v(psi, chi, j, i);
                }
            );

            Kokkos::parallel_for("RecoverCovariantWindColumn", compact_policy,
                KOKKOS_LAMBDA(const int j, const int i) {
                    for (int k = top - 1; k >= 0; --k) {
                        const Real du1_dz = vorticity.calculate_covariant_q1_vertical_shear_at_u(w, omega2, k, j, i);
                        const Real du2_dz = vorticity.calculate_covariant_q2_vertical_shear_at_v(w, omega1, k, j, i);

                        covariant1(k, j, i) = covariant1(k + 1, j, i) - du1_dz * spacing(k);
                        covariant2(k, j, i) = covariant2(k + 1, j, i) - du2_dz * spacing(k);
                    }
                }
            );

            Kokkos::fence();

            const auto u1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), covariant1);
            const auto u2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), covariant2);
            const auto contra1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), top_contravariant1);
            const auto contra2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), top_contravariant2);

            // Analytic derivatives of the manufactured linear potentials.
            const Real dpsi_dq1 = real(0.25) * psi_factor / dq1;
            const Real dpsi_dq2 = real(-0.375) * psi_factor / dq2;
            const Real dchi_dq1 = real(-0.125) * chi_factor / dq1;
            const Real dchi_dq2 = real(0.25) * chi_factor / dq2;

            Real column_error = real(0.0);
            Real representation_error = real(0.0);
            bool finite = true;

            const auto compare = [&](Real actual, Real expected, Real& maximum_error) {
                if (!std::isfinite(actual) || !std::isfinite(expected)) {
                    finite = false;
                    return;
                }

                maximum_error = std::max(maximum_error,
                    std::abs(actual - expected) / std::max(real(1.0), std::abs(expected)));
            };

            for (int j = h; j < ny - h; ++j) {
                const Real y = static_cast<Real>(layout.global_start_j + j - h);
                const Real phi_u = south_edge + (y + real(0.5)) * dq2;
                const Real phi_v = south_edge + (y + real(1.0)) * dq2;
                const Real h1_u = spherical ? radius * std::cos(phi_u) : real(1.0);
                const Real h1_v = spherical ? radius * std::cos(phi_v) : real(1.0);
                const Real h2 = spherical ? radius : real(1.0);

                const Real expected_top_e = -dpsi_dq2 / h2 + dchi_dq1 / h1_u;
                const Real expected_top_n = dpsi_dq1 / h1_v + dchi_dq2 / h2;

                for (int i = h; i < nx - h; ++i) {
                    // Compare representations in physical units:
                    //   u_E = u_1/h1 = h1*u^1
                    //   u_N = u_2/h2 = h2*u^2.
                    compare(u1_host(top, j, i) / h1_u, h1_u * contra1_host(j, i), representation_error);
                    compare(u2_host(top, j, i) / h2, h2 * contra2_host(j, i), representation_error);

                    for (int k = 0; k < nz; ++k) {
                        const Real expected_e = expected_top_e + shear_e * (z[k] - z[top]);
                        const Real expected_n = expected_top_n + shear_n * (z[k] - z[top]);

                        compare(u1_host(k, j, i) / h1_u, expected_e, column_error);
                        compare(u2_host(k, j, i) / h2, expected_n, column_error);
                    }
                }
            }

            const bool passed = finite && column_error <= tolerance && representation_error <= tolerance;

            std::printf("%s stretched=%d mode=%d column_error=%.3e representation_error=%.3e %s\n",
                geometry.name(), static_cast<int>(stretched), mode,
                static_cast<double>(column_error), static_cast<double>(representation_error),
                passed ? "PASS" : "FAIL");

            if (!passed) ++failures;
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    try {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size != 1) throw std::runtime_error("This test requires one MPI rank.");

        const auto layout = make_layout();
        const CartesianGeometry cartesian(layout, real(2.0), real(3.0));

        const Real radius = real(6371220.0);
        const Real south_edge = real(-0.3);
        const RegularLatLonGeometry rll(layout, real(0.08), real(0.04), real(0.0), south_edge, radius);

        test_column(cartesian, false, real(1.0), real(0.0));
        test_column(rll, true, radius, south_edge);
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_horizontal_wind_column_recovery: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) std::puts("test_horizontal_wind_column_recovery: PASS");

    Kokkos::finalize();
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
