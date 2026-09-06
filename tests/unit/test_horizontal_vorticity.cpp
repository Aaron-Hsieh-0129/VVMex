#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/HorizontalVorticity.hpp"

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

template<typename Layout>
void test_vorticity(const HorizontalGeometry& geometry, bool spherical, Real radius, Real south_edge, const char* layout_name) {
    using View3D = Kokkos::View<Real***, Layout>;

    const auto layout = geometry.layout();
    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;
    constexpr int nz = 5;

    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();
    const auto operation = make_horizontal_vorticity_device_view(geometry);

    View3D w("horizontal_vorticity_w", nz, ny, nx);
    View3D u1("horizontal_vorticity_covariant_q1", nz, ny, nx);
    View3D u2("horizontal_vorticity_covariant_q2", nz, ny, nx);
    View3D omega1("horizontal_vorticity_q1", nz, ny, nx);
    View3D omega2("horizontal_vorticity_q2", nz, ny, nx);
    View3D recovered1("horizontal_vorticity_recovered_q1", nz, ny, nx);
    View3D recovered2("horizontal_vorticity_recovered_q2", nz, ny, nx);

    Kokkos::View<Real*> spacing("horizontal_vorticity_spacing", nz - 1);
    Kokkos::View<Real*> inverse_spacing("horizontal_vorticity_inverse_spacing", nz - 1);

    auto w_host = Kokkos::create_mirror_view(w);
    auto u1_host = Kokkos::create_mirror_view(u1);
    auto u2_host = Kokkos::create_mirror_view(u2);
    auto spacing_host = Kokkos::create_mirror_view(spacing);
    auto inverse_spacing_host = Kokkos::create_mirror_view(inverse_spacing);

    const Real tolerance = sizeof(Real) == sizeof(double) ? real(1e-10) : real(5e-4);

    for (bool stretched : {false, true}) {
        const std::array<Real, nz> z = stretched
            ? std::array<Real, nz>{real(0), real(40), real(120), real(240), real(400)}
            : std::array<Real, nz>{real(0), real(100), real(200), real(300), real(400)};

        for (int k = 0; k < nz - 1; ++k) {
            spacing_host(k) = z[k + 1] - z[k];
            inverse_spacing_host(k) = real(1.0) / spacing_host(k);
        }

        Kokkos::deep_copy(spacing, spacing_host);
        Kokkos::deep_copy(inverse_spacing, inverse_spacing_host);

        for (int mode = 0; mode < 4; ++mode) {
            const Real shear_e = (mode & 1) ? real(3e-4) : real(0.0);
            const Real shear_n = (mode & 1) ? real(-4e-4) : real(0.0);
            const Real coordinate_scale = spherical ? radius : real(1.0);
            const Real a = (mode & 2) ? coordinate_scale * real(1e-4) : real(0.0);
            const Real b = (mode & 2) ? coordinate_scale * real(-2e-4) : real(0.0);

            // Manufacture physical winds:
            //   u_E = 12 + shear_e*z
            //   u_N = -8 + shear_n*z
            // and physical vertical velocity:
            //   w = a*q1 + b*q2.
            //
            // Convert the horizontal winds to covariant components using
            // analytic Cartesian/RLL scale factors, independently of the operator.
            for (int j = 0; j < ny; ++j) {
                const Real global_j = static_cast<Real>(layout.global_start_j + j - h);
                const Real q2 = south_edge + (global_j + real(0.5)) * dq2;
                const Real h1_at_u = spherical ? radius * std::cos(q2) : real(1.0);
                const Real h2 = spherical ? radius : real(1.0);

                for (int i = 0; i < nx; ++i) {
                    const Real global_i = static_cast<Real>(layout.global_start_i + i - h);
                    const Real q1 = (global_i + real(0.5)) * dq1;

                    for (int k = 0; k < nz; ++k) {
                        w_host(k, j, i) = a * q1 + b * q2;
                        u1_host(k, j, i) = h1_at_u * (real(12.0) + shear_e * z[k]);
                        u2_host(k, j, i) = h2 * (real(-8.0) + shear_n * z[k]);
                    }
                }
            }

            Kokkos::deep_copy(w, w_host);
            Kokkos::deep_copy(u1, u1_host);
            Kokkos::deep_copy(u2, u2_host);

            const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h});
            const auto compact_policy = Kokkos::Experimental::require(
                policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

            Kokkos::parallel_for("TestHorizontalVorticity", compact_policy,
                KOKKOS_LAMBDA(const int j, const int i) {
                    for (int k = 0; k < nz - 1; ++k) {
                        omega1(k, j, i) = operation.calculate_contravariant_q1_at_v(w, u2, k, j, i, inverse_spacing(k));
                        omega2(k, j, i) = operation.calculate_contravariant_q2_at_u(w, u1, k, j, i, inverse_spacing(k));
                    }
                }
            );

            Kokkos::parallel_for("TestCovariantWindDownwardRecovery", compact_policy,
                KOKKOS_LAMBDA(const int j, const int i) {
                    recovered1(nz - 1, j, i) = u1(nz - 1, j, i);
                    recovered2(nz - 1, j, i) = u2(nz - 1, j, i);

                    for (int k = nz - 2; k >= 0; --k) {
                        const Real du1_dz = operation.calculate_covariant_q1_vertical_shear_at_u(w, omega2, k, j, i);
                        const Real du2_dz = operation.calculate_covariant_q2_vertical_shear_at_v(w, omega1, k, j, i);

                        recovered1(k, j, i) = recovered1(k + 1, j, i) - du1_dz * spacing(k);
                        recovered2(k, j, i) = recovered2(k + 1, j, i) - du2_dz * spacing(k);
                    }
                }
            );

            Kokkos::fence();

            const auto omega1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), omega1);
            const auto omega2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), omega2);
            const auto recovered1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recovered1);
            const auto recovered2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recovered2);

            Real vorticity_error = real(0.0);
            Real recovery_error = real(0.0);
            bool finite = true;
            bool zero_case_correct = true;

            for (int j = h; j < ny - h; ++j) {
                const Real global_j = static_cast<Real>(layout.global_start_j + j - h);
                const Real phi_u = south_edge + (global_j + real(0.5)) * dq2;
                const Real phi_v = south_edge + (global_j + real(1.0)) * dq2;

                const Real h1_at_u = spherical ? radius * std::cos(phi_u) : real(1.0);
                const Real h1_at_v = spherical ? radius * std::cos(phi_v) : real(1.0);
                const Real h2 = spherical ? radius : real(1.0);

                // Independent analytic physical-vorticity components.
                const Real expected_e = b / h2 - shear_n;
                const Real expected_n = shear_e - a / h1_at_u;

                for (int i = h; i < nx - h; ++i) {
                    for (int k = 0; k < nz - 1; ++k) {
                        const Real actual_e = h1_at_v * omega1_host(k, j, i);
                        const Real actual_n = h2 * omega2_host(k, j, i);

                        if (!std::isfinite(actual_e) || !std::isfinite(actual_n)) {
                            finite = false;
                        } else {
                            vorticity_error = std::max(vorticity_error, std::abs(actual_e - expected_e) / real(1e-3));
                            vorticity_error = std::max(vorticity_error, std::abs(actual_n - expected_n) / real(1e-3));
                        }

                        if (mode == 0) {
                            zero_case_correct = zero_case_correct
                                && omega1_host(k, j, i) == real(0.0)
                                && omega2_host(k, j, i) == real(0.0);
                        }
                    }

                    for (int k = 0; k < nz; ++k) {
                        const Real expected_u = real(12.0) + shear_e * z[k];
                        const Real expected_v = real(-8.0) + shear_n * z[k];
                        const Real actual_u = recovered1_host(k, j, i) / h1_at_u;
                        const Real actual_v = recovered2_host(k, j, i) / h2;

                        if (!std::isfinite(actual_u) || !std::isfinite(actual_v)) {
                            finite = false;
                        } else {
                            recovery_error = std::max(recovery_error,
                                std::abs(actual_u - expected_u) / std::max(real(1.0), std::abs(expected_u)));
                            recovery_error = std::max(recovery_error,
                                std::abs(actual_v - expected_v) / std::max(real(1.0), std::abs(expected_v)));
                        }
                    }
                }
            }

            const bool passed = finite && zero_case_correct
                && vorticity_error <= tolerance && recovery_error <= tolerance;

            std::printf("%s %s stretched=%d mode=%d vorticity_error=%.3e recovery_error=%.3e %s\n",
                geometry.name(), layout_name, static_cast<int>(stretched), mode,
                static_cast<double>(vorticity_error), static_cast<double>(recovery_error),
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

        test_vorticity<Kokkos::LayoutLeft>(cartesian, false, real(1.0), real(0.0), "LayoutLeft");
        test_vorticity<Kokkos::LayoutRight>(cartesian, false, real(1.0), real(0.0), "LayoutRight");
        test_vorticity<Kokkos::LayoutLeft>(rll, true, radius, south_edge, "LayoutLeft");
        test_vorticity<Kokkos::LayoutRight>(rll, true, radius, south_edge, "LayoutRight");
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_horizontal_vorticity: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) std::puts("test_horizontal_vorticity: PASS");

    Kokkos::finalize();
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
