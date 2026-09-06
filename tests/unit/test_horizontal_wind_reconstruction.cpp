#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/HorizontalWindReconstruction.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>

namespace {

using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometry;
using VVM::Core::Geometry::RegularLatLonGeometry;
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

void test_reconstruction(const HorizontalGeometry& geometry, const bool spherical,
    const VVM::Real radius, const VVM::Real south_edge) {

    const auto layout = geometry.layout();
    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;
    const VVM::Real dq1 = geometry.dq1();
    const VVM::Real dq2 = geometry.dq2();

    const VVM::Real radius_squared = radius * radius;
    const VVM::Real amplitude = spherical ? radius_squared * VVM::real(1.0e-5) : VVM::real(1.0);
    const VVM::Real component_scale = spherical ? VVM::real(1.0e-5) : VVM::real(1.0);
    const VVM::Real tolerance = sizeof(VVM::Real) == sizeof(double) ? VVM::real(2.0e-11) : VVM::real(2.0e-4);

    const auto reconstruction = make_horizontal_wind_reconstruction_device_view(geometry);

    Kokkos::View<VVM::Real**> psi("reconstruction_psi", ny, nx);
    Kokkos::View<VVM::Real**> chi("reconstruction_chi", ny, nx);
    Kokkos::View<VVM::Real**> q1("reconstructed_q1_at_u", ny, nx);
    Kokkos::View<VVM::Real**> q2("reconstructed_q2_at_v", ny, nx);

    auto psi_host = Kokkos::create_mirror_view(psi);
    auto chi_host = Kokkos::create_mirror_view(chi);

    for (int mode = 0; mode < 4; ++mode) {
        const VVM::Real psi_factor = (mode & 1) ? amplitude : VVM::real(0.0);
        const VVM::Real chi_factor = (mode & 2) ? amplitude : VVM::real(0.0);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const VVM::Real x = static_cast<VVM::Real>(i - h);
                const VVM::Real y = static_cast<VVM::Real>(j - h);

                psi_host(j, i) = VVM::real(7.0) * amplitude +
                    psi_factor * (VVM::real(0.25) * x * x - VVM::real(0.375) * y * y + VVM::real(0.125) * x * y);

                chi_host(j, i) = VVM::real(-3.0) * amplitude +
                    chi_factor * (-VVM::real(0.125) * x * x + VVM::real(0.25) * y * y - VVM::real(0.0625) * x * y);
            }
        }

        Kokkos::deep_copy(psi, psi_host);
        Kokkos::deep_copy(chi, chi_host);

        const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h});
        const auto compact_policy =
            Kokkos::Experimental::require(policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

        Kokkos::parallel_for("TestHorizontalWindReconstruction", compact_policy,
            KOKKOS_LAMBDA(const int j, const int i) {
                q1(j, i) = reconstruction.calculate_contravariant_q1_at_u(psi, chi, j, i);
                q2(j, i) = reconstruction.calculate_contravariant_q2_at_v(psi, chi, j, i);
            });

        const auto q1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q1);
        const auto q2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q2);

        VVM::Real maximum_normalized_error = VVM::real(0.0);
        bool finite = true;
        bool constant_is_zero = true;

        for (int j = h; j < ny - h; ++j) {
            for (int i = h; i < nx - h; ++i) {
                const VVM::Real x = static_cast<VVM::Real>(i - h);
                const VVM::Real y = static_cast<VVM::Real>(j - h);

                // Exact polynomial differences for the required staggering.
                const VVM::Real psi_q2_backward =
                    psi_factor * (-VVM::real(0.375) * (VVM::real(2.0) * y - VVM::real(1.0)) + VVM::real(0.125) * x) / dq2;

                const VVM::Real psi_q1_backward =
                    psi_factor * (VVM::real(0.25) * (VVM::real(2.0) * x - VVM::real(1.0)) + VVM::real(0.125) * y) / dq1;

                const VVM::Real chi_q1_forward =
                    chi_factor * (-VVM::real(0.125) * (VVM::real(2.0) * x + VVM::real(1.0)) - VVM::real(0.0625) * y) / dq1;

                const VVM::Real chi_q2_forward =
                    chi_factor * (VVM::real(0.25) * (VVM::real(2.0) * y + VVM::real(1.0)) - VVM::real(0.0625) * x) / dq2;

                VVM::Real jacobian_u = VVM::real(1.0);
                VVM::Real jacobian_v = VVM::real(1.0);
                VVM::Real g11_u = VVM::real(1.0);
                VVM::Real g22_v = VVM::real(1.0);

                if (spherical) {
                    // U is latitude-centered; V is staggered in latitude.
                    const VVM::Real latitude_u = south_edge + (y + VVM::real(0.5)) * dq2;
                    const VVM::Real latitude_v = south_edge + (y + VVM::real(1.0)) * dq2;
                    const VVM::Real cos_u = std::cos(latitude_u);
                    const VVM::Real cos_v = std::cos(latitude_v);

                    jacobian_u = radius_squared * cos_u;
                    jacobian_v = radius_squared * cos_v;
                    g11_u = VVM::real(1.0) / (radius_squared * cos_u * cos_u);
                    g22_v = VVM::real(1.0) / radius_squared;
                }

                const VVM::Real expected_q1 = -psi_q2_backward / jacobian_u + g11_u * chi_q1_forward;
                const VVM::Real expected_q2 = psi_q1_backward / jacobian_v + g22_v * chi_q2_forward;

                if (!std::isfinite(q1_host(j, i)) || !std::isfinite(q2_host(j, i))) {
                    finite = false;
                    continue;
                }

                const VVM::Real scale_q1 = std::max(component_scale, std::abs(expected_q1));
                const VVM::Real scale_q2 = std::max(component_scale, std::abs(expected_q2));

                maximum_normalized_error = std::max(
                    maximum_normalized_error, std::abs(q1_host(j, i) - expected_q1) / scale_q1);
                maximum_normalized_error = std::max(
                    maximum_normalized_error, std::abs(q2_host(j, i) - expected_q2) / scale_q2);

                if (mode == 0 && (q1_host(j, i) != VVM::real(0.0) || q2_host(j, i) != VVM::real(0.0))) {
                    constant_is_zero = false;
                }
            }
        }

        const bool passed = finite && constant_is_zero && maximum_normalized_error <= tolerance;

        std::printf("%s mode=%d normalized_error=%.17e %s\n",
            geometry.name(), mode, static_cast<double>(maximum_normalized_error), passed ? "PASS" : "FAIL");

        if (!passed) {
            ++failures;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        try {
            const auto layout = make_layout();

            const CartesianGeometry cartesian(layout, VVM::real(2.0), VVM::real(3.0));
            test_reconstruction(cartesian, false, VVM::real(1.0), VVM::real(0.0));

            const VVM::Real radius = VVM::real(6371220.0);
            const VVM::Real south_edge = VVM::real(-0.3);

            const RegularLatLonGeometry rll(layout, VVM::real(0.08), VVM::real(0.04),
                VVM::real(0.0), south_edge, radius);

            test_reconstruction(rll, true, radius, south_edge);
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return failures == 0 ? 0 : 1;
}
