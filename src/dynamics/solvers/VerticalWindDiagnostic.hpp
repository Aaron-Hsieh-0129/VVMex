#ifndef VVM_DYNAMICS_SOLVERS_VERTICAL_WIND_DIAGNOSTIC_HPP
#define VVM_DYNAMICS_SOLVERS_VERTICAL_WIND_DIAGNOSTIC_HPP

#include <cmath>
#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM {
namespace Dynamics {

// Coefficients for one vertical tridiagonal row. The line unknown is the
// density-weighted vertical velocity rho_up(k) * w(k). The horizontal direct
// coefficients are retained here because they enter the line diagonal. The
// nonorthogonal g^12 contribution has no center coefficient in the CVVM
// stencil and is evaluated by VerticalWindDiagnosticDeviceView.
struct VerticalWindEllipticRow {
    Real lower = real(0.0);
    Real diagonal = real(0.0);
    Real upper = real(0.0);
    Real east = real(0.0);
    Real west = real(0.0);
    Real north = real(0.0);
    Real south = real(0.0);
    Real shift = real(0.0);

    template <typename View>
    KOKKOS_INLINE_FUNCTION Real
    horizontal_neighbors(const View& previous, int k, int j, int i) const noexcept {

        return east * previous(k, j, i + 1) + west * previous(k, j, i - 1) +
               north * previous(k, j + 1, i) + south * previous(k, j - 1, i);
    }

    template <typename View>
    KOKKOS_INLINE_FUNCTION Real
    line_rhs(const View& previous, Real weighted_rhs, int k, int j, int i) const noexcept {

        return shift * previous(k, j, i) + weighted_rhs + horizontal_neighbors(previous, k, j, i);
    }

    template <typename View, typename Profile>
    KOKKOS_INLINE_FUNCTION Real
    jacobi_value(const View& previous,
        const Profile& rhobar_up,
        Real weighted_rhs,
        int k,
        int j,
        int i) const noexcept {

        const Real denominator = diagonal * rhobar_up(k) - shift;

        if (denominator == real(0.0)) {
            return previous(k, j, i);
        }

        const Real vertical = -lower * rhobar_up(k - 1) * previous(k - 1, j, i) -
                              upper * rhobar_up(k + 1) * previous(k + 1, j, i);

        return (weighted_rhs + horizontal_neighbors(previous, k, j, i) + vertical) / denominator;
    }
};

// Generalized stationary-horizontal-chart arguments for the vertical wind
// diagnostic and elliptic solve.
//
// Metric conventions:
//
//     J               = sqrt(g)
//     weighted_contra = J * g^ij
//     g_cov           = g_ij
//
// For a two-dimensional horizontal metric:
//
//     J^2 g^11 =  g_22
//     J^2 g^12 = -g_12
//     J^2 g^22 =  g_11
//
// CVVM RELAX_3D uses both J*g^ij and J^2*g^ij. The latter is recovered
// exactly through the 2-D metric identity above.
//
// No allocation, communication, synchronization, or iteration control belongs
// to this object.
struct VerticalWindDiagnosticDeviceView {
    Core::Geometry::GeometryField2D jacobian_t;
    Core::Geometry::GeometryField2D jacobian_u;
    Core::Geometry::GeometryField2D jacobian_v;
    Core::Geometry::GeometryField2D jacobian_z;

    Core::Geometry::SymmetricTensorDeviceView g_cov_u;
    Core::Geometry::SymmetricTensorDeviceView g_cov_v;

    Core::Geometry::SymmetricTensorDeviceView weighted_contra_u;
    Core::Geometry::SymmetricTensorDeviceView weighted_contra_v;

    // Legacy physical-vorticity compatibility factors.
    //
    // Production generalized dynamics should use the canonical
    // contravariant methods below. These retain Cartesian/RLL physical
    // compatibility paths.
    Core::Geometry::GeometryField2D legacy_h1_u;
    Core::Geometry::GeometryField2D legacy_h1_v;
    Core::Geometry::GeometryField2D legacy_h2_u;

    Real dq1 = real(1.0);
    Real dq2 = real(1.0);

    // Legacy physical VVM convention:
    //
    //     xi  = physical omega_1
    //     eta = -physical omega_2
    //
    // This path intentionally retains the previous Cartesian/RLL orthogonal
    // factorization. Nonorthogonal geometry must use the canonical method.
    template <typename XiView, typename EtaView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_weighted_rhs_at_t(
        const XiView& xi, const EtaView& eta, int k, int j, int i) const noexcept {

        const Real h2 = legacy_h2_u(j, i);

        return -h2 * (eta(k, j, i) - eta(k, j, i - 1)) / dq1 -
               (legacy_h1_v(j, i) * xi(k, j, i) - legacy_h1_v(j - 1, i) * xi(k, j - 1, i)) / dq2;
    }

    // Generalized CVVM RELAX_3D YTEM.
    //
    // Persistent VVM convention:
    //
    //     xi_con  =  omega^1
    //     eta_con = -omega^2
    //
    // The result already contains J and must NOT be multiplied by J again.
    template <typename XiConView, typename EtaConView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_weighted_rhs_from_vvm_contravariant_at_t(const XiConView& xi_con,
        const EtaConView& eta_con,
        const int k,
        const int j,
        const int i) const noexcept {

        // CVVM:
        //
        //     Z3DX = omega^1
        //     Z3DY = omega^2 = -eta_con
        //     GG   = J^2 g^ij
        //
        // In 2-D:
        //
        //     GG11 =  g22
        //     GG12 = -g12
        //     GG22 =  g11

        const Real omega2_east = -eta_con(k, j, i);
        const Real omega2_west = -eta_con(k, j, i - 1);
        const Real omega1_north = xi_con(k, j, i);
        const Real omega1_south = xi_con(k, j - 1, i);

        const Real direct_q1 =
            (g_cov_u.a22(j, i) * omega2_east - g_cov_u.a22(j, i - 1) * omega2_west) / dq1;
        const Real direct_q2 =
            -(g_cov_v.a11(j, i) * omega1_north - g_cov_v.a11(j - 1, i) * omega1_south) / dq2;

        // CVVM mixed q1 derivative:
        //
        //   -d_q1( GG12 * omega1 )
        //
        // evaluated with the native staggered four-point interpolation.
        const Real cross_q1 = -((-g_cov_v.a12(j, i + 1)) * xi_con(k, j, i + 1) +
                                  (-g_cov_v.a12(j - 1, i + 1)) * xi_con(k, j - 1, i + 1) -
                                  (-g_cov_v.a12(j, i - 1)) * xi_con(k, j, i - 1) -
                                  (-g_cov_v.a12(j - 1, i - 1)) * xi_con(k, j - 1, i - 1)) /
                              (real(4.0) * dq1);

        // CVVM mixed q2 derivative:
        //
        //   +d_q2( GG12 * omega2 )
        const Real cross_q2 = ((-g_cov_u.a12(j + 1, i)) * (-eta_con(k, j + 1, i)) +
                                  (-g_cov_u.a12(j + 1, i - 1)) * (-eta_con(k, j + 1, i - 1)) -
                                  (-g_cov_u.a12(j - 1, i)) * (-eta_con(k, j - 1, i)) -
                                  (-g_cov_u.a12(j - 1, i - 1)) * (-eta_con(k, j - 1, i - 1))) /
                              (real(4.0) * dq2);

        return direct_q1 + direct_q2 + cross_q1 + cross_q2;
    }

    // Legacy physical horizontal-vorticity divergence.
    template <typename XiView, typename EtaView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_vorticity_divergence_at_z(
        const XiView& xi, const EtaView& eta, int k, int j, int i) const noexcept {

        const Real h2 = legacy_h2_u(j, i);

        return (h2 * (xi(k, j, i + 1) - xi(k, j, i)) / dq1 -
                   (legacy_h1_u(j + 1, i) * eta(k, j + 1, i) - legacy_h1_u(j, i) * eta(k, j, i)) /
                       dq2) /
               jacobian_z(j, i);
    }

    // Generalized divergence:
    //
    //   div_h omega_h =
    //       1/J [
    //           d_q1(J omega^1)
    //         + d_q2(J omega^2)
    //       ].
    //
    // omega^1 is native at V, omega^2 is native at U.
    template <typename Omega1View, typename Omega2View>
    KOKKOS_INLINE_FUNCTION Real
    calculate_vorticity_divergence_from_contravariant_at_z(const Omega1View& omega1,
        const Omega2View& omega2,
        const int k,
        const int j,
        const int i) const noexcept {
        const Real flux_q1 =
            (jacobian_v(j, i + 1) * omega1(k, j, i + 1) - jacobian_v(j, i) * omega1(k, j, i)) / dq1;

        const Real flux_q2 =
            (jacobian_u(j + 1, i) * omega2(k, j + 1, i) - jacobian_u(j, i) * omega2(k, j, i)) / dq2;

        return (flux_q1 + flux_q2) / jacobian_z(j, i);
    }

    template <typename XiView, typename EtaView, typename Profile, typename ZetaView>
    KOKKOS_INLINE_FUNCTION void
    integrate_zeta_column(const XiView& xi,
        const EtaView& eta,
        const Profile& spacing,
        const ZetaView& zeta,
        int bottom,
        int top,
        int j,
        int i,
        bool extend_upper_ghost) const noexcept {

        for (int k = top - 1; k >= bottom; --k) {
            zeta(k, j, i) = zeta(k + 1, j, i) +
                            spacing(k) * calculate_vorticity_divergence_at_z(xi, eta, k, j, i);
        }

        if (extend_upper_ghost) {
            zeta(top + 1, j, i) =
                zeta(top, j, i) -
                spacing(top) * calculate_vorticity_divergence_at_z(xi, eta, top, j, i);
        }
    }

    template <typename Omega1View, typename Omega2View, typename Profile, typename ZetaView>
    KOKKOS_INLINE_FUNCTION void
    integrate_zeta_column_from_contravariant(const Omega1View& omega1,
        const Omega2View& omega2,
        const Profile& spacing,
        const ZetaView& zeta,
        const int bottom,
        const int top,
        const int j,
        const int i,
        const bool extend_upper_ghost) const noexcept {

        for (int k = top - 1; k >= bottom; --k) {
            zeta(k, j, i) =
                zeta(k + 1, j, i) +
                spacing(k) *
                    calculate_vorticity_divergence_from_contravariant_at_z(omega1, omega2, k, j, i);
        }

        if (extend_upper_ghost) {
            zeta(top + 1, j, i) =
                zeta(top, j, i) -
                spacing(top) * calculate_vorticity_divergence_from_contravariant_at_z(omega1,
                                   omega2,
                                   top,
                                   j,
                                   i);
        }
    }

    // Horizontal neighbor part of CVVM RELAX_3D.
    //
    // Direct terms use J*g^11 and J*g^22.
    // Mixed terms use J*g^12 and retain the exact native CVVM staggering.
    template <typename View>
    KOKKOS_INLINE_FUNCTION Real
    calculate_horizontal_neighbors_at_t(
        const View& previous, const int k, const int j, const int i) const noexcept {

        const Real rdq1 = real(1.0) / dq1;

        const Real rdq2 = real(1.0) / dq2;

        const Real direct = weighted_contra_u.a11(j, i) * previous(k, j, i + 1) * rdq1 * rdq1 +
                            weighted_contra_u.a11(j, i - 1) * previous(k, j, i - 1) * rdq1 * rdq1 +
                            weighted_contra_v.a22(j, i) * previous(k, j + 1, i) * rdq2 * rdq2 +
                            weighted_contra_v.a22(j - 1, i) * previous(k, j - 1, i) * rdq2 * rdq2;

        const Real cross =
            (weighted_contra_v.a12(j, i + 1) * (previous(k, j + 1, i + 1) - previous(k, j, i + 1)) +

                weighted_contra_v.a12(j - 1, i + 1) *
                    (previous(k, j, i + 1) - previous(k, j - 1, i + 1)) -

                weighted_contra_v.a12(j, i - 1) *
                    (previous(k, j + 1, i - 1) - previous(k, j, i - 1)) -

                weighted_contra_v.a12(j - 1, i - 1) *
                    (previous(k, j, i - 1) - previous(k, j - 1, i - 1)) +

                weighted_contra_u.a12(j + 1, i) *
                    (previous(k, j + 1, i + 1) - previous(k, j + 1, i)) +

                weighted_contra_u.a12(j + 1, i - 1) *
                    (previous(k, j + 1, i) - previous(k, j + 1, i - 1)) -

                weighted_contra_u.a12(j - 1, i) *
                    (previous(k, j - 1, i + 1) - previous(k, j - 1, i)) -

                weighted_contra_u.a12(j - 1, i - 1) *
                    (previous(k, j - 1, i) - previous(k, j - 1, i - 1))) /
            (real(4.0) * dq1 * dq2);

        return direct + cross;
    }

    // Vertical tridiagonal coefficients.
    //
    // The coefficients are now (j,i,k), not merely (j,k), because J and
    // horizontal metrics may vary in both horizontal directions.
    template <typename Rho, typename RhoUp, typename FlexMid, typename FlexUp>
    KOKKOS_INLINE_FUNCTION VerticalWindEllipticRow
    calculate_row_at_t(const Rho& rho,
        const RhoUp& rho_up,
        const FlexMid& flex_mid,
        const FlexUp& flex_up,
        Real inverse_dz,
        Real shift,
        int k,
        int j,
        int i) const noexcept {

        VerticalWindEllipticRow row;

        row.east = weighted_contra_u.a11(j, i) / (dq1 * dq1);
        row.west = weighted_contra_u.a11(j, i - 1) / (dq1 * dq1);
        row.north = weighted_contra_v.a22(j, i) / (dq2 * dq2);
        row.south = weighted_contra_v.a22(j - 1, i) / (dq2 * dq2);
        row.shift = shift;

        const Real vertical = jacobian_t(j, i) * flex_up(k) * inverse_dz * inverse_dz;
        row.lower = -vertical * flex_mid(k) / rho(k);
        row.upper = -vertical * flex_mid(k + 1) / rho(k + 1);
        row.diagonal = (shift + row.east + row.west + row.north + row.south) / rho_up(k) -
                       row.lower - row.upper;

        return row;
    }
};

inline VerticalWindDiagnosticDeviceView
make_vertical_wind_diagnostic_device_view(const Core::Geometry::HorizontalGeometry& geometry) {

    using namespace Core::Geometry;

    if (geometry.layout().halo < 1 || geometry.layout().local_physical_nx < 2 ||
        geometry.layout().local_physical_ny < 2) {
        throw std::invalid_argument("Vertical wind diagnostic requires "
                                    "two active horizontal axes and at least "
                                    "one halo cell.");
    }

    if (!std::isfinite(geometry.dq1()) || !std::isfinite(geometry.dq2()) ||
        geometry.dq1() <= real(0.0) || geometry.dq2() <= real(0.0)) {
        throw std::invalid_argument("Invalid horizontal "
                                    "computational-coordinate increments.");
    }

    const auto t = geometry.device_view(HorizontalLocation::T);
    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);
    const auto z = geometry.device_view(HorizontalLocation::Z);

    VerticalWindDiagnosticDeviceView result;

    result.jacobian_t = t.sqrt_g;
    result.jacobian_u = u.sqrt_g;
    result.jacobian_v = v.sqrt_g;
    result.jacobian_z = z.sqrt_g;

    result.g_cov_u = u.g_cov;
    result.g_cov_v = v.g_cov;

    result.weighted_contra_u = u.sqrt_g_g_contra;
    result.weighted_contra_v = v.sqrt_g_g_contra;

    result.legacy_h1_u = u.contravariant_to_physical.a11;
    result.legacy_h1_v = v.contravariant_to_physical.a11;
    result.legacy_h2_u = u.contravariant_to_physical.a22;

    result.dq1 = geometry.dq1();
    result.dq2 = geometry.dq2();
    return result;
}

} // namespace Dynamics
} // namespace VVM

#endif
