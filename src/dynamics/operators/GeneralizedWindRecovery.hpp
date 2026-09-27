#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_WIND_RECOVERY_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_WIND_RECOVERY_HPP

#include <cmath>
#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM::Dynamics::Operators {

// Local CVVM UVTOP_3D reconstruction. psi is at Z, chi at T; output
// components are at U/V. Coordinates are stationary and independent of z.
//
// The off-diagonal contributions are metric-times-derivative products at
// their NATIVE faces, then averaged to the target face. In a varying metric,
// multiplying an averaged derivative by the target metric is not equivalent.
//
// Covariant divergent wind is d(chi) directly. Only the rotational part is
// lowered. Do not lower the complete interpolated contravariant wind: that
// would apply a second interpolation to the divergent part.
//
// One scalar halo in each horizontal direction is needed. Metric halos must
// be valid in the receiving chart. The caller owns topology and BCs.
struct GeneralizedHorizontalWindReconstructionDeviceView {
    using Metric = Core::Geometry::GeometryField2D;

    Metric jacobian_u, jacobian_v;
    Metric weighted_11_u, weighted_12_u, weighted_22_u;
    Metric weighted_11_v, weighted_12_v, weighted_22_v;
    Real dq1 = real(0.0), dq2 = real(0.0);

    template <typename PsiView, typename ChiView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_contravariant_q1_at_u(
        const PsiView& psi, const ChiView& chi, int j, int i) const noexcept {
        const Real rotational = -(psi(j, i) - psi(j - 1, i)) / (jacobian_u(j, i) * dq2);

        const Real direct =
            weighted_11_u(j, i) / jacobian_u(j, i) * (chi(j, i + 1) - chi(j, i)) / dq1;

        const Real cross =
            (weighted_12_v(j, i) / jacobian_v(j, i) * (chi(j + 1, i) - chi(j, i)) +
                weighted_12_v(j - 1, i) / jacobian_v(j - 1, i) * (chi(j, i) - chi(j - 1, i)) +
                weighted_12_v(j, i + 1) / jacobian_v(j, i + 1) *
                    (chi(j + 1, i + 1) - chi(j, i + 1)) +
                weighted_12_v(j - 1, i + 1) / jacobian_v(j - 1, i + 1) *
                    (chi(j, i + 1) - chi(j - 1, i + 1))) /
            (real(4.0) * dq2);

        return rotational + (direct + cross);
    }

    template <typename PsiView, typename ChiView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_contravariant_q2_at_v(
        const PsiView& psi, const ChiView& chi, int j, int i) const noexcept {
        const Real rotational = (psi(j, i) - psi(j, i - 1)) / (jacobian_v(j, i) * dq1);

        const Real direct =
            weighted_22_v(j, i) / jacobian_v(j, i) * (chi(j + 1, i) - chi(j, i)) / dq2;

        const Real cross =
            (weighted_12_u(j, i) / jacobian_u(j, i) * (chi(j, i + 1) - chi(j, i)) +
                weighted_12_u(j, i - 1) / jacobian_u(j, i - 1) * (chi(j, i) - chi(j, i - 1)) +
                weighted_12_u(j + 1, i) / jacobian_u(j + 1, i) *
                    (chi(j + 1, i + 1) - chi(j + 1, i)) +
                weighted_12_u(j + 1, i - 1) / jacobian_u(j + 1, i - 1) *
                    (chi(j + 1, i) - chi(j + 1, i - 1))) /
            (real(4.0) * dq1);

        return rotational + (direct + cross);
    }

    template <typename PsiView, typename ChiView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_covariant_q1_at_u(
        const PsiView& psi, const ChiView& chi, int j, int i) const noexcept {
        // g_11/J = J*g^22; g_12/J = -J*g^12 for a 2D metric.
        const Real direct = -weighted_22_u(j, i) * (psi(j, i) - psi(j - 1, i)) / dq2;

        const Real cross = -(weighted_12_v(j, i + 1) * (psi(j, i + 1) - psi(j, i)) +
                               weighted_12_v(j, i) * (psi(j, i) - psi(j, i - 1)) +
                               weighted_12_v(j - 1, i + 1) * (psi(j - 1, i + 1) - psi(j - 1, i)) +
                               weighted_12_v(j - 1, i) * (psi(j - 1, i) - psi(j - 1, i - 1))) /
                           (real(4.0) * dq1);

        const Real divergent = (chi(j, i + 1) - chi(j, i)) / dq1;

        return (direct + cross) + divergent;
    }

    template <typename PsiView, typename ChiView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_covariant_q2_at_v(
        const PsiView& psi, const ChiView& chi, int j, int i) const noexcept {
        const Real direct = weighted_11_v(j, i) * (psi(j, i) - psi(j, i - 1)) / dq1;

        const Real cross = (weighted_12_u(j + 1, i) * (psi(j + 1, i) - psi(j, i)) +
                               weighted_12_u(j, i) * (psi(j, i) - psi(j - 1, i)) +
                               weighted_12_u(j + 1, i - 1) * (psi(j + 1, i - 1) - psi(j, i - 1)) +
                               weighted_12_u(j, i - 1) * (psi(j, i - 1) - psi(j - 1, i - 1))) /
                           (real(4.0) * dq2);

        const Real divergent = (chi(j + 1, i) - chi(j, i)) / dq2;

        return (direct + cross) + divergent;
    }
};

// Inverse horizontal curl for the covariant wind column:
//   d_z u_1 = d_1 w + J*omega^2,
//   d_z u_2 = d_2 w - J*omega^1.
//
// Input vorticity uses TRUE tensor signs; convert eta_con at the caller.
// No physical horizontal components or diagonal-metric assumptions.
struct GeneralizedCovariantWindShearDeviceView {
    Core::Geometry::GeometryField2D jacobian_u, jacobian_v;
    Real rdq1 = real(0.0), rdq2 = real(0.0);

    template <typename WView, typename Omega2View>
    KOKKOS_INLINE_FUNCTION Real
    calculate_covariant_q1_vertical_shear_at_u(
        const WView& w, const Omega2View& omega2, int k, int j, int i) const noexcept {
        const Real dw_dq1 = (w(k, j, i + 1) - w(k, j, i)) * rdq1;

        return dw_dq1 + jacobian_u(j, i) * omega2(k, j, i);
    }

    template <typename WView, typename Omega1View>
    KOKKOS_INLINE_FUNCTION Real
    calculate_covariant_q2_vertical_shear_at_v(
        const WView& w, const Omega1View& omega1, int k, int j, int i) const noexcept {
        const Real dw_dq2 = (w(k, j + 1, i) - w(k, j, i)) * rdq2;

        return dw_dq2 - jacobian_v(j, i) * omega1(k, j, i);
    }
};

inline void
validate_generalized_wind_recovery_geometry(const Core::Geometry::HorizontalGeometry& geometry) {
    if (geometry.layout().halo < 1 || geometry.layout().local_physical_nx < 1 ||
        geometry.layout().local_physical_ny < 1 || !(geometry.dq1() > real(0.0)) ||
        !(geometry.dq2() > real(0.0)) || !std::isfinite(geometry.dq1()) ||
        !std::isfinite(geometry.dq2()) || !std::isfinite(real(1.0) / geometry.dq1()) ||
        !std::isfinite(real(1.0) / geometry.dq2())) {
        throw std::invalid_argument(
            "Generalized wind recovery requires a nonempty domain, one halo, "
            "and finite positive computational spacing and reciprocals.");
    }
}

inline GeneralizedHorizontalWindReconstructionDeviceView
make_generalized_horizontal_wind_reconstruction_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {
    using Core::Geometry::HorizontalLocation;

    validate_generalized_wind_recovery_geometry(geometry);

    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);

    return {u.sqrt_g,
        v.sqrt_g,
        u.sqrt_g_g_contra.a11,
        u.sqrt_g_g_contra.a12,
        u.sqrt_g_g_contra.a22,
        v.sqrt_g_g_contra.a11,
        v.sqrt_g_g_contra.a12,
        v.sqrt_g_g_contra.a22,
        geometry.dq1(),
        geometry.dq2()};
}

inline GeneralizedCovariantWindShearDeviceView
make_generalized_covariant_wind_shear_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {
    using Core::Geometry::HorizontalLocation;

    validate_generalized_wind_recovery_geometry(geometry);

    return {geometry.device_view(HorizontalLocation::U).sqrt_g,
        geometry.device_view(HorizontalLocation::V).sqrt_g,
        real(1.0) / geometry.dq1(),
        real(1.0) / geometry.dq2()};
}

} // namespace VVM::Dynamics::Operators

#endif
