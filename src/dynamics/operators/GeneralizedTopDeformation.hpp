#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_TOP_DEFORMATION_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_TOP_DEFORMATION_HPP

#include <cmath>
#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM::Dynamics::Operators {

struct TopDeformationTerms {
    Real stretching = real(0.0);
    Real twisting = real(0.0);
    Real planetary = real(0.0);
};

// CVVM ZETA_3D deformation at a rigid lid, in a stationary horizontal chart.
// Uses the same component-form split as the horizontal deformation.
//
// Fields supplies callable members/accessors:
//   w                     physical vertical velocity at horizontal T
//   omega1_over_rho       omega^1/rho_up at V, vertical interfaces
//   omega2_over_rho       omega^2/rho_up at U, vertical interfaces
//   omega3_over_rho       relative omega^3/rho at Z, wind levels
//   f3_at_z               vertical planetary vorticity, NOT divided by rho
//   rho, rho_up           reference density at wind/interface levels
//   inverse_spacing_mid   inverse vertical spacing at wind levels
//   inverse_spacing_up    inverse vertical spacing at interfaces
//
// Horizontal derivatives of w are covector components partial_a(w).
// Their contraction with contravariant omega^a requires no metric raising
// and remains valid for nonzero g12. Inputs must already be in one chart.
//
// Scope: z-independent horizontal mapping, unchanged physical z, uniform
// computational dq1/dq2, and the existing CVVM/VVM staggering. The caller
// supplies w(k_top)=0 across the stencil, k_top>=1, valid one-cell halos,
// and positive finite used densities and vertical spacings.
//
// Output is an unnormalized omega^3 tendency, equal to physical vertical
// vorticity tendency in this horizontal-only coordinate. Planetary gives
// stretching only; add planetary transport exactly once outside this
// operator. Horizontal planetary components are not included.
// No allocation, field mutation, launch, communication or synchronization.
struct GeneralizedTopDeformationDeviceView {
    Real dq1 = real(0.0);
    Real dq2 = real(0.0);

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION TopDeformationTerms
    calculate_at_z(const Fields& fields, int k_top, int j, int i) const noexcept {
        const int lower = k_top - 1;

        const Real left_w = fields.w(lower, j, i) + fields.w(lower, j + 1, i);
        const Real right_w = fields.w(lower, j, i + 1) + fields.w(lower, j + 1, i + 1);

        const Real relative_stretching =
            left_w *
                (fields.omega3_over_rho(k_top, j, i - 1) + fields.omega3_over_rho(k_top, j, i)) +
            right_w *
                (fields.omega3_over_rho(k_top, j, i) + fields.omega3_over_rho(k_top, j, i + 1));

        // rho(k_top) cancels the density normalization of planetary
        // vorticity at this single wind level.
        const Real planetary_stretching =
            left_w * (fields.f3_at_z(j, i - 1) + fields.f3_at_z(j, i)) +
            right_w * (fields.f3_at_z(j, i) + fields.f3_at_z(j, i + 1));

        const Real omega1_sum =
            fields.omega1_over_rho(lower, j, i) + fields.omega1_over_rho(lower, j, i + 1);

        const Real omega2_sum =
            fields.omega2_over_rho(lower, j, i) + fields.omega2_over_rho(lower, j + 1, i);

        const Real difference_q1 = fields.w(lower, j + 1, i + 1) - fields.w(lower, j + 1, i) +
                                   fields.w(lower, j, i + 1) - fields.w(lower, j, i);

        const Real difference_q2 = fields.w(lower, j + 1, i + 1) - fields.w(lower, j, i + 1) +
                                   fields.w(lower, j + 1, i) - fields.w(lower, j, i);

        const Real inverse_mid = fields.inverse_spacing_mid(k_top);
        const Real lower_factor =
            fields.rho_up(lower) * inverse_mid / fields.inverse_spacing_up(lower);

        TopDeformationTerms result;
        result.stretching = -real(0.125) * fields.rho(k_top) * inverse_mid * relative_stretching;
        result.twisting = real(0.125) * lower_factor *
                          (omega1_sum * difference_q1 / dq1 + omega2_sum * difference_q2 / dq2);
        result.planetary = -real(0.125) * inverse_mid * planetary_stretching;
        return result;
    }
};

inline GeneralizedTopDeformationDeviceView
make_generalized_top_deformation_device_view(const Core::Geometry::HorizontalGeometry& geometry) {
    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();

    if (!(dq1 > real(0.0)) || !(dq2 > real(0.0)) || !std::isfinite(dq1) || !std::isfinite(dq2)) {
        throw std::invalid_argument("GeneralizedTopDeformation requires finite positive spacing.");
    }

    return {dq1, dq2};
}

} // namespace VVM::Dynamics::Operators
#endif
