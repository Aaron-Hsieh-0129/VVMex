#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_HORIZONTAL_DEFORMATION_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_HORIZONTAL_DEFORMATION_HPP

#include <cmath>
#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM::Dynamics::Operators {

struct HorizontalDeformationTerms {
    Real stretching = real(0.0);
    Real twisting = real(0.0);
    Real planetary = real(0.0);
};

// Component-form partner of GeneralizedHorizontalVorticityTransport:
//
//   D^a = omega^b * partial_b(u^a),  a = 1, 2; b = 1, 2, 3.
//
// This is NOT a standalone covariant derivative of a vector. Connection
// terms cancel between transport and deformation in the complete
// vorticity equation. Do not add Christoffel terms to this split alone.
//
// All horizontal wind/vorticity inputs are contravariant, at their native
// staggered locations. No metric lowering/raising is performed here;
// g12 need not vanish. Any conversion from physical/covariant components
// must be performed upstream, with cross-component interpolation where
// required. Halos must already be in the receiving coordinate chart.
//
// Fields supplies callable members/accessors:
//   u1, u2                u^1 at U and u^2 at V, wind levels
//   omega1_over_rho       omega^1/rho_up at V, vertical interfaces
//   omega2_over_rho       omega^2/rho_up at U, vertical interfaces
//   omega3_over_rho       relative omega^3/rho at Z, wind levels
//   f3_at_z               vertical planetary vorticity, NOT divided by rho
//   rho, rho_up           reference densities at the two vertical locations
//   fn1, fn2              mass-weighted vertical interpolation factors
//   inverse_spacing       inverse vertical spacing at the target interface
//
// Returns actual contravariant vorticity tendencies, WITHOUT density
// normalization. omega^2 is NOT the VVM negative-sign eta_con component.
// Planetary contains only f^3 * partial_z(u^a), separately from relative
// twisting. Horizontal planetary components are outside this contract.
//
// Scope: stationary, z-independent horizontal coordinates, unchanged
// physical z, uniform computational dq1/dq2, and the existing CVVM/VVM
// staggering. The caller supplies valid k/k+1 and one horizontal halo,
// positive finite density/spacing, and applies physical boundary masks.
// No allocation, field mutation, launch, communication or synchronization.
struct GeneralizedHorizontalDeformationDeviceView {
    Real dq1 = real(0.0);
    Real dq2 = real(0.0);

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION HorizontalDeformationTerms
    calculate_omega1_at_v(const Fields& fields, int k, int j, int i) const noexcept {
        Real stretching = real(0.0);
        Real cross = real(0.0);
        Real vertical = real(0.0);
        Real planetary = real(0.0);

        for (int jj = j; jj <= j + 1; ++jj) {
            stretching +=
                (fields.omega1_over_rho(k, jj, i) + fields.omega1_over_rho(k, jj - 1, i)) *
                (fields.fn1(k) * (fields.u1(k + 1, jj, i) - fields.u1(k + 1, jj, i - 1)) +
                    fields.fn2(k) * (fields.u1(k, jj, i) - fields.u1(k, jj, i - 1)));
        }

        for (int ii = i - 1; ii <= i; ++ii) {
            cross += (fields.omega2_over_rho(k, j + 1, ii) + fields.omega2_over_rho(k, j, ii)) *
                     (fields.fn2(k) * (fields.u1(k, j + 1, ii) - fields.u1(k, j, ii)) +
                         fields.fn1(k) * (fields.u1(k + 1, j + 1, ii) - fields.u1(k + 1, j, ii)));

            const Real wind_difference = fields.u1(k + 1, j, ii) - fields.u1(k, j, ii) +
                                         fields.u1(k + 1, j + 1, ii) - fields.u1(k, j + 1, ii);

            vertical += wind_difference *
                        (fields.omega3_over_rho(k, j, ii) + fields.omega3_over_rho(k + 1, j, ii));

            planetary += wind_difference * (fields.f3_at_z(j, ii) / fields.rho(k) +
                                               fields.f3_at_z(j, ii) / fields.rho(k + 1));
        }

        const Real scale = real(0.125);
        const Real vertical_factor = fields.rho_up(k) * fields.inverse_spacing(k);

        HorizontalDeformationTerms result;
        result.stretching = scale * stretching / dq1;
        result.twisting = scale * (cross / dq2 + vertical_factor * vertical);
        result.planetary = scale * vertical_factor * planetary;
        return result;
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION HorizontalDeformationTerms
    calculate_omega2_at_u(const Fields& fields, int k, int j, int i) const noexcept {
        Real stretching = real(0.0);
        Real cross = real(0.0);
        Real vertical = real(0.0);
        Real planetary = real(0.0);

        for (int ii = i; ii <= i + 1; ++ii) {
            stretching +=
                (fields.omega2_over_rho(k, j, ii - 1) + fields.omega2_over_rho(k, j, ii)) *
                (fields.fn1(k) * (fields.u2(k + 1, j, ii) - fields.u2(k + 1, j - 1, ii)) +
                    fields.fn2(k) * (fields.u2(k, j, ii) - fields.u2(k, j - 1, ii)));
        }

        for (int jj = j - 1; jj <= j; ++jj) {
            cross += (fields.omega1_over_rho(k, jj, i + 1) + fields.omega1_over_rho(k, jj, i)) *
                     (fields.fn2(k) * (fields.u2(k, jj, i + 1) - fields.u2(k, jj, i)) +
                         fields.fn1(k) * (fields.u2(k + 1, jj, i + 1) - fields.u2(k + 1, jj, i)));

            const Real wind_difference = fields.u2(k + 1, jj, i) - fields.u2(k, jj, i) +
                                         fields.u2(k + 1, jj, i + 1) - fields.u2(k, jj, i + 1);

            vertical += wind_difference *
                        (fields.omega3_over_rho(k, jj, i) + fields.omega3_over_rho(k + 1, jj, i));

            planetary += wind_difference * (fields.f3_at_z(jj, i) / fields.rho(k) +
                                               fields.f3_at_z(jj, i) / fields.rho(k + 1));
        }

        const Real scale = real(0.125);
        const Real vertical_factor = fields.rho_up(k) * fields.inverse_spacing(k);

        HorizontalDeformationTerms result;
        result.stretching = scale * stretching / dq2;
        result.twisting = scale * (cross / dq1 + vertical_factor * vertical);
        result.planetary = scale * vertical_factor * planetary;
        return result;
    }
};

inline GeneralizedHorizontalDeformationDeviceView
make_generalized_horizontal_deformation_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {
    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();

    if (!(dq1 > real(0.0)) || !(dq2 > real(0.0)) || !std::isfinite(dq1) || !std::isfinite(dq2)) {
        throw std::invalid_argument(
            "GeneralizedHorizontalDeformation requires finite positive spacing.");
    }

    return {dq1, dq2};
}

} // namespace VVM::Dynamics::Operators
#endif
