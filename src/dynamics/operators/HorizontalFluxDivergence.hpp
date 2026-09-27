#ifndef VVM_DYNAMICS_OPERATORS_HORIZONTAL_FLUX_DIVERGENCE_HPP
#define VVM_DYNAMICS_OPERATORS_HORIZONTAL_FLUX_DIVERGENCE_HPP

#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/vvm_types.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// The supplied fluxes are contravariant flux components at their native faces:
//
//     flux_q1_plus   : F^1 at U(j, i)
//     flux_q1_minus  : F^1 at U(j, i - 1)
//     flux_q2_plus   : F^2 at V(j, i)
//     flux_q2_minus  : F^2 at V(j - 1, i)
//
// This primitive computes positive divergence. Advection tendencies use its
// negative when they require flux convergence.
struct HorizontalFluxDivergenceDeviceView {
    Core::Geometry::HorizontalGeometryDeviceView t;
    Core::Geometry::HorizontalGeometryDeviceView u;
    Core::Geometry::HorizontalGeometryDeviceView v;
    Core::Geometry::HorizontalGeometryDeviceView z;

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    jacobian_weighted_at_t(const int j,
        const int i,
        const VVM::Real flux_q1_plus,
        const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus,
        const VVM::Real flux_q2_minus) const noexcept {

        const VVM::Real q1_difference =
            (u.sqrt_g(j, i) * flux_q1_plus - u.sqrt_g(j, i - 1) * flux_q1_minus) / t.dq1;

        const VVM::Real q2_difference =
            (v.sqrt_g(j, i) * flux_q2_plus - v.sqrt_g(j - 1, i) * flux_q2_minus) / t.dq2;

        return q1_difference + q2_difference;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    at_t(const int j,
        const int i,
        const VVM::Real flux_q1_plus,
        const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus,
        const VVM::Real flux_q2_minus) const noexcept {

        const VVM::Real q1_difference =
            (u.sqrt_g(j, i) * flux_q1_plus - u.sqrt_g(j, i - 1) * flux_q1_minus) / t.dq1;

        const VVM::Real q2_difference =
            (v.sqrt_g(j, i) * flux_q2_plus - v.sqrt_g(j - 1, i) * flux_q2_minus) / t.dq2;

        return t.inv_sqrt_g(j, i) * (q1_difference + q2_difference);
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    jacobian_weighted_at_z(const int j,
        const int i,
        const VVM::Real flux_q1_plus,
        const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus,
        const VVM::Real flux_q2_minus) const noexcept {

        const VVM::Real q1_difference =
            (v.sqrt_g(j, i + 1) * flux_q1_plus - v.sqrt_g(j, i) * flux_q1_minus) / z.dq1;

        const VVM::Real q2_difference =
            (u.sqrt_g(j + 1, i) * flux_q2_plus - u.sqrt_g(j, i) * flux_q2_minus) / z.dq2;

        return q1_difference + q2_difference;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    at_z(const int j,
        const int i,
        const VVM::Real flux_q1_plus,
        const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus,
        const VVM::Real flux_q2_minus) const noexcept {

        return z.inv_sqrt_g(j, i) * jacobian_weighted_at_z(j,
                                        i,
                                        flux_q1_plus,
                                        flux_q1_minus,
                                        flux_q2_plus,
                                        flux_q2_minus);
    }

    // U target:
    //
    //   q1 fluxes: T(j, i + 1) / T(j, i)
    //   q2 fluxes: Z(j, i)     / Z(j - 1, i)
    //
    // This allows a scalar living natively at U to use the same
    // conservative generalized divergence operator.
    KOKKOS_INLINE_FUNCTION
    VVM::Real
    jacobian_weighted_at_u(const int j,
        const int i,
        const VVM::Real flux_q1_plus,
        const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus,
        const VVM::Real flux_q2_minus) const noexcept {

        const VVM::Real q1_difference =
            (t.sqrt_g(j, i + 1) * flux_q1_plus - t.sqrt_g(j, i) * flux_q1_minus) / u.dq1;

        const VVM::Real q2_difference =
            (z.sqrt_g(j, i) * flux_q2_plus - z.sqrt_g(j - 1, i) * flux_q2_minus) / u.dq2;

        return q1_difference + q2_difference;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    at_u(const int j,
        const int i,
        const VVM::Real flux_q1_plus,
        const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus,
        const VVM::Real flux_q2_minus) const noexcept {

        return u.inv_sqrt_g(j, i) * jacobian_weighted_at_u(j,
                                        i,
                                        flux_q1_plus,
                                        flux_q1_minus,
                                        flux_q2_plus,
                                        flux_q2_minus);
    }

    // V target:
    //
    //   q1 fluxes: Z(j, i)     / Z(j, i - 1)
    //   q2 fluxes: T(j + 1, i) / T(j, i)
    //
    // This allows a scalar living natively at V to use the same
    // conservative generalized divergence operator.
    KOKKOS_INLINE_FUNCTION
    VVM::Real
    jacobian_weighted_at_v(const int j,
        const int i,
        const VVM::Real flux_q1_plus,
        const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus,
        const VVM::Real flux_q2_minus) const noexcept {

        const VVM::Real q1_difference =
            (z.sqrt_g(j, i) * flux_q1_plus - z.sqrt_g(j, i - 1) * flux_q1_minus) / v.dq1;

        const VVM::Real q2_difference =
            (t.sqrt_g(j + 1, i) * flux_q2_plus - t.sqrt_g(j, i) * flux_q2_minus) / v.dq2;

        return q1_difference + q2_difference;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    at_v(const int j,
        const int i,
        const VVM::Real flux_q1_plus,
        const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus,
        const VVM::Real flux_q2_minus) const noexcept {

        return v.inv_sqrt_g(j, i) * jacobian_weighted_at_v(j,
                                        i,
                                        flux_q1_plus,
                                        flux_q1_minus,
                                        flux_q2_plus,
                                        flux_q2_minus);
    }
};

inline HorizontalFluxDivergenceDeviceView
make_horizontal_flux_divergence_device_view(const Core::Geometry::HorizontalGeometry& geometry) {

    HorizontalFluxDivergenceDeviceView result;

    result.t = geometry.device_view(Core::Geometry::HorizontalLocation::T);
    result.u = geometry.device_view(Core::Geometry::HorizontalLocation::U);
    result.v = geometry.device_view(Core::Geometry::HorizontalLocation::V);
    result.z = geometry.device_view(Core::Geometry::HorizontalLocation::Z);

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_FLUX_DIVERGENCE_HPP
