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
// This primitive computes positive divergence. Advection tendencies use its negative when they require flux convergence.
struct HorizontalFluxDivergenceDeviceView {
    Core::Geometry::HorizontalGeometryDeviceView t;
    Core::Geometry::HorizontalGeometryDeviceView u;
    Core::Geometry::HorizontalGeometryDeviceView v;

    KOKKOS_INLINE_FUNCTION
    VVM::Real at_t(const int j, const int i,
        const VVM::Real flux_q1_plus, const VVM::Real flux_q1_minus,
        const VVM::Real flux_q2_plus, const VVM::Real flux_q2_minus) const noexcept {

        const VVM::Real q1_difference = (u.sqrt_g(j, i) * flux_q1_plus - u.sqrt_g(j, i - 1) * flux_q1_minus) / t.dq1;
        const VVM::Real q2_difference = (v.sqrt_g(j, i) * flux_q2_plus - v.sqrt_g(j - 1, i) * flux_q2_minus) / t.dq2;

        return t.inv_sqrt_g(j, i) * (q1_difference + q2_difference);
    }
};

inline HorizontalFluxDivergenceDeviceView make_horizontal_flux_divergence_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    HorizontalFluxDivergenceDeviceView result;

    result.t = geometry.device_view(Core::Geometry::HorizontalLocation::T);
    result.u = geometry.device_view(Core::Geometry::HorizontalLocation::U);
    result.v = geometry.device_view(Core::Geometry::HorizontalLocation::V);

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_FLUX_DIVERGENCE_HPP
