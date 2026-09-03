#ifndef VVM_DYNAMICS_OPERATORS_HORIZONTAL_LAPLACE_BELTRAMI_HPP
#define VVM_DYNAMICS_OPERATORS_HORIZONTAL_LAPLACE_BELTRAMI_HPP

#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"
#include "core/vvm_types.hpp"
#include "dynamics/operators/HorizontalFluxDivergence.hpp"
#include "dynamics/operators/HorizontalScalarGradient.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

struct HorizontalLaplaceBeltramiDeviceView {
    HorizontalScalarGradientDeviceView gradient;
    HorizontalFluxDivergenceDeviceView divergence;

    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_at_t(const int j, const int i, const ScalarStencilAtT& scalar) const noexcept {
        const ContravariantGradientAroundT gradient_around_t = gradient.calculate_uv_around_t(j, i, scalar);

        return divergence.at_t(
            j,
            i,
            gradient_around_t.q1_at_u_j_i,
            gradient_around_t.q1_at_u_j_im1,
            gradient_around_t.q2_at_v_j_i,
            gradient_around_t.q2_at_v_jm1_i);
    }

    template<typename ScalarView>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_at_t(const ScalarView& scalar, const int j, const int i) const noexcept {
        return calculate_at_t(j, i, load_scalar_stencil_at_t(scalar, j, i));
    }

    template<typename ScalarView>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_at_t(const ScalarView& scalar, const int k, const int j, const int i) const noexcept {
        return calculate_at_t(j, i, load_scalar_stencil_at_t(scalar, k, j, i));
    }
};

inline HorizontalLaplaceBeltramiDeviceView make_horizontal_laplace_beltrami_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    HorizontalLaplaceBeltramiDeviceView result;

    result.gradient = make_horizontal_scalar_gradient_device_view(geometry);
    result.divergence = make_horizontal_flux_divergence_device_view(geometry);

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_LAPLACE_BELTRAMI_HPP
