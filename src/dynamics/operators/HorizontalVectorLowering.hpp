#ifndef VVM_DYNAMICS_OPERATORS_HORIZONTAL_VECTOR_LOWERING_HPP
#define VVM_DYNAMICS_OPERATORS_HORIZONTAL_VECTOR_LOWERING_HPP

#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/vvm_types.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Contravariant components needed to calculate u_1 at U(j,i):
//
//     q2_at_v_j_i          q2_at_v_j_ip1
//
//                   q1_at_u_j_i
//
//     q2_at_v_jm1_i        q2_at_v_jm1_ip1
//
struct ContravariantVectorStencilAtU {
    VVM::Real q1_at_u_j_i = VVM::real(0.0);

    VVM::Real q2_at_v_j_i = VVM::real(0.0);
    VVM::Real q2_at_v_j_ip1 = VVM::real(0.0);
    VVM::Real q2_at_v_jm1_i = VVM::real(0.0);
    VVM::Real q2_at_v_jm1_ip1 = VVM::real(0.0);
};

// Contravariant components needed to calculate u_2 at V(j,i):
//
//     q1_at_u_jp1_im1      q1_at_u_jp1_i
//
//                   q2_at_v_j_i
//
//     q1_at_u_j_im1        q1_at_u_j_i
//
struct ContravariantVectorStencilAtV {
    VVM::Real q2_at_v_j_i = VVM::real(0.0);

    VVM::Real q1_at_u_j_i = VVM::real(0.0);
    VVM::Real q1_at_u_jp1_i = VVM::real(0.0);
    VVM::Real q1_at_u_j_im1 = VVM::real(0.0);
    VVM::Real q1_at_u_jp1_im1 = VVM::real(0.0);
};

// The off-diagonal metric contribution is evaluated at its native
// staggering before interpolation:
//
//     (g_12 u^2)_U ~= 1/4 sum_V (g_12 u^2)_V
//     (g_12 u^1)_V ~= 1/4 sum_U (g_12 u^1)_U
//
// Do not interpolate the vector component first and then multiply by
// g_12 at the target point. This ordering follows the CVVM discretization
// and matters when g_12 varies spatially.

template<typename ContravariantQ1View, typename ContravariantQ2View>
KOKKOS_INLINE_FUNCTION
ContravariantVectorStencilAtU load_contravariant_vector_stencil_at_u(
    const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
    const int j, const int i) noexcept {

    ContravariantVectorStencilAtU stencil;

    stencil.q1_at_u_j_i = contravariant_q1_at_u(j, i);

    stencil.q2_at_v_j_i = contravariant_q2_at_v(j, i);
    stencil.q2_at_v_j_ip1 = contravariant_q2_at_v(j, i+1);
    stencil.q2_at_v_jm1_i = contravariant_q2_at_v(j-1, i);
    stencil.q2_at_v_jm1_ip1 = contravariant_q2_at_v(j-1, i+1);

    return stencil;
}

template<typename ContravariantQ1View, typename ContravariantQ2View>
KOKKOS_INLINE_FUNCTION
ContravariantVectorStencilAtV load_contravariant_vector_stencil_at_v(
    const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
    const int j, const int i) noexcept {

    ContravariantVectorStencilAtV stencil;

    stencil.q2_at_v_j_i = contravariant_q2_at_v(j, i);

    stencil.q1_at_u_j_i = contravariant_q1_at_u(j, i);
    stencil.q1_at_u_jp1_i = contravariant_q1_at_u(j+1, i);
    stencil.q1_at_u_j_im1 = contravariant_q1_at_u(j, i-1);
    stencil.q1_at_u_jp1_im1 = contravariant_q1_at_u(j+1, i-1);

    return stencil;
}

template<typename ContravariantQ1View, typename ContravariantQ2View>
KOKKOS_INLINE_FUNCTION
ContravariantVectorStencilAtU load_contravariant_vector_stencil_at_u(
    const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
    const int k, const int j, const int i) noexcept {

    ContravariantVectorStencilAtU stencil;

    stencil.q1_at_u_j_i = contravariant_q1_at_u(k, j, i);

    stencil.q2_at_v_j_i = contravariant_q2_at_v(k, j, i);
    stencil.q2_at_v_j_ip1 = contravariant_q2_at_v(k, j, i+1);
    stencil.q2_at_v_jm1_i = contravariant_q2_at_v(k, j-1, i);
    stencil.q2_at_v_jm1_ip1 = contravariant_q2_at_v(k, j-1, i+1);

    return stencil;
}

template<typename ContravariantQ1View, typename ContravariantQ2View>
KOKKOS_INLINE_FUNCTION
ContravariantVectorStencilAtV load_contravariant_vector_stencil_at_v(
    const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
    const int k, const int j, const int i) noexcept {

    ContravariantVectorStencilAtV stencil;

    stencil.q2_at_v_j_i = contravariant_q2_at_v(k, j, i);

    stencil.q1_at_u_j_i = contravariant_q1_at_u(k, j, i);
    stencil.q1_at_u_jp1_i = contravariant_q1_at_u(k, j+1, i);
    stencil.q1_at_u_j_im1 = contravariant_q1_at_u(k, j, i-1);
    stencil.q1_at_u_jp1_im1 = contravariant_q1_at_u(k, j+1, i-1);

    return stencil;
}

struct HorizontalVectorLoweringDeviceView {
    Core::Geometry::HorizontalGeometryDeviceView u;
    Core::Geometry::HorizontalGeometryDeviceView v;

    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_covariant_q1_at_u(const int j, const int i,
        const ContravariantVectorStencilAtU& vector) const noexcept {

        const VVM::Real direct_component = u.g_cov.a11(j, i) * vector.q1_at_u_j_i;

        const VVM::Real cross_component = VVM::real(0.25) * (
            v.g_cov.a12(j, i) * vector.q2_at_v_j_i + v.g_cov.a12(j, i+1) * vector.q2_at_v_j_ip1 +
            v.g_cov.a12(j-1, i) * vector.q2_at_v_jm1_i + v.g_cov.a12(j-1, i+1) * vector.q2_at_v_jm1_ip1);

        return direct_component + cross_component;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_covariant_q2_at_v(
        const int j,
        const int i,
        const ContravariantVectorStencilAtV& vector) const noexcept {

        const VVM::Real direct_component = v.g_cov.a22(j, i) * vector.q2_at_v_j_i;

        const VVM::Real cross_component = VVM::real(0.25) * (
            u.g_cov.a12(j, i) * vector.q1_at_u_j_i + u.g_cov.a12(j+1, i) * vector.q1_at_u_jp1_i +
            u.g_cov.a12(j, i-1) * vector.q1_at_u_j_im1 + u.g_cov.a12(j+1, i-1) * vector.q1_at_u_jp1_im1);

        return direct_component + cross_component;
    }

    template<typename ContravariantQ1View, typename ContravariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_covariant_q1_at_u(
        const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
        const int j, const int i) const noexcept {

        return calculate_covariant_q1_at_u(j, i, load_contravariant_vector_stencil_at_u(contravariant_q1_at_u, contravariant_q2_at_v, j, i));
    }

    template<typename ContravariantQ1View, typename ContravariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_covariant_q2_at_v(
        const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
        const int j, const int i) const noexcept {

        return calculate_covariant_q2_at_v(j, i, load_contravariant_vector_stencil_at_v(contravariant_q1_at_u, contravariant_q2_at_v, j, i));
    }

    template<typename ContravariantQ1View, typename ContravariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_covariant_q1_at_u(
        const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
        const int k, const int j, const int i) const noexcept {

        return calculate_covariant_q1_at_u(j, i, load_contravariant_vector_stencil_at_u(contravariant_q1_at_u, contravariant_q2_at_v, k, j, i));
    }

    template<typename ContravariantQ1View, typename ContravariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_covariant_q2_at_v(
        const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
        const int k, const int j, const int i) const noexcept {

        return calculate_covariant_q2_at_v(j, i, load_contravariant_vector_stencil_at_v(contravariant_q1_at_u, contravariant_q2_at_v, k, j, i));
    }
};

inline HorizontalVectorLoweringDeviceView make_horizontal_vector_lowering_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    HorizontalVectorLoweringDeviceView result;

    result.u = geometry.device_view(Core::Geometry::HorizontalLocation::U);
    result.v = geometry.device_view(Core::Geometry::HorizontalLocation::V);

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_VECTOR_LOWERING_HPP
