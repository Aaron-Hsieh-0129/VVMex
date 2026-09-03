#ifndef VVM_DYNAMICS_OPERATORS_HORIZONTAL_CURL_HPP
#define VVM_DYNAMICS_OPERATORS_HORIZONTAL_CURL_HPP

#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/vvm_types.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Covariant horizontal vector components surrounding one Z point:
//    Z is at (i+1/2, j+1/2)
//
//     q1_at_u_jp1_i
//             |
//             Z
//             |
//      q1_at_u_j_i
//
//     q2_at_v_j_i -- Z -- q2_at_v_j_ip1
//
// q1_at_u contains the lower-index component u_1 at U points.
// q2_at_v contains the lower-index component u_2 at V points.
//
// These are covariant computational components, not physical east/north
// components and not contravariant components.
struct CovariantVectorStencilAtZ {
    VVM::Real q1_at_u_j_i = VVM::real(0.0);
    VVM::Real q1_at_u_jp1_i = VVM::real(0.0);

    VVM::Real q2_at_v_j_i = VVM::real(0.0);
    VVM::Real q2_at_v_j_ip1 = VVM::real(0.0);
};

template<typename CovariantQ1View, typename CovariantQ2View>
KOKKOS_INLINE_FUNCTION
CovariantVectorStencilAtZ load_covariant_vector_stencil_at_z(
    const CovariantQ1View& covariant_q1_at_u, const CovariantQ2View& covariant_q2_at_v,
    const int j, const int i) noexcept {
    CovariantVectorStencilAtZ stencil;

    stencil.q1_at_u_j_i = covariant_q1_at_u(j, i);
    stencil.q1_at_u_jp1_i = covariant_q1_at_u(j + 1, i);

    stencil.q2_at_v_j_i = covariant_q2_at_v(j, i);
    stencil.q2_at_v_j_ip1 = covariant_q2_at_v(j, i + 1);

    return stencil;
}

template<typename CovariantQ1View, typename CovariantQ2View>
KOKKOS_INLINE_FUNCTION
CovariantVectorStencilAtZ load_covariant_vector_stencil_at_z(
    const CovariantQ1View& covariant_q1_at_u, const CovariantQ2View& covariant_q2_at_v,
    const int k, const int j, const int i) noexcept {
    CovariantVectorStencilAtZ stencil;

    stencil.q1_at_u_j_i = covariant_q1_at_u(k, j, i);
    stencil.q1_at_u_jp1_i = covariant_q1_at_u(k, j + 1, i);

    stencil.q2_at_v_j_i = covariant_q2_at_v(k, j, i);
    stencil.q2_at_v_j_ip1 = covariant_q2_at_v(k, j, i + 1);

    return stencil;
}

struct HorizontalCurlDeviceView {
    Core::Geometry::HorizontalGeometryDeviceView z;

    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_at_z(const int j, const int i, const CovariantVectorStencilAtZ& vector) const noexcept {
        const VVM::Real dq2_component_dq1 = (vector.q2_at_v_j_ip1 - vector.q2_at_v_j_i) / z.dq1;
        const VVM::Real dq1_component_dq2 = (vector.q1_at_u_jp1_i - vector.q1_at_u_j_i) / z.dq2;

        return z.inv_sqrt_g(j, i) * (dq2_component_dq1 - dq1_component_dq2);
    }

    template<typename CovariantQ1View, typename CovariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_at_z(
        const CovariantQ1View& covariant_q1_at_u, const CovariantQ2View& covariant_q2_at_v,
        const int j, const int i) const noexcept {

        return calculate_at_z(j, i, load_covariant_vector_stencil_at_z(covariant_q1_at_u, covariant_q2_at_v, j, i));
    }

    template<typename CovariantQ1View, typename CovariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_at_z(const CovariantQ1View& covariant_q1_at_u, const CovariantQ2View& covariant_q2_at_v,
        const int k, const int j, const int i) const noexcept {

        return calculate_at_z(j, i, load_covariant_vector_stencil_at_z(covariant_q1_at_u, covariant_q2_at_v, k, j, i));
    }
};

inline HorizontalCurlDeviceView make_horizontal_curl_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    HorizontalCurlDeviceView result;

    result.z = geometry.device_view(Core::Geometry::HorizontalLocation::Z);

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_CURL_HPP
