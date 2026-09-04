#ifndef VVM_DYNAMICS_OPERATORS_VERTICAL_VORTICITY_HPP
#define VVM_DYNAMICS_OPERATORS_VERTICAL_VORTICITY_HPP

#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"
#include "core/vvm_types.hpp"
#include "dynamics/operators/HorizontalCurl.hpp"
#include "dynamics/operators/HorizontalVectorLowering.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Computes vertical vorticity directly from the native contravariant
// horizontal wind components:
//
//     contravariant_q1_at_u : u^1 at U points
//     contravariant_q2_at_v : u^2 at V points
//
// The operator first lowers the vector index using the stagger-aware
// covariant metric and then applies the covariant curl at Z.
struct VerticalVorticityDeviceView {
    HorizontalVectorLoweringDeviceView vector_lowering;
    HorizontalCurlDeviceView curl;

    template<typename ContravariantQ1View, typename ContravariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_at_z(const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
        const int j, const int i) const noexcept {

        CovariantVectorStencilAtZ covariant_vector;

        covariant_vector.q1_at_u_j_i = vector_lowering.calculate_covariant_q1_at_u(contravariant_q1_at_u, contravariant_q2_at_v, j, i);
        covariant_vector.q1_at_u_jp1_i = vector_lowering.calculate_covariant_q1_at_u(contravariant_q1_at_u, contravariant_q2_at_v, j+1, i);
        covariant_vector.q2_at_v_j_i = vector_lowering.calculate_covariant_q2_at_v(contravariant_q1_at_u, contravariant_q2_at_v, j, i);
        covariant_vector.q2_at_v_j_ip1 = vector_lowering.calculate_covariant_q2_at_v(contravariant_q1_at_u, contravariant_q2_at_v, j, i+1);

        return curl.calculate_at_z(j, i, covariant_vector);
    }

    template<typename ContravariantQ1View, typename ContravariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_at_z(const ContravariantQ1View& contravariant_q1_at_u, const ContravariantQ2View& contravariant_q2_at_v,
        const int k, const int j, const int i) const noexcept {

        CovariantVectorStencilAtZ covariant_vector;

        covariant_vector.q1_at_u_j_i = vector_lowering.calculate_covariant_q1_at_u(contravariant_q1_at_u, contravariant_q2_at_v, k, j, i);
        covariant_vector.q1_at_u_jp1_i = vector_lowering.calculate_covariant_q1_at_u(contravariant_q1_at_u, contravariant_q2_at_v, k, j+1, i);
        covariant_vector.q2_at_v_j_i = vector_lowering.calculate_covariant_q2_at_v(contravariant_q1_at_u, contravariant_q2_at_v, k, j, i);
        covariant_vector.q2_at_v_j_ip1 = vector_lowering.calculate_covariant_q2_at_v(contravariant_q1_at_u, contravariant_q2_at_v, k, j, i + 1);

        return curl.calculate_at_z(j, i, covariant_vector);
    }
};

inline VerticalVorticityDeviceView make_vertical_vorticity_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    VerticalVorticityDeviceView result;

    result.vector_lowering = make_horizontal_vector_lowering_device_view(geometry);
    result.curl = make_horizontal_curl_device_view(geometry);

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_VERTICAL_VORTICITY_HPP
