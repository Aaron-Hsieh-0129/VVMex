#ifndef VVM_DYNAMICS_OPERATORS_HORIZONTAL_SCALAR_GRADIENT_HPP
#define VVM_DYNAMICS_OPERATORS_HORIZONTAL_SCALAR_GRADIENT_HPP

#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/vvm_types.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Scalar values surrounding one T point:
//
//     northwest   north   northeast
//     west        center  east
//     southwest   south   southeast
//
// The positive-face gradient components are located at:
//
//     q1_at_u : U(j, i), the positive-q1 face
//     q2_at_v : V(j, i), the positive-q2 face
struct ScalarStencilAtT {
    VVM::Real center = VVM::real(0.0);
    VVM::Real west = VVM::real(0.0);
    VVM::Real east = VVM::real(0.0);
    VVM::Real south = VVM::real(0.0);
    VVM::Real north = VVM::real(0.0);
    VVM::Real southwest = VVM::real(0.0);
    VVM::Real southeast = VVM::real(0.0);
    VVM::Real northwest = VVM::real(0.0);
    VVM::Real northeast = VVM::real(0.0);
};

// Scalar values surrounding one Z point use the same logical nine-point
// stencil as ScalarStencilAtT. The face locations are different:
//
//     q1_at_v : V(j, i + 1), the positive-q1 face
//     q2_at_u : U(j + 1, i), the positive-q2 face
struct ScalarStencilAtZ {
    VVM::Real center = VVM::real(0.0);
    VVM::Real west = VVM::real(0.0);
    VVM::Real east = VVM::real(0.0);
    VVM::Real south = VVM::real(0.0);
    VVM::Real north = VVM::real(0.0);
    VVM::Real southwest = VVM::real(0.0);
    VVM::Real southeast = VVM::real(0.0);
    VVM::Real northwest = VVM::real(0.0);
    VVM::Real northeast = VVM::real(0.0);
};

struct ContravariantGradientAroundT {
    VVM::Real q1_at_u_j_i = VVM::real(0.0);
    VVM::Real q1_at_u_j_im1 = VVM::real(0.0);
    VVM::Real q2_at_v_j_i = VVM::real(0.0);
    VVM::Real q2_at_v_jm1_i = VVM::real(0.0);
};

struct ContravariantGradientAroundZ {
    VVM::Real q1_at_v_j_ip1 = VVM::real(0.0);
    VVM::Real q1_at_v_j_i = VVM::real(0.0);
    VVM::Real q2_at_u_jp1_i = VVM::real(0.0);
    VVM::Real q2_at_u_j_i = VVM::real(0.0);
};

template<typename ScalarView>
KOKKOS_INLINE_FUNCTION
ScalarStencilAtT load_scalar_stencil_at_t(const ScalarView& scalar, const int j, const int i) noexcept {
    ScalarStencilAtT stencil;

    stencil.center = scalar(j, i);
    stencil.west = scalar(j, i - 1);
    stencil.east = scalar(j, i + 1);
    stencil.south = scalar(j - 1, i);
    stencil.north = scalar(j + 1, i);
    stencil.southwest = scalar(j - 1, i - 1);
    stencil.southeast = scalar(j - 1, i + 1);
    stencil.northwest = scalar(j + 1, i - 1);
    stencil.northeast = scalar(j + 1, i + 1);

    return stencil;
}

template<typename ScalarView>
KOKKOS_INLINE_FUNCTION
ScalarStencilAtT load_scalar_stencil_at_t(const ScalarView& scalar, const int k, const int j, const int i) noexcept {
    ScalarStencilAtT stencil;

    stencil.center = scalar(k, j, i);
    stencil.west = scalar(k, j, i - 1);
    stencil.east = scalar(k, j, i + 1);
    stencil.south = scalar(k, j - 1, i);
    stencil.north = scalar(k, j + 1, i);
    stencil.southwest = scalar(k, j - 1, i - 1);
    stencil.southeast = scalar(k, j - 1, i + 1);
    stencil.northwest = scalar(k, j + 1, i - 1);
    stencil.northeast = scalar(k, j + 1, i + 1);

    return stencil;
}

template<typename ScalarView>
KOKKOS_INLINE_FUNCTION
ScalarStencilAtZ load_scalar_stencil_at_z(const ScalarView& scalar, const int j, const int i) noexcept {
    ScalarStencilAtZ stencil;

    stencil.center = scalar(j, i);
    stencil.west = scalar(j, i - 1);
    stencil.east = scalar(j, i + 1);
    stencil.south = scalar(j - 1, i);
    stencil.north = scalar(j + 1, i);
    stencil.southwest = scalar(j - 1, i - 1);
    stencil.southeast = scalar(j - 1, i + 1);
    stencil.northwest = scalar(j + 1, i - 1);
    stencil.northeast = scalar(j + 1, i + 1);

    return stencil;
}

template<typename ScalarView>
KOKKOS_INLINE_FUNCTION
ScalarStencilAtZ load_scalar_stencil_at_z(const ScalarView& scalar, const int k, const int j, const int i) noexcept {
    ScalarStencilAtZ stencil;

    stencil.center = scalar(k, j, i);
    stencil.west = scalar(k, j, i - 1);
    stencil.east = scalar(k, j, i + 1);
    stencil.south = scalar(k, j - 1, i);
    stencil.north = scalar(k, j + 1, i);
    stencil.southwest = scalar(k, j - 1, i - 1);
    stencil.southeast = scalar(k, j - 1, i + 1);
    stencil.northwest = scalar(k, j + 1, i - 1);
    stencil.northeast = scalar(k, j + 1, i + 1);

    return stencil;
}

struct HorizontalScalarGradientDeviceView {
    Core::Geometry::HorizontalGeometryDeviceView u;
    Core::Geometry::HorizontalGeometryDeviceView v;

    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_q1_at_u(const int j, const int i, const ScalarStencilAtT& scalar) const noexcept {
        const VVM::Real dscalar_dq1_at_u = (scalar.east - scalar.center) / u.dq1;
        const VVM::Real dscalar_dq2_at_u = (scalar.north + scalar.northeast - scalar.south - scalar.southeast) / (VVM::real(4.0) * u.dq2);

        return u.g_contra_11(j, i) * dscalar_dq1_at_u + u.g_contra_12(j, i) * dscalar_dq2_at_u;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_q2_at_v(const int j, const int i, const ScalarStencilAtT& scalar) const noexcept {
        const VVM::Real dscalar_dq1_at_v = (scalar.east + scalar.northeast - scalar.west - scalar.northwest) / (VVM::real(4.0) * v.dq1);
        const VVM::Real dscalar_dq2_at_v = (scalar.north - scalar.center) / v.dq2;

        return v.g_contra_12(j, i) * dscalar_dq1_at_v + v.g_contra_22(j, i) * dscalar_dq2_at_v;
    }

    KOKKOS_INLINE_FUNCTION
    ContravariantGradientAroundT calculate_uv_around_t(const int j, const int i, const ScalarStencilAtT& scalar) const noexcept {

        const VVM::Real dscalar_dq1_at_u_j_im1 = (scalar.center - scalar.west) / u.dq1;
        const VVM::Real dscalar_dq2_at_u_j_im1 = (scalar.north + scalar.northwest - scalar.south - scalar.southwest) / (VVM::real(4.0) * u.dq2);

        const VVM::Real dscalar_dq1_at_v_jm1_i = (scalar.east + scalar.southeast - scalar.west - scalar.southwest) / (VVM::real(4.0) * v.dq1);
        const VVM::Real dscalar_dq2_at_v_jm1_i = (scalar.center - scalar.south) / v.dq2;

        ContravariantGradientAroundT result;

        result.q1_at_u_j_i = calculate_q1_at_u(j, i, scalar);
        result.q2_at_v_j_i = calculate_q2_at_v(j, i, scalar);

        result.q1_at_u_j_im1 = u.g_contra_11(j, i - 1) * dscalar_dq1_at_u_j_im1 + u.g_contra_12(j, i - 1) * dscalar_dq2_at_u_j_im1;
        result.q2_at_v_jm1_i = v.g_contra_12(j - 1, i) * dscalar_dq1_at_v_jm1_i + v.g_contra_22(j - 1, i) * dscalar_dq2_at_v_jm1_i;

        return result;
    }

    KOKKOS_INLINE_FUNCTION
    ContravariantGradientAroundZ calculate_vu_around_z(const int j, const int i, const ScalarStencilAtZ& scalar) const noexcept {
        const VVM::Real dscalar_dq1_at_v_j_ip1 = (scalar.east - scalar.center) / v.dq1;
        const VVM::Real dscalar_dq2_at_v_j_ip1 = (scalar.north + scalar.northeast - scalar.south - scalar.southeast) / (VVM::real(4.0) * v.dq2);

        const VVM::Real dscalar_dq1_at_v_j_i = (scalar.center - scalar.west) / v.dq1;
        const VVM::Real dscalar_dq2_at_v_j_i = (scalar.north + scalar.northwest - scalar.south - scalar.southwest) / (VVM::real(4.0) * v.dq2);

        const VVM::Real dscalar_dq1_at_u_jp1_i = (scalar.east + scalar.northeast - scalar.west - scalar.northwest) / (VVM::real(4.0) * u.dq1);
        const VVM::Real dscalar_dq2_at_u_jp1_i = (scalar.north - scalar.center) / u.dq2;

        const VVM::Real dscalar_dq1_at_u_j_i = (scalar.east + scalar.southeast - scalar.west - scalar.southwest) / (VVM::real(4.0) * u.dq1);
        const VVM::Real dscalar_dq2_at_u_j_i = (scalar.center - scalar.south) / u.dq2;

        ContravariantGradientAroundZ result;

        result.q1_at_v_j_ip1 = v.g_contra_11(j, i + 1) * dscalar_dq1_at_v_j_ip1 + v.g_contra_12(j, i + 1) * dscalar_dq2_at_v_j_ip1;
        result.q1_at_v_j_i = v.g_contra_11(j, i) * dscalar_dq1_at_v_j_i + v.g_contra_12(j, i) * dscalar_dq2_at_v_j_i;
        result.q2_at_u_jp1_i = u.g_contra_12(j + 1, i) * dscalar_dq1_at_u_jp1_i + u.g_contra_22(j + 1, i) * dscalar_dq2_at_u_jp1_i;
        result.q2_at_u_j_i = u.g_contra_12(j, i) * dscalar_dq1_at_u_j_i + u.g_contra_22(j, i) * dscalar_dq2_at_u_j_i;

        return result;
    }
};

inline HorizontalScalarGradientDeviceView make_horizontal_scalar_gradient_device_view(const Core::Geometry::HorizontalGeometry& geometry) {
    HorizontalScalarGradientDeviceView result;

    result.u = geometry.device_view(Core::Geometry::HorizontalLocation::U);
    result.v = geometry.device_view(Core::Geometry::HorizontalLocation::V);

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM


#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_SCALAR_GRADIENT_HPP
