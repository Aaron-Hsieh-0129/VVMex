#ifndef VVM_DYNAMICS_OPERATORS_HORIZONTAL_VORTICITY_HPP
#define VVM_DYNAMICS_OPERATORS_HORIZONTAL_VORTICITY_HPP

#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Horizontal components of relative vorticity for a height-independent
// horizontal metric and physical vertical coordinate z.
//
// Inputs:
//   w                   : physical vertical velocity, horizontally at T
//   covariant_q1_at_u    : covariant horizontal wind u_1
//   covariant_q2_at_v    : covariant horizontal wind u_2
//
// Outputs:
//   contravariant_q1_at_v : omega^1, horizontally at V
//   contravariant_q2_at_u : omega^2, horizontally at U
//
// Both vorticity components are vertically between wind levels k and k+1.
// inverse_vertical_spacing is the reciprocal physical distance between
// those levels. The caller must provide a positive, finite value.
//
// In Cartesian VVMex:
//   omega^1 = xi
//   omega^2 = -eta
//
// These methods do not change the convention of existing State fields.
struct HorizontalVorticityDeviceView {
    Core::Geometry::GeometryField2D sqrt_g_at_u;
    Core::Geometry::GeometryField2D sqrt_g_at_v;

    VVM::Real dq1 = VVM::real(0.0);
    VVM::Real dq2 = VVM::real(0.0);

    template<typename WView, typename CovariantQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_contravariant_q1_at_v(const WView& w, const CovariantQ2View& covariant_q2_at_v,
        const int k, const int j, const int i, const VVM::Real inverse_vertical_spacing) const noexcept {

        const VVM::Real dw_dq2 = (w(k, j + 1, i) - w(k, j, i)) / dq2;
        const VVM::Real du2_dz = (covariant_q2_at_v(k + 1, j, i) - covariant_q2_at_v(k, j, i)) * inverse_vertical_spacing;

        return (dw_dq2 - du2_dz) / sqrt_g_at_v(j, i);
    }

    template<typename WView, typename CovariantQ1View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_contravariant_q2_at_u(const WView& w, const CovariantQ1View& covariant_q1_at_u,
        const int k, const int j, const int i, const VVM::Real inverse_vertical_spacing) const noexcept {

        const VVM::Real du1_dz = (covariant_q1_at_u(k + 1, j, i) - covariant_q1_at_u(k, j, i)) * inverse_vertical_spacing;
        const VVM::Real dw_dq1 = (w(k, j, i + 1) - w(k, j, i)) / dq1;

        return (du1_dz - dw_dq1) / sqrt_g_at_u(j, i);
    }

    // CVVM WIND_3D covariant-wind integration uses:
    //   u_1(k) = u_1(k+1) - du_1/dz(k) * physical_dz(k).
    template<typename WView, typename VorticityQ2View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_covariant_q1_vertical_shear_at_u(const WView& w, const VorticityQ2View& contravariant_q2_at_u,
        const int k, const int j, const int i) const noexcept {

        const VVM::Real dw_dq1 = (w(k, j, i + 1) - w(k, j, i)) / dq1;

        return dw_dq1 + sqrt_g_at_u(j, i) * contravariant_q2_at_u(k, j, i);
    }

    template<typename WView, typename VorticityQ1View>
    KOKKOS_INLINE_FUNCTION
    VVM::Real calculate_covariant_q2_vertical_shear_at_v(const WView& w, const VorticityQ1View& contravariant_q1_at_v,
        const int k, const int j, const int i) const noexcept {

        const VVM::Real dw_dq2 = (w(k, j + 1, i) - w(k, j, i)) / dq2;

        return dw_dq2 - sqrt_g_at_v(j, i) * contravariant_q1_at_v(k, j, i);
    }
};

inline HorizontalVorticityDeviceView make_horizontal_vorticity_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    using Core::Geometry::GeometryKind;
    using Core::Geometry::HorizontalLocation;

    if (geometry.kind() != GeometryKind::Cartesian && geometry.kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument("HorizontalVorticity currently supports Cartesian and regular latitude-longitude geometry.");
    }

    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);

    HorizontalVorticityDeviceView result;
    result.sqrt_g_at_u = u.sqrt_g;
    result.sqrt_g_at_v = v.sqrt_g;
    result.dq1 = geometry.dq1();
    result.dq2 = geometry.dq2();

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_VORTICITY_HPP
