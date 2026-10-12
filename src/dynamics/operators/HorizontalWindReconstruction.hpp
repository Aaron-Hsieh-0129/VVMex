#ifndef VVM_DYNAMICS_OPERATORS_HORIZONTAL_WIND_RECONSTRUCTION_HPP
#define VVM_DYNAMICS_OPERATORS_HORIZONTAL_WIND_RECONSTRUCTION_HPP

#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"
#include "dynamics/operators/GeneralizedWindRecovery.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Reconstruct contravariant horizontal wind from:
//     psi at Z: streamfunction
//     chi at T: velocity potential
//
// Cartesian retains its original arithmetic. Mapped charts delegate to the
// full native-face reconstruction, including all cross-metric terms.
struct HorizontalWindReconstructionDeviceView {
    Kokkos::View<GeneralizedHorizontalWindReconstructionDeviceView> generalized;
    bool cartesian = true;
    Core::Geometry::GeometryField2D sqrt_g_at_u;
    Core::Geometry::GeometryField2D sqrt_g_at_v;
    Core::Geometry::GeometryField2D inv_sqrt_g_at_u;
    Core::Geometry::GeometryField2D inv_sqrt_g_at_v;
    Core::Geometry::GeometryField2D sqrt_g_g_contra_11_at_u;
    Core::Geometry::GeometryField2D sqrt_g_g_contra_22_at_v;

    VVM::Real dq1 = VVM::real(0.0);
    VVM::Real dq2 = VVM::real(0.0);

    VVM::Real rdq1 = VVM::real(0.0);
    VVM::Real rdq2 = VVM::real(0.0);

    template <typename PsiView, typename ChiView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    calculate_contravariant_q1_at_u(
        const PsiView& psi, const ChiView& chi, const int j, const int i) const noexcept {
        if (!cartesian) {
            return generalized().calculate_contravariant_q1_at_u(psi, chi, j, i);
        }
        const VVM::Real rotational = -(psi(j, i) - psi(j - 1, i)) / (sqrt_g_at_u(j, i) * dq2);
        const VVM::Real g_contra_11 = inv_sqrt_g_at_u(j, i) * sqrt_g_g_contra_11_at_u(j, i);
        const VVM::Real divergent = g_contra_11 * (chi(j, i + 1) - chi(j, i)) / dq1;

        return rotational + divergent;
    }

    template <typename PsiView, typename ChiView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    calculate_contravariant_q2_at_v(
        const PsiView& psi, const ChiView& chi, const int j, const int i) const noexcept {
        if (!cartesian) {
            return generalized().calculate_contravariant_q2_at_v(psi, chi, j, i);
        }

        const VVM::Real rotational = (psi(j, i) - psi(j, i - 1)) / (sqrt_g_at_v(j, i) * dq1);
        const VVM::Real g_contra_22 = inv_sqrt_g_at_v(j, i) * sqrt_g_g_contra_22_at_v(j, i);
        const VVM::Real divergent = g_contra_22 * (chi(j + 1, i) - chi(j, i)) / dq2;

        return rotational + divergent;
    }

    // For an orthogonal horizontal metric:
    //   g_11 / J = 1 / (J*g^11).
    // The covariant divergent contribution is simply dchi/dq1.
    template <typename PsiView, typename ChiView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    calculate_covariant_q1_at_u(
        const PsiView& psi, const ChiView& chi, const int j, const int i) const noexcept {
        if (!cartesian) {
            return generalized().calculate_covariant_q1_at_u(psi, chi, j, i);
        }

        const VVM::Real rotational =
            -(psi(j, i) - psi(j - 1, i)) * rdq2 / sqrt_g_g_contra_11_at_u(j, i);

        const VVM::Real divergent = (chi(j, i + 1) - chi(j, i)) * rdq1;

        return rotational + divergent;
    }

    // For an orthogonal horizontal metric:
    //   g_22 / J = 1 / (J*g^22).
    // The covariant divergent contribution is simply dchi/dq2.
    template <typename PsiView, typename ChiView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    calculate_covariant_q2_at_v(
        const PsiView& psi, const ChiView& chi, const int j, const int i) const noexcept {
        if (!cartesian) {
            return generalized().calculate_covariant_q2_at_v(psi, chi, j, i);
        }

        const VVM::Real rotational =
            (psi(j, i) - psi(j, i - 1)) * rdq1 / sqrt_g_g_contra_22_at_v(j, i);

        const VVM::Real divergent = (chi(j + 1, i) - chi(j, i)) * rdq2;

        return rotational + divergent;
    }
};

inline HorizontalWindReconstructionDeviceView
make_horizontal_wind_reconstruction_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    using Core::Geometry::GeometryKind;
    using Core::Geometry::HorizontalLocation;

    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);

    HorizontalWindReconstructionDeviceView result;
    result.cartesian = geometry.kind() == GeometryKind::Cartesian;
    if (!result.cartesian) {
        result.generalized = decltype(result.generalized)("mapped_horizontal_wind_reconstruction");
        auto host = Kokkos::create_mirror_view(result.generalized);
        host() = make_generalized_horizontal_wind_reconstruction_device_view(geometry);
        Kokkos::deep_copy(result.generalized, host);
    }

    result.sqrt_g_at_u = u.sqrt_g;
    result.sqrt_g_at_v = v.sqrt_g;
    result.inv_sqrt_g_at_u = u.inv_sqrt_g;
    result.inv_sqrt_g_at_v = v.inv_sqrt_g;
    result.sqrt_g_g_contra_11_at_u = u.sqrt_g_g_contra.a11;
    result.sqrt_g_g_contra_22_at_v = v.sqrt_g_g_contra.a22;
    result.dq1 = geometry.dq1();
    result.dq2 = geometry.dq2();

    result.rdq1 = u.rdq1;
    result.rdq2 = u.rdq2;

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_WIND_RECONSTRUCTION_HPP
