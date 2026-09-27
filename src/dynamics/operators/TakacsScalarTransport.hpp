#ifndef VVM_DYNAMICS_OPERATORS_TAKACS_SCALAR_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_TAKACS_SCALAR_TRANSPORT_HPP

#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"
#include "core/vvm_types.hpp"
#include "dynamics/operators/HorizontalFluxDivergence.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// CVVM Takacs face reconstruction.
//
// mass_flux is the unweighted contravariant mass flux at the face:
//
//     horizontal: rho * u^1 or rho * u^2
//     vertical:   rho_up * w
//
// Horizontal Jacobian factors are deliberately not included here.
// HorizontalFluxDivergence applies the native U/V face Jacobians exactly once.
struct TakacsFaceFluxDeviceView {
    VVM::Real alpha = VVM::real(1.0);

    KOKKOS_INLINE_FUNCTION
    static VVM::Real
    positive_part(const VVM::Real value) noexcept {
        return VVM::real(0.5) * (value + Kokkos::abs(value));
    }

    KOKKOS_INLINE_FUNCTION
    static VVM::Real
    negative_part(const VVM::Real value) noexcept {
        return VVM::real(0.5) * (value - Kokkos::abs(value));
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    calculate(const VVM::Real mass_flux_previous,
        const VVM::Real mass_flux,
        const VVM::Real mass_flux_next,
        const VVM::Real scalar_previous,
        const VVM::Real scalar_left,
        const VVM::Real scalar_right,
        const VVM::Real scalar_next) const noexcept {

        const VVM::Real positive_previous = positive_part(mass_flux_previous);
        const VVM::Real positive = positive_part(mass_flux);
        const VVM::Real negative = negative_part(mass_flux);
        const VVM::Real negative_next = negative_part(mass_flux_next);

        const VVM::Real centered_flux = VVM::real(0.5) * mass_flux * (scalar_right + scalar_left);

        const VVM::Real correction = positive * (scalar_right - scalar_left) -
                                     Kokkos::sqrt(positive) * Kokkos::sqrt(positive_previous) *
                                         (scalar_left - scalar_previous) +
                                     negative * (scalar_left - scalar_right) +
                                     Kokkos::sqrt(Kokkos::abs(negative)) *
                                         Kokkos::sqrt(Kokkos::abs(negative_next)) *
                                         (scalar_right - scalar_next);

        return centered_flux - alpha * correction / VVM::real(6.0);
    }

    // First interior vertical face. The exterior positive-flow stencil is not
    // available, matching CVVM ADVEC_3D at KLOW.
    KOKKOS_INLINE_FUNCTION
    VVM::Real
    calculate_at_lower_boundary(const VVM::Real mass_flux,
        const VVM::Real mass_flux_next,
        const VVM::Real scalar_left,
        const VVM::Real scalar_right,
        const VVM::Real scalar_next) const noexcept {

        const VVM::Real centered_flux = VVM::real(0.5) * mass_flux * (scalar_right + scalar_left);

        if (mass_flux >= VVM::real(0.0)) {
            return centered_flux;
        }

        const VVM::Real negative = negative_part(mass_flux);
        const VVM::Real negative_next = negative_part(mass_flux_next);

        const VVM::Real correction =
            negative * (scalar_left - scalar_right) + Kokkos::sqrt(Kokkos::abs(negative)) *
                                                          Kokkos::sqrt(Kokkos::abs(negative_next)) *
                                                          (scalar_right - scalar_next);

        return centered_flux - alpha * correction / VVM::real(6.0);
    }

    // Last interior vertical face. The exterior negative-flow stencil is not
    // available, matching CVVM ADVEC_3D at NK1.
    KOKKOS_INLINE_FUNCTION
    VVM::Real
    calculate_at_upper_boundary(const VVM::Real mass_flux_previous,
        const VVM::Real mass_flux,
        const VVM::Real scalar_previous,
        const VVM::Real scalar_left,
        const VVM::Real scalar_right) const noexcept {

        const VVM::Real centered_flux = VVM::real(0.5) * mass_flux * (scalar_right + scalar_left);

        if (mass_flux < VVM::real(0.0)) {
            return centered_flux;
        }

        const VVM::Real positive_previous = positive_part(mass_flux_previous);
        const VVM::Real positive = positive_part(mass_flux);

        const VVM::Real correction = positive * (scalar_right - scalar_left) -
                                     Kokkos::sqrt(positive) * Kokkos::sqrt(positive_previous) *
                                         (scalar_left - scalar_previous);

        return centered_flux - alpha * correction / VVM::real(6.0);
    }
};

// Dry scalar transport at T points.
//
// Required input representation:
//
//     scalar_q
//         Density-normalized prognostic scalar at T points.
//
//     mass_flux_q1
//         rho * u^1 at U points, without the horizontal Jacobian.
//
//     mass_flux_q2
//         rho * u^2 at V points, without the horizontal Jacobian.
//
//     mass_flux_vertical
//         rho_up * w at vertical faces.
//
// The horizontal operation is
//
//     -(1 / J) [d(J rho u^1 q)/dq1 + d(J rho u^2 q)/dq2].
//
// The vertical operation is
//
//     -d(rho_up w q)/dz.
//
// calculate_tendency_at_t divides their sum by the T-level density.
struct TakacsScalarTransportDeviceView {
    HorizontalFluxDivergenceDeviceView horizontal_divergence;
    TakacsFaceFluxDeviceView face_flux;

    template <typename ScalarView, typename MassFluxQ1View, typename MassFluxQ2View>
    KOKKOS_INLINE_FUNCTION VVM::Real
    calculate_horizontal_flux_convergence_at_t(const ScalarView& scalar_q,
        const MassFluxQ1View& mass_flux_q1,
        const MassFluxQ2View& mass_flux_q2,
        const int k,
        const int j,
        const int i) const noexcept {

        const VVM::Real flux_q1_plus = face_flux.calculate(mass_flux_q1(k, j, i - 1),
            mass_flux_q1(k, j, i),
            mass_flux_q1(k, j, i + 1),
            scalar_q(k, j, i - 1),
            scalar_q(k, j, i),
            scalar_q(k, j, i + 1),
            scalar_q(k, j, i + 2));

        const VVM::Real flux_q1_minus = face_flux.calculate(mass_flux_q1(k, j, i - 2),
            mass_flux_q1(k, j, i - 1),
            mass_flux_q1(k, j, i),
            scalar_q(k, j, i - 2),
            scalar_q(k, j, i - 1),
            scalar_q(k, j, i),
            scalar_q(k, j, i + 1));

        const VVM::Real flux_q2_plus = face_flux.calculate(mass_flux_q2(k, j - 1, i),
            mass_flux_q2(k, j, i),
            mass_flux_q2(k, j + 1, i),
            scalar_q(k, j - 1, i),
            scalar_q(k, j, i),
            scalar_q(k, j + 1, i),
            scalar_q(k, j + 2, i));

        const VVM::Real flux_q2_minus = face_flux.calculate(mass_flux_q2(k, j - 2, i),
            mass_flux_q2(k, j - 1, i),
            mass_flux_q2(k, j, i),
            scalar_q(k, j - 2, i),
            scalar_q(k, j - 1, i),
            scalar_q(k, j, i),
            scalar_q(k, j + 1, i));

        return -horizontal_divergence
                    .at_t(j, i, flux_q1_plus, flux_q1_minus, flux_q2_plus, flux_q2_minus);
    }

    template <typename ScalarView, typename VerticalMassFluxView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    calculate_vertical_face_flux(const ScalarView& scalar_q,
        const VerticalMassFluxView& mass_flux_vertical,
        const int face_k,
        const int j,
        const int i,
        const int k_begin,
        const int k_end) const noexcept {

        // Closed lower and upper boundaries. k_end is the exclusive final
        // scalar-cell index, so the boundary faces are k_begin - 1 and
        // k_end - 1.
        if (face_k < k_begin || face_k >= k_end - 1) {
            return VVM::real(0.0);
        }

        if (face_k == k_begin) {
            return face_flux.calculate_at_lower_boundary(mass_flux_vertical(face_k, j, i),
                mass_flux_vertical(face_k + 1, j, i),
                scalar_q(face_k, j, i),
                scalar_q(face_k + 1, j, i),
                scalar_q(face_k + 2, j, i));
        }

        if (face_k == k_end - 2) {
            return face_flux.calculate_at_upper_boundary(mass_flux_vertical(face_k - 1, j, i),
                mass_flux_vertical(face_k, j, i),
                scalar_q(face_k - 1, j, i),
                scalar_q(face_k, j, i),
                scalar_q(face_k + 1, j, i));
        }

        return face_flux.calculate(mass_flux_vertical(face_k - 1, j, i),
            mass_flux_vertical(face_k, j, i),
            mass_flux_vertical(face_k + 1, j, i),
            scalar_q(face_k - 1, j, i),
            scalar_q(face_k, j, i),
            scalar_q(face_k + 1, j, i),
            scalar_q(face_k + 2, j, i));
    }

    template <typename ScalarView, typename VerticalMassFluxView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    calculate_vertical_flux_convergence_at_t(const ScalarView& scalar_q,
        const VerticalMassFluxView& mass_flux_vertical,
        const int k,
        const int j,
        const int i,
        const int k_begin,
        const int k_end,
        const VVM::Real inverse_vertical_cell_spacing) const noexcept {

        const VVM::Real flux_above =
            calculate_vertical_face_flux(scalar_q, mass_flux_vertical, k, j, i, k_begin, k_end);

        const VVM::Real flux_below =
            calculate_vertical_face_flux(scalar_q, mass_flux_vertical, k - 1, j, i, k_begin, k_end);

        return -(flux_above - flux_below) * inverse_vertical_cell_spacing;
    }

    template <typename ScalarView,
        typename MassFluxQ1View,
        typename MassFluxQ2View,
        typename VerticalMassFluxView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    calculate_tendency_at_t(const ScalarView& scalar_q,
        const MassFluxQ1View& mass_flux_q1,
        const MassFluxQ2View& mass_flux_q2,
        const VerticalMassFluxView& mass_flux_vertical,
        const int k,
        const int j,
        const int i,
        const int k_begin,
        const int k_end,
        const VVM::Real density,
        const VVM::Real inverse_vertical_cell_spacing) const noexcept {

        const VVM::Real horizontal = calculate_horizontal_flux_convergence_at_t(scalar_q,
            mass_flux_q1,
            mass_flux_q2,
            k,
            j,
            i);

        const VVM::Real vertical = calculate_vertical_flux_convergence_at_t(scalar_q,
            mass_flux_vertical,
            k,
            j,
            i,
            k_begin,
            k_end,
            inverse_vertical_cell_spacing);

        return (horizontal + vertical) / density;
    }
};

inline TakacsScalarTransportDeviceView
make_takacs_scalar_transport_device_view(const Core::Geometry::HorizontalGeometry& geometry,
    const VVM::Real alpha = VVM::real(1.0)) {

    TakacsScalarTransportDeviceView result;

    result.horizontal_divergence = make_horizontal_flux_divergence_device_view(geometry);
    result.face_flux.alpha = alpha;

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_TAKACS_SCALAR_TRANSPORT_HPP
