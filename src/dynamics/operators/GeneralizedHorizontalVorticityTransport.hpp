#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_HORIZONTAL_VORTICITY_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_HORIZONTAL_VORTICITY_TRANSPORT_HPP

#include <cmath>
#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"
#include "dynamics/operators/TakacsScalarTransport.hpp"

namespace VVM::Dynamics::Operators {

struct HorizontalVorticityTransportTerms {
    Real q1 = real(0.0);
    Real q2 = real(0.0);
    Real vertical = real(0.0);
};

// Componentwise conservative transport in a stationary horizontal chart:
//
//   T^a = -1/J * d_b[J rho u^b (omega^a/rho)]
//         -d_z[rho w (omega^a/rho)],  b = 1, 2.
//
// This is the transport part of the contravariant vorticity equation,
// not a standalone covariant derivative of a vector. It must be paired
// with the corresponding component-form stretching/twisting terms.
//
// Fields supplies callable members/accessors:
//   u1, u2                contravariant winds at U, V
//   omega1_over_rho        omega^1/rho_up at V, vertical interfaces
//   omega2_over_rho        omega^2/rho_up at U, vertical interfaces
//   rho_up                density at vertical interfaces
//   fn1, fn2              mass-weighted vertical interpolation factors
//   inverse_spacing       inverse horizontal-vorticity cell thickness
// WView supplies physical w at horizontal T, vertical interfaces.
//
// omega^2 is the actual tensor component, NOT VVM's eta_con = -omega^2.
// No physical-component or diagonal-metric conversion belongs here.
// Nonorthogonal metrics enter through the supplied components and J;
// this transport stencil does not perform metric lowering/raising.
//
// Scope: z-independent horizontal mapping, unchanged physical vertical
// coordinate, uniform computational dq1/dq2, existing VVM staggering.
// Panel halos must already be expressed in the receiving chart.
//
// [k_begin, k_end) is the complete physical vorticity column, with at
// least two levels. Two horizontal halos and vertical entries from
// k_begin-1 through k_end are required. Boundary fluxes retain CVVM's
// one-sided reconstruction; supplied data, not this operator, impose BCs.
// No allocation, launch, synchronization, communication or input writes.
struct GeneralizedHorizontalVorticityTransportDeviceView {
    Core::Geometry::GeometryField2D jacobian_t;
    Core::Geometry::GeometryField2D jacobian_u;
    Core::Geometry::GeometryField2D jacobian_v;
    Core::Geometry::GeometryField2D jacobian_z;

    Real dq1 = real(0.0);
    Real dq2 = real(0.0);
    TakacsFaceFluxDeviceView face_flux;

    template <bool First, typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    scalar(const Fields& fields, int k, int j, int i) const noexcept {
        if constexpr (First) {
            return fields.omega1_over_rho(k, j, i);
        }
        else {
            return fields.omega2_over_rho(k, j, i);
        }
    }

    template <bool First, typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    horizontal_mass_flux(const Fields& fields, int k, int j, int i, int direction) const noexcept {
        const int next_j = First ? j + 1 : j;
        const int next_i = First ? i : i + 1;
        Real lower, upper, jacobian;

        if (direction == 0) {
            lower = fields.u1(k, j, i) + fields.u1(k, next_j, next_i);
            upper = fields.u1(k + 1, j, i) + fields.u1(k + 1, next_j, next_i);
            jacobian = First ? jacobian_z(j, i) : jacobian_t(j, i + 1);
        }
        else {
            lower = fields.u2(k, j, i) + fields.u2(k, next_j, next_i);
            upper = fields.u2(k + 1, j, i) + fields.u2(k + 1, next_j, next_i);
            jacobian = First ? jacobian_t(j + 1, i) : jacobian_z(j, i);
        }

        // CVVM reconstructs the J-weighted flux itself, including its
        // Takacs correction. Do not multiply this face flux by J again.
        return real(0.25) * (fields.fn1(k) * upper + fields.fn2(k) * lower) * jacobian;
    }

    template <bool First, typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    horizontal_face(const Fields& fields, int k, int j, int i, int direction) const noexcept {
        const int di = direction == 0 ? 1 : 0;
        const int dj = direction == 1 ? 1 : 0;

        return face_flux.calculate(
            horizontal_mass_flux<First>(fields, k, j - dj, i - di, direction),
            horizontal_mass_flux<First>(fields, k, j, i, direction),
            horizontal_mass_flux<First>(fields, k, j + dj, i + di, direction),
            scalar<First>(fields, k, j - dj, i - di),
            scalar<First>(fields, k, j, i),
            scalar<First>(fields, k, j + dj, i + di),
            scalar<First>(fields, k, j + 2 * dj, i + 2 * di));
    }

    template <bool First, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION Real
    vertical_mass_flux(const Fields& fields, const WView& w, int k, int j, int i) const noexcept {
        const int next_j = First ? j + 1 : j;
        const int next_i = First ? i : i + 1;

        return real(0.25) * (fields.rho_up(k) * (w(k, j, i) + w(k, next_j, next_i)) +
                                fields.rho_up(k + 1) * (w(k + 1, j, i) + w(k + 1, next_j, next_i)));
    }

    template <bool First, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION Real
    vertical_face(const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {
        if (k == k_begin - 1) {
            return face_flux.calculate_at_lower_boundary(
                vertical_mass_flux<First>(fields, w, k, j, i),
                vertical_mass_flux<First>(fields, w, k + 1, j, i),
                scalar<First>(fields, k, j, i),
                scalar<First>(fields, k + 1, j, i),
                scalar<First>(fields, k + 2, j, i));
        }

        if (k == k_end - 1) {
            return face_flux.calculate_at_upper_boundary(
                vertical_mass_flux<First>(fields, w, k - 1, j, i),
                vertical_mass_flux<First>(fields, w, k, j, i),
                scalar<First>(fields, k - 1, j, i),
                scalar<First>(fields, k, j, i),
                scalar<First>(fields, k + 1, j, i));
        }

        return face_flux.calculate(vertical_mass_flux<First>(fields, w, k - 1, j, i),
            vertical_mass_flux<First>(fields, w, k, j, i),
            vertical_mass_flux<First>(fields, w, k + 1, j, i),
            scalar<First>(fields, k - 1, j, i),
            scalar<First>(fields, k, j, i),
            scalar<First>(fields, k + 1, j, i),
            scalar<First>(fields, k + 2, j, i));
    }

    template <bool First, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION HorizontalVorticityTransportTerms
    calculate(const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {
        const Real jacobian = First ? jacobian_v(j, i) : jacobian_u(j, i);

        HorizontalVorticityTransportTerms result;
        result.q1 = -(horizontal_face<First>(fields, k, j, i, 0) -
                        horizontal_face<First>(fields, k, j, i - 1, 0)) /
                    (jacobian * dq1);

        result.q2 = -(horizontal_face<First>(fields, k, j, i, 1) -
                        horizontal_face<First>(fields, k, j - 1, i, 1)) /
                    (jacobian * dq2);

        result.vertical = -fields.inverse_spacing(k) *
                          (vertical_face<First>(fields, w, k, j, i, k_begin, k_end) -
                              vertical_face<First>(fields, w, k - 1, j, i, k_begin, k_end));

        return result;
    }

    template <typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION HorizontalVorticityTransportTerms
    calculate_omega1_at_v(
        const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {
        return calculate<true>(fields, w, k, j, i, k_begin, k_end);
    }

    template <typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION HorizontalVorticityTransportTerms
    calculate_omega2_at_u(
        const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {
        return calculate<false>(fields, w, k, j, i, k_begin, k_end);
    }
};

inline GeneralizedHorizontalVorticityTransportDeviceView
make_generalized_horizontal_vorticity_transport_device_view(
    const Core::Geometry::HorizontalGeometry& geometry, Real alpha = real(1.0)) {
    using Core::Geometry::HorizontalLocation;

    if (geometry.layout().halo < 2) {
        throw std::invalid_argument(
            "GeneralizedHorizontalVorticityTransport requires two horizontal halo cells.");
    }

    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();

    if (!(dq1 > real(0.0)) || !(dq2 > real(0.0)) || !std::isfinite(dq1) || !std::isfinite(dq2) ||
        !std::isfinite(alpha)) {
        throw std::invalid_argument(
            "GeneralizedHorizontalVorticityTransport requires finite positive spacing "
            "and a finite Takacs coefficient.");
    }

    GeneralizedHorizontalVorticityTransportDeviceView result;
    result.jacobian_t = geometry.device_view(HorizontalLocation::T).sqrt_g;
    result.jacobian_u = geometry.device_view(HorizontalLocation::U).sqrt_g;
    result.jacobian_v = geometry.device_view(HorizontalLocation::V).sqrt_g;
    result.jacobian_z = geometry.device_view(HorizontalLocation::Z).sqrt_g;
    result.dq1 = dq1;
    result.dq2 = dq2;
    result.face_flux.alpha = alpha;

    return result;
}

} // namespace VVM::Dynamics::Operators
#endif
