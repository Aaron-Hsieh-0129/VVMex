#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_TOP_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_TOP_TRANSPORT_HPP

#include <cmath>
#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"
#include "dynamics/operators/TakacsScalarTransport.hpp"

namespace VVM::Dynamics::Operators {

struct TopTransportTerms {
    Real q1 = real(0.0);
    Real q2 = real(0.0);
    Real vertical = real(0.0);
};

// CVVM ZETA_3D transport at a rigid lid, in a stationary horizontal chart.
//
// Fields supplies callable members/accessors:
//   u1, u2                 contravariant winds at U/V, wind levels
//   w                      physical vertical wind at T, interface levels
//   omega3_over_rho         relative omega^3/rho at Z, wind levels
//   f3_at_z                vertical planetary vorticity, NOT divided by rho
//   rho, rho_up            reference density at wind/interface levels
//   inverse_spacing_mid    inverse vertical cell thickness at wind levels
//
// Horizontal transport is -1/J * partial_a(J rho u^a q), a = 1,2.
// q is omega^3/rho or f^3/rho, selected by planetary. The result is an
// unnormalized omega^3 tendency. Do not divide the result by rho again.
// Planetary stretching belongs to GeneralizedTopDeformation, not here.
//
// Scope: z-independent horizontal mapping, unchanged physical z, uniform
// computational dq1/dq2, existing CVVM/VVM staggering, and w(top)=0.
// No diagonal-metric assumption: the input winds are already contravariant.
// Component transformations and panel halo exchange belong to the caller.
//
// The caller supplies two horizontal halo cells, top>=2, valid wind-level
// data at top-2..top and interface data at top-2..top-1, positive finite
// densities/Jacobians/spacings, and the rigid-lid boundary condition.
// No data above top are read. The vertical lid flux is identically zero;
// transport through the LOWER face of the top cell is retained.
//
// No allocation, input mutation, launch, communication or synchronization.
struct GeneralizedTopTransportDeviceView {
    Core::Geometry::GeometryField2D jacobian_u;
    Core::Geometry::GeometryField2D jacobian_v;
    Core::Geometry::GeometryField2D jacobian_z;

    Real dq1 = real(0.0);
    Real dq2 = real(0.0);
    TakacsFaceFluxDeviceView face_flux;

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    scalar(const Fields& fields, int k, int j, int i, bool planetary) const noexcept {
        return planetary ? fields.f3_at_z(j, i) / fields.rho(k) : fields.omega3_over_rho(k, j, i);
    }

    // direction: 0 = q1; 1 = q2.
    // Four native U/V winds are interpolated to a face of the Z control cell.
    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    mass_flux(const Fields& fields, int k, int j, int i, int direction) const noexcept {
        Real sum = real(0.0);

        for (int jj = j; jj <= j + 1; ++jj) {
            for (int ii = i; ii <= i + 1; ++ii) {
                sum += direction == 0 ? fields.u1(k, jj, ii) : fields.u2(k, jj, ii);
            }
        }

        const Real jacobian = direction == 0 ? jacobian_v(j, i + 1) : jacobian_u(j + 1, i);

        // Reconstruct the J-weighted mass flux, including the Takacs
        // correction, as in CVVM. Do not apply another face Jacobian.
        return real(0.25) * fields.rho(k) * sum * jacobian;
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    face(const Fields& fields, int k, int j, int i, int direction, bool planetary) const noexcept {
        const int di = direction == 0 ? 1 : 0;
        const int dj = direction == 1 ? 1 : 0;

        return face_flux.calculate(mass_flux(fields, k, j - dj, i - di, direction),
            mass_flux(fields, k, j, i, direction),
            mass_flux(fields, k, j + dj, i + di, direction),
            scalar(fields, k, j - dj, i - di, planetary),
            scalar(fields, k, j, i, planetary),
            scalar(fields, k, j + dj, i + di, planetary),
            scalar(fields, k, j + 2 * dj, i + 2 * di, planetary));
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    vertical_mass_flux(const Fields& fields, int k, int j, int i) const noexcept {
        // J cancels from the vertical divergence for this z-independent
        // horizontal mapping. The supplied mass flux is rho_up*w, not J*rho_up*w.
        return real(0.25) * fields.rho_up(k) *
               (fields.w(k, j, i) + fields.w(k, j, i + 1) + fields.w(k, j + 1, i) +
                   fields.w(k, j + 1, i + 1));
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION TopTransportTerms
    calculate_at_z(
        const Fields& fields, int top, int j, int i, bool planetary = false) const noexcept {
        TopTransportTerms result;

        result.q1 =
            -(face(fields, top, j, i, 0, planetary) - face(fields, top, j, i - 1, 0, planetary)) /
            (jacobian_z(j, i) * dq1);

        result.q2 =
            -(face(fields, top, j, i, 1, planetary) - face(fields, top, j - 1, i, 1, planetary)) /
            (jacobian_z(j, i) * dq2);

        // -[F_lid - F_lower]/dz = +F_lower/dz, because F_lid=0.
        // At the lower face, negative flow uses the centered branch of
        // the one-sided Takacs rule; this is not flow THROUGH the lid.
        result.vertical =
            face_flux.calculate_at_upper_boundary(vertical_mass_flux(fields, top - 2, j, i),
                vertical_mass_flux(fields, top - 1, j, i),
                scalar(fields, top - 2, j, i, planetary),
                scalar(fields, top - 1, j, i, planetary),
                scalar(fields, top, j, i, planetary)) *
            fields.inverse_spacing_mid(top);

        return result;
    }
};

inline GeneralizedTopTransportDeviceView
make_generalized_top_transport_device_view(const Core::Geometry::HorizontalGeometry& geometry,
    Real alpha = real(1.0)) {
    using Core::Geometry::HorizontalLocation;

    if (geometry.layout().halo < 2) {
        throw std::invalid_argument("GeneralizedTopTransport requires two horizontal halo cells.");
    }

    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();

    if (!(dq1 > real(0.0)) || !(dq2 > real(0.0)) || !std::isfinite(dq1) || !std::isfinite(dq2) ||
        !std::isfinite(alpha)) {
        throw std::invalid_argument("GeneralizedTopTransport requires finite positive spacing "
                                    "and a finite Takacs coefficient.");
    }

    GeneralizedTopTransportDeviceView result;
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
