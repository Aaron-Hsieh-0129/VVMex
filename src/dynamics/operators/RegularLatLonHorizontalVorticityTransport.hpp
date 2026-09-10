#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_VORTICITY_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_VORTICITY_TRANSPORT_HPP

#include <stdexcept>

#include "dynamics/operators/RegularLatLonHorizontalDeformation.hpp"
#include "dynamics/operators/TakacsScalarTransport.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

struct HorizontalVorticityTransportTerms {
    VVM::Real q1 = VVM::real(0.0);
    VVM::Real q2 = VVM::real(0.0);
    VVM::Real vertical = VVM::real(0.0);
};

// Flat-RLL CVVM transport of canonical horizontal vorticity, converted
// back to physical xi and legacy-sign eta tendencies.
//
// Fields uses the RegularLatLonHorizontalDeformationFields contract.
// This component reads only u, v, xi, eta, rho_up, fn1, fn2 and
// inverse_spacing. The separate w view contains physical vertical wind
// at horizontal T.
//
// xi and eta are already divided by rho_up. They are converted to
// canonical components before reconstruction. No further output density
// normalization is applied.
//
// k_begin and k_end specify the complete physical horizontal-vorticity
// column [k_begin, k_end), not an arbitrary subrange. The caller evaluates
// only cells in that range. At least two cells, two horizontal halo cells,
// and valid vertical entries from k_begin-1 through k_end are required.
//
// The vertical flux at k_begin-1 uses the lower one-sided stencil;
// the flux at k_end-1 uses the upper one-sided stencil. Their supplied
// wind and vorticity values determine the flux; it is not forced to zero.
//
// Profiles and backing storage must remain valid for queued work/replay.
// This device component does not allocate, launch, modify inputs, impose
// boundaries, communicate or synchronize.
struct RegularLatLonHorizontalVorticityTransportDeviceView {
    // Share the existing native-location physical/canonical conversions.
    RegularLatLonHorizontalDeformationDeviceView components;

    Core::Geometry::GeometryField2D jacobian_t;
    Core::Geometry::GeometryField2D jacobian_u;
    Core::Geometry::GeometryField2D jacobian_v;
    Core::Geometry::GeometryField2D jacobian_z;

    TakacsFaceFluxDeviceView face_flux;

    template <bool Xi, typename Fields>
    KOKKOS_INLINE_FUNCTION VVM::Real
    scalar(const Fields& fields, int k, int j, int i) const noexcept {
        if constexpr (Xi) {
            return components.omega1_over_rho(fields, k, j, i);
        }
        else {
            return components.omega2_over_rho(fields, k, j, i);
        }
    }

    template <bool Xi, typename Fields>
    KOKKOS_INLINE_FUNCTION VVM::Real
    horizontal_mass_flux(const Fields& fields, int k, int j, int i, int direction) const noexcept {

        const int next_j = Xi ? j + 1 : j;
        const int next_i = Xi ? i : i + 1;
        VVM::Real lower;
        VVM::Real upper;
        VVM::Real jacobian;

        if (direction == 0) {
            lower = components.u1(fields, k, j, i) + components.u1(fields, k, next_j, next_i);
            upper =
                components.u1(fields, k + 1, j, i) + components.u1(fields, k + 1, next_j, next_i);

            if constexpr (Xi) {
                jacobian = jacobian_z(j, i);
            }
            else {
                jacobian = jacobian_t(j, i + 1);
            }
        }
        else {
            lower = components.u2(fields, k, j, i) + components.u2(fields, k, next_j, next_i);
            upper =
                components.u2(fields, k + 1, j, i) + components.u2(fields, k + 1, next_j, next_i);

            if constexpr (Xi) {
                jacobian = jacobian_t(j + 1, i);
            }
            else {
                jacobian = jacobian_z(j, i);
            }
        }

        // CVVM reconstructs with this Jacobian-weighted transport velocity.
        return VVM::real(0.25) * (fields.fn1(k) * upper + fields.fn2(k) * lower) * jacobian;
    }

    template <bool Xi, typename Fields>
    KOKKOS_INLINE_FUNCTION VVM::Real
    horizontal_face(const Fields& fields, int k, int j, int i, int direction) const noexcept {

        const int di = direction == 0 ? 1 : 0;
        const int dj = direction == 1 ? 1 : 0;

        // Reuse the arithmetic helper with CVVM's weighted face velocities.
        // This does not change the existing scalar-transport convention.
        return face_flux.calculate(horizontal_mass_flux<Xi>(fields, k, j - dj, i - di, direction),
            horizontal_mass_flux<Xi>(fields, k, j, i, direction),
            horizontal_mass_flux<Xi>(fields, k, j + dj, i + di, direction),
            scalar<Xi>(fields, k, j - dj, i - di),
            scalar<Xi>(fields, k, j, i),
            scalar<Xi>(fields, k, j + dj, i + di),
            scalar<Xi>(fields, k, j + 2 * dj, i + 2 * di));
    }

    template <bool Xi, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    vertical_mass_flux(const Fields& fields, const WView& w, int k, int j, int i) const noexcept {

        const int next_j = Xi ? j + 1 : j;
        const int next_i = Xi ? i : i + 1;

        return VVM::real(0.25) *
               (fields.rho_up(k) * (w(k, j, i) + w(k, next_j, next_i)) +
                   fields.rho_up(k + 1) * (w(k + 1, j, i) + w(k + 1, next_j, next_i)));
    }

    template <bool Xi, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION VVM::Real
    vertical_face(const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {

        if (k == k_begin - 1) {
            return face_flux.calculate_at_lower_boundary(vertical_mass_flux<Xi>(fields, w, k, j, i),
                vertical_mass_flux<Xi>(fields, w, k + 1, j, i),
                scalar<Xi>(fields, k, j, i),
                scalar<Xi>(fields, k + 1, j, i),
                scalar<Xi>(fields, k + 2, j, i));
        }

        if (k == k_end - 1) {
            return face_flux.calculate_at_upper_boundary(
                vertical_mass_flux<Xi>(fields, w, k - 1, j, i),
                vertical_mass_flux<Xi>(fields, w, k, j, i),
                scalar<Xi>(fields, k - 1, j, i),
                scalar<Xi>(fields, k, j, i),
                scalar<Xi>(fields, k + 1, j, i));
        }

        return face_flux.calculate(vertical_mass_flux<Xi>(fields, w, k - 1, j, i),
            vertical_mass_flux<Xi>(fields, w, k, j, i),
            vertical_mass_flux<Xi>(fields, w, k + 1, j, i),
            scalar<Xi>(fields, k - 1, j, i),
            scalar<Xi>(fields, k, j, i),
            scalar<Xi>(fields, k + 1, j, i),
            scalar<Xi>(fields, k + 2, j, i));
    }

    template <bool Xi, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION HorizontalVorticityTransportTerms
    calculate(const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {

        const VVM::Real jacobian = Xi ? jacobian_v(j, i) : jacobian_u(j, i);
        const VVM::Real physical_scale = Xi ? components.h1_at_v(j, i) : -components.h2_at_u(j, i);

        HorizontalVorticityTransportTerms result;
        result.q1 = -physical_scale *
                    (horizontal_face<Xi>(fields, k, j, i, 0) -
                        horizontal_face<Xi>(fields, k, j, i - 1, 0)) /
                    (jacobian * components.dq1);
        result.q2 = -physical_scale *
                    (horizontal_face<Xi>(fields, k, j, i, 1) -
                        horizontal_face<Xi>(fields, k, j - 1, i, 1)) /
                    (jacobian * components.dq2);
        result.vertical = -physical_scale * fields.inverse_spacing(k) *
                          (vertical_face<Xi>(fields, w, k, j, i, k_begin, k_end) -
                              vertical_face<Xi>(fields, w, k - 1, j, i, k_begin, k_end));
        return result;
    }

    template <typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION HorizontalVorticityTransportTerms
    calculate_xi_at_v(
        const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {

        return calculate<true>(fields, w, k, j, i, k_begin, k_end);
    }

    template <typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION HorizontalVorticityTransportTerms
    calculate_eta_at_u(
        const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {

        return calculate<false>(fields, w, k, j, i, k_begin, k_end);
    }
};

inline RegularLatLonHorizontalVorticityTransportDeviceView
make_regular_lat_lon_horizontal_vorticity_transport_device_view(
    const Core::Geometry::HorizontalGeometry& geometry, VVM::Real alpha = VVM::real(1.0)) {

    using Core::Geometry::HorizontalLocation;

    RegularLatLonHorizontalVorticityTransportDeviceView result;
    result.components = make_regular_lat_lon_horizontal_deformation_device_view(geometry);

    if (geometry.layout().halo < 2) {
        throw std::invalid_argument("RegularLatLonHorizontalVorticityTransport requires at least "
                                    "two horizontal halo cells.");
    }

    result.jacobian_t = geometry.device_view(HorizontalLocation::T).sqrt_g;
    result.jacobian_u = geometry.device_view(HorizontalLocation::U).sqrt_g;
    result.jacobian_v = geometry.device_view(HorizontalLocation::V).sqrt_g;
    result.jacobian_z = geometry.device_view(HorizontalLocation::Z).sqrt_g;
    result.face_flux.alpha = alpha;
    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_VORTICITY_TRANSPORT_HPP
