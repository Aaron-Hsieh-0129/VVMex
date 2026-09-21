#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_VORTICITY_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_VORTICITY_TRANSPORT_HPP

#include "dynamics/operators/GeneralizedHorizontalVorticityTransport.hpp"
#include "dynamics/operators/RegularLatLonHorizontalDeformation.hpp"

namespace VVM::Dynamics::Operators {

// Temporary compatibility adapter for existing RLL callers/tests.
// All transport stencils live in the generalized base. Only the old
// physical-input/physical-output representation contract remains here.
// Production horizontal transport bypasses this adapter.
struct RegularLatLonHorizontalVorticityTransportDeviceView
    : GeneralizedHorizontalVorticityTransportDeviceView {
    using Base = GeneralizedHorizontalVorticityTransportDeviceView;

    RegularLatLonHorizontalDeformationDeviceView components;

    template <typename Fields>
    struct LegacyFields {
        const Fields& fields;
        const RegularLatLonHorizontalDeformationDeviceView& components;

        KOKKOS_INLINE_FUNCTION Real
        u1(int k, int j, int i) const noexcept {
            return components.u1(fields, k, j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        u2(int k, int j, int i) const noexcept {
            return components.u2(fields, k, j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        omega1_over_rho(int k, int j, int i) const noexcept {
            return components.omega1_over_rho(fields, k, j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        omega2_over_rho(int k, int j, int i) const noexcept {
            return components.omega2_over_rho(fields, k, j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        rho_up(int k) const noexcept {
            return fields.rho_up(k);
        }

        KOKKOS_INLINE_FUNCTION Real
        fn1(int k) const noexcept {
            return fields.fn1(k);
        }

        KOKKOS_INLINE_FUNCTION Real
        fn2(int k) const noexcept {
            return fields.fn2(k);
        }

        KOKKOS_INLINE_FUNCTION Real
        inverse_spacing(int k) const noexcept {
            return fields.inverse_spacing(k);
        }
    };

    template <bool Xi, typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    scalar(const Fields& fields, int k, int j, int i) const noexcept {
        return Base::scalar<Xi>(LegacyFields<Fields>{fields, components}, k, j, i);
    }

    template <bool Xi, typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    horizontal_mass_flux(const Fields& fields, int k, int j, int i, int direction) const noexcept {
        return Base::horizontal_mass_flux<Xi>(LegacyFields<Fields>{fields, components},
            k,
            j,
            i,
            direction);
    }

    template <bool Xi, typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    horizontal_face(const Fields& fields, int k, int j, int i, int direction) const noexcept {
        return Base::horizontal_face<Xi>(LegacyFields<Fields>{fields, components},
            k,
            j,
            i,
            direction);
    }

    template <bool Xi, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION Real
    vertical_mass_flux(const Fields& fields, const WView& w, int k, int j, int i) const noexcept {
        return Base::vertical_mass_flux<Xi>(LegacyFields<Fields>{fields, components}, w, k, j, i);
    }

    template <bool Xi, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION Real
    vertical_face(const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {
        return Base::vertical_face<Xi>(LegacyFields<Fields>{fields, components},
            w,
            k,
            j,
            i,
            k_begin,
            k_end);
    }

    template <bool Xi, typename Fields, typename WView>
    KOKKOS_INLINE_FUNCTION HorizontalVorticityTransportTerms
    calculate(const Fields& fields, const WView& w, int k, int j, int i, int k_begin, int k_end)
        const noexcept {
        auto result = Base::calculate<Xi>(LegacyFields<Fields>{fields, components},
            w,
            k,
            j,
            i,
            k_begin,
            k_end);

        const Real scale = Xi ? components.h1_at_v(j, i) : -components.h2_at_u(j, i);

        result.q1 *= scale;
        result.q2 *= scale;
        result.vertical *= scale;

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
    const Core::Geometry::HorizontalGeometry& geometry, Real alpha = real(1.0)) {
    RegularLatLonHorizontalVorticityTransportDeviceView result;

    result.components = make_regular_lat_lon_horizontal_deformation_device_view(geometry);

    static_cast<GeneralizedHorizontalVorticityTransportDeviceView&>(result) =
        make_generalized_horizontal_vorticity_transport_device_view(geometry, alpha);

    return result;
}

} // namespace VVM::Dynamics::Operators
#endif
