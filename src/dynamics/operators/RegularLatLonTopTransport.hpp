#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_TRANSPORT_HPP

#include <stdexcept>

#include "dynamics/operators/GeneralizedTopTransport.hpp"

namespace VVM::Dynamics::Operators {

// Legacy field adapter only. Production uses GeneralizedTopTransport
// directly; this interface remains for existing field-level callers.
//
// Legacy inputs: physical u/v/w, zeta already divided by rho, and f_at_z
// not density-normalized. All transport stencils belong to the generalized
// operator. No horizontal-vorticity/deformation adapter is needed here.
struct RegularLatLonTopTransportDeviceView {
    using Base = GeneralizedTopTransportDeviceView;

    Base numerical;

    Core::Geometry::GeometryField2D h1_at_u;
    Core::Geometry::GeometryField2D h2_at_v;

    template <typename Fields>
    struct LegacyFields {
        const Fields& fields;
        const RegularLatLonTopTransportDeviceView& adapter;

        KOKKOS_INLINE_FUNCTION Real
        u1(int k, int j, int i) const noexcept {
            return fields.u(k, j, i) / adapter.h1_at_u(j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        u2(int k, int j, int i) const noexcept {
            return fields.v(k, j, i) / adapter.h2_at_v(j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        w(int k, int j, int i) const noexcept {
            return fields.w(k, j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        omega3_over_rho(int k, int j, int i) const noexcept {
            // The legacy zeta accessor is already density-normalized.
            return fields.zeta(k, j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        f3_at_z(int j, int i) const noexcept {
            return fields.f_at_z(j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        rho(int k) const noexcept {
            return fields.rho(k);
        }

        KOKKOS_INLINE_FUNCTION Real
        rho_up(int k) const noexcept {
            return fields.rho_up(k);
        }

        KOKKOS_INLINE_FUNCTION Real
        inverse_spacing_mid(int k) const noexcept {
            return fields.inverse_spacing_mid(k);
        }
    };

    KOKKOS_INLINE_FUNCTION Base
    generalized_view() const noexcept {
        return numerical;
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    scalar(const Fields& fields, int k, int j, int i, bool planetary) const noexcept {
        return numerical.scalar(LegacyFields<Fields>{fields, *this}, k, j, i, planetary);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    mass_flux(const Fields& fields, int k, int j, int i, int direction) const noexcept {
        return numerical.mass_flux(LegacyFields<Fields>{fields, *this}, k, j, i, direction);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    face(const Fields& fields, int k, int j, int i, int direction, bool planetary) const noexcept {
        return numerical.face(LegacyFields<Fields>{fields, *this}, k, j, i, direction, planetary);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    vertical_mass_flux(const Fields& fields, int k, int j, int i) const noexcept {
        return numerical.vertical_mass_flux(LegacyFields<Fields>{fields, *this}, k, j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION TopTransportTerms
    calculate_at_z(
        const Fields& fields, int top, int j, int i, bool planetary = false) const noexcept {
        return numerical.calculate_at_z(LegacyFields<Fields>{fields, *this}, top, j, i, planetary);
    }
};

inline RegularLatLonTopTransportDeviceView
make_regular_lat_lon_top_transport_device_view(const Core::Geometry::HorizontalGeometry& geometry,
    Real alpha = real(1.0)) {
    using Core::Geometry::GeometryKind;
    using Core::Geometry::HorizontalLocation;

    if (geometry.kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument("RegularLatLonTopTransport compatibility requires "
                                    "regular latitude-longitude geometry.");
    }

    RegularLatLonTopTransportDeviceView result;

    result.numerical = make_generalized_top_transport_device_view(geometry, alpha);
    result.h1_at_u = geometry.device_view(HorizontalLocation::U).contravariant_to_physical.a11;
    result.h2_at_v = geometry.device_view(HorizontalLocation::V).contravariant_to_physical.a22;

    return result;
}

} // namespace VVM::Dynamics::Operators
#endif
