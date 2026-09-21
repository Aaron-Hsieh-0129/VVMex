#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_TRANSPORT_HPP

#include "dynamics/operators/GeneralizedTopTransport.hpp"
#include "dynamics/operators/RegularLatLonHorizontalVorticityTransport.hpp"

namespace VVM::Dynamics::Operators {

// Temporary compatibility adapter for existing RLL callers.
// Legacy fields contain physical u/v/w, zeta already divided by rho,
// and f_at_z NOT divided by rho. Production bypasses this adapter.
// No independent top-transport stencil remains here.
struct RegularLatLonTopTransportDeviceView {
    using Base = GeneralizedTopTransportDeviceView;

    // Preserve the legacy public layout until its remaining consumers move.
    RegularLatLonHorizontalVorticityTransportDeviceView horizontal;

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
        w(int k, int j, int i) const noexcept {
            return fields.w(k, j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        omega3_over_rho(int k, int j, int i) const noexcept {
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
        Base result;
        result.jacobian_u = horizontal.jacobian_u;
        result.jacobian_v = horizontal.jacobian_v;
        result.jacobian_z = horizontal.jacobian_z;
        result.dq1 = horizontal.components.dq1;
        result.dq2 = horizontal.components.dq2;
        result.face_flux = horizontal.face_flux;

        return result;
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    scalar(const Fields& fields, int k, int j, int i, bool planetary) const noexcept {
        return generalized_view().scalar(LegacyFields<Fields>{fields, horizontal.components},
            k,
            j,
            i,
            planetary);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    mass_flux(const Fields& fields, int k, int j, int i, int direction) const noexcept {
        return generalized_view().mass_flux(LegacyFields<Fields>{fields, horizontal.components},
            k,
            j,
            i,
            direction);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    face(const Fields& fields, int k, int j, int i, int direction, bool planetary) const noexcept {
        return generalized_view().face(LegacyFields<Fields>{fields, horizontal.components},
            k,
            j,
            i,
            direction,
            planetary);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    vertical_mass_flux(const Fields& fields, int k, int j, int i) const noexcept {
        return generalized_view()
            .vertical_mass_flux(LegacyFields<Fields>{fields, horizontal.components}, k, j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION HorizontalVorticityTransportTerms
    calculate_at_z(
        const Fields& fields, int top, int j, int i, bool planetary = false) const noexcept {
        const auto t =
            generalized_view().calculate_at_z(LegacyFields<Fields>{fields, horizontal.components},
                top,
                j,
                i,
                planetary);

        return {t.q1, t.q2, t.vertical};
    }
};

inline RegularLatLonTopTransportDeviceView
make_regular_lat_lon_top_transport_device_view(const Core::Geometry::HorizontalGeometry& geometry,
    Real alpha = real(1.0)) {
    return {make_regular_lat_lon_horizontal_vorticity_transport_device_view(geometry, alpha)};
}

} // namespace VVM::Dynamics::Operators
#endif
