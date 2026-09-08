#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_TRANSPORT_HPP

#include "dynamics/operators/RegularLatLonHorizontalVorticityTransport.hpp"

namespace VVM::Dynamics::Operators {

// CVVM ZETA_3D transport at the rigid lid. zeta is relative vertical
// vorticity already divided by rho; u/v/w are physical winds. The optional
// planetary contribution is evaluated separately from relative transport.
struct RegularLatLonTopTransportDeviceView {
    RegularLatLonHorizontalVorticityTransportDeviceView horizontal;

    template<typename Fields>
    KOKKOS_INLINE_FUNCTION
    Real scalar(const Fields& f, int k, int j, int i, bool planetary) const {
        return planetary ? f.f_at_z(j, i) / f.rho(k) : f.zeta(k, j, i);
    }

    template<typename Fields>
    KOKKOS_INLINE_FUNCTION
    Real mass_flux(const Fields& f, int k, int j, int i, int direction) const {
        Real sum = real(0.0);
        for (int jj = j; jj <= j + 1; ++jj) {
            for (int ii = i; ii <= i + 1; ++ii) {
                sum += direction == 0 ? horizontal.components.u1(f, k, jj, ii)
                                      : horizontal.components.u2(f, k, jj, ii);
            }
        }
        const Real jacobian = direction == 0 ? horizontal.jacobian_v(j, i + 1)
                                             : horizontal.jacobian_u(j + 1, i);
        return real(0.25) * f.rho(k) * sum * jacobian;
    }

    template<typename Fields>
    KOKKOS_INLINE_FUNCTION
    Real face(const Fields& f, int k, int j, int i, int direction, bool planetary) const {
        const int di = direction == 0 ? 1 : 0;
        const int dj = direction == 1 ? 1 : 0;
        return horizontal.face_flux.calculate(
            mass_flux(f, k, j - dj, i - di, direction), mass_flux(f, k, j, i, direction),
            mass_flux(f, k, j + dj, i + di, direction),
            scalar(f, k, j - dj, i - di, planetary), scalar(f, k, j, i, planetary),
            scalar(f, k, j + dj, i + di, planetary), scalar(f, k, j + 2 * dj, i + 2 * di, planetary));
    }

    template<typename Fields>
    KOKKOS_INLINE_FUNCTION
    Real vertical_mass_flux(const Fields& f, int k, int j, int i) const {
        return real(0.25) * f.rho_up(k)
            * (f.w(k, j, i) + f.w(k, j, i + 1) + f.w(k, j + 1, i) + f.w(k, j + 1, i + 1));
    }

    template<typename Fields>
    KOKKOS_INLINE_FUNCTION
    HorizontalVorticityTransportTerms calculate_at_z(const Fields& f, int top, int j, int i, bool planetary = false) const {
        HorizontalVorticityTransportTerms result;
        result.q1 = -(face(f, top, j, i, 0, planetary) - face(f, top, j, i - 1, 0, planetary))
            / (horizontal.jacobian_z(j, i) * horizontal.components.dq1);
        result.q2 = -(face(f, top, j, i, 1, planetary) - face(f, top, j - 1, i, 1, planetary))
            / (horizontal.jacobian_z(j, i) * horizontal.components.dq2);
        // W(top)=0. Only the lower flux enters, with the original one-sided
        // Takacs rule (centered when flow enters from the rigid lid).
        result.vertical = horizontal.face_flux.calculate_at_upper_boundary(
            vertical_mass_flux(f, top - 2, j, i), vertical_mass_flux(f, top - 1, j, i),
            scalar(f, top - 2, j, i, planetary), scalar(f, top - 1, j, i, planetary),
            scalar(f, top, j, i, planetary)) * f.inverse_spacing_mid(top);
        return result;
    }
};

inline RegularLatLonTopTransportDeviceView make_regular_lat_lon_top_transport_device_view(
    const Core::Geometry::HorizontalGeometry& geometry, Real alpha = real(1.0)) {
    return {make_regular_lat_lon_horizontal_vorticity_transport_device_view(geometry, alpha)};
}

} // namespace VVM::Dynamics::Operators
#endif
