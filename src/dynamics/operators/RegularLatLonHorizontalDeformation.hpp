#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_DEFORMATION_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_DEFORMATION_HPP

#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Borrowed views only. All volumes use (k, j, i).
//
// u, v:
//     Physical winds at U and V.
//
// xi, eta:
//     Physical eastward and legacy-sign horizontal vorticity divided by
//     rhobar_up, at V and U respectively.
//
// zeta:
//     Physical relative vertical vorticity divided by rhobar, at Z.
//
// f_at_z:
//     Planetary vertical vorticity at Z, without density normalization.
//
// rho, rho_up:
//     Reference density at wind and horizontal-vorticity levels.
//
// fn1, fn2:
//     CVVM mass-weighted vertical factors:
//       fn1(k) = fact1_xi_eta(k) * rhobar(k+1)
//       fn2(k) = fact2_xi_eta(k) * rhobar(k)
//     The caller prepares these outside capture.
//
// inverse_spacing:
//     flex_height_coef_up(k) / dz, prepared outside capture.
//
// All views and geometry must remain valid for queued work and replay.
// The caller supplies valid halos, extents and positive used densities
// and spacing. No field is modified by this component.
template <typename VolumeView, typename ProfileView, typename PlaneView>
struct RegularLatLonHorizontalDeformationFields {
    VolumeView u;
    VolumeView v;
    VolumeView xi;
    VolumeView eta;
    VolumeView zeta;
    PlaneView f_at_z;
    ProfileView rho;
    ProfileView rho_up;
    ProfileView fn1;
    ProfileView fn2;
    ProfileView inverse_spacing;
};

struct HorizontalDeformationTerms {
    VVM::Real stretching = VVM::real(0.0);
    VVM::Real twisting = VVM::real(0.0);
    VVM::Real planetary = VVM::real(0.0);
};

// CVVM RKSI_3D/RETA_3D deformation stencils for flat RLL geometry.
// Output components have physical vorticity-tendency units.
//
// The planetary result contains only the horizontal-vorticity vertical-
// shear contribution. It is not a complete three-component Coriolis
// operator. In particular, it does not implement planetary transport or
// the top-zeta planetary contribution.
//
// These terms must be combined with a consistent canonical-component
// transport discretization. They do not enable an RLL model path.
struct RegularLatLonHorizontalDeformationDeviceView {
    Core::Geometry::GeometryField2D h1_at_u;
    Core::Geometry::GeometryField2D h2_at_v;
    Core::Geometry::GeometryField2D h1_at_v;
    Core::Geometry::GeometryField2D h2_at_u;

    VVM::Real dq1 = VVM::real(0.0);
    VVM::Real dq2 = VVM::real(0.0);

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION VVM::Real
    u1(const Fields& fields, int k, int j, int i) const noexcept {
        return fields.u(k, j, i) / h1_at_u(j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION VVM::Real
    u2(const Fields& fields, int k, int j, int i) const noexcept {
        return fields.v(k, j, i) / h2_at_v(j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION VVM::Real
    omega1_over_rho(const Fields& fields, int k, int j, int i) const noexcept {
        return fields.xi(k, j, i) / h1_at_v(j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION VVM::Real
    omega2_over_rho(const Fields& fields, int k, int j, int i) const noexcept {
        return -fields.eta(k, j, i) / h2_at_u(j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION HorizontalDeformationTerms
    calculate_xi_at_v(const Fields& fields, int k, int j, int i) const noexcept {

        VVM::Real stretching = VVM::real(0.0);
        VVM::Real cross = VVM::real(0.0);
        VVM::Real vertical = VVM::real(0.0);
        VVM::Real planetary = VVM::real(0.0);

        for (int jj = j; jj <= j + 1; ++jj) {
            stretching +=
                (omega1_over_rho(fields, k, jj, i) + omega1_over_rho(fields, k, jj - 1, i)) *
                (fields.fn1(k) * (u1(fields, k + 1, jj, i) - u1(fields, k + 1, jj, i - 1)) +
                    fields.fn2(k) * (u1(fields, k, jj, i) - u1(fields, k, jj, i - 1)));
        }

        for (int ii = i - 1; ii <= i; ++ii) {
            cross += (omega2_over_rho(fields, k, j + 1, ii) + omega2_over_rho(fields, k, j, ii)) *
                     (fields.fn2(k) * (u1(fields, k, j + 1, ii) - u1(fields, k, j, ii)) +
                         fields.fn1(k) * (u1(fields, k + 1, j + 1, ii) - u1(fields, k + 1, j, ii)));

            const VVM::Real wind_difference = u1(fields, k + 1, j, ii) - u1(fields, k, j, ii) +
                                              u1(fields, k + 1, j + 1, ii) -
                                              u1(fields, k, j + 1, ii);

            vertical += wind_difference * (fields.zeta(k, j, ii) + fields.zeta(k + 1, j, ii));

            planetary += wind_difference * (fields.f_at_z(j, ii) / fields.rho(k) +
                                               fields.f_at_z(j, ii) / fields.rho(k + 1));
        }

        const VVM::Real scale = h1_at_v(j, i) * VVM::real(0.125);
        const VVM::Real vertical_factor = fields.rho_up(k) * fields.inverse_spacing(k);

        HorizontalDeformationTerms result;
        result.stretching = scale * stretching / dq1;
        result.twisting = scale * (cross / dq2 + vertical_factor * vertical);
        result.planetary = scale * vertical_factor * planetary;
        return result;
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION HorizontalDeformationTerms
    calculate_eta_at_u(const Fields& fields, int k, int j, int i) const noexcept {

        VVM::Real stretching = VVM::real(0.0);
        VVM::Real cross = VVM::real(0.0);
        VVM::Real vertical = VVM::real(0.0);
        VVM::Real planetary = VVM::real(0.0);

        for (int ii = i; ii <= i + 1; ++ii) {
            stretching +=
                (omega2_over_rho(fields, k, j, ii - 1) + omega2_over_rho(fields, k, j, ii)) *
                (fields.fn1(k) * (u2(fields, k + 1, j, ii) - u2(fields, k + 1, j - 1, ii)) +
                    fields.fn2(k) * (u2(fields, k, j, ii) - u2(fields, k, j - 1, ii)));
        }

        for (int jj = j - 1; jj <= j; ++jj) {
            cross += (omega1_over_rho(fields, k, jj, i + 1) + omega1_over_rho(fields, k, jj, i)) *
                     (fields.fn2(k) * (u2(fields, k, jj, i + 1) - u2(fields, k, jj, i)) +
                         fields.fn1(k) * (u2(fields, k + 1, jj, i + 1) - u2(fields, k + 1, jj, i)));

            const VVM::Real wind_difference = u2(fields, k + 1, jj, i) - u2(fields, k, jj, i) +
                                              u2(fields, k + 1, jj, i + 1) -
                                              u2(fields, k, jj, i + 1);

            vertical += wind_difference * (fields.zeta(k, jj, i) + fields.zeta(k + 1, jj, i));

            planetary += wind_difference * (fields.f_at_z(jj, i) / fields.rho(k) +
                                               fields.f_at_z(jj, i) / fields.rho(k + 1));
        }

        // Convert canonical omega2 tendency to legacy-sign physical eta.
        const VVM::Real scale = -h2_at_u(j, i) * VVM::real(0.125);
        const VVM::Real vertical_factor = fields.rho_up(k) * fields.inverse_spacing(k);

        HorizontalDeformationTerms result;
        result.stretching = scale * stretching / dq2;
        result.twisting = scale * (cross / dq1 + vertical_factor * vertical);
        result.planetary = scale * vertical_factor * planetary;
        return result;
    }
};

inline RegularLatLonHorizontalDeformationDeviceView
make_regular_lat_lon_horizontal_deformation_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    using Core::Geometry::GeometryKind;
    using Core::Geometry::HorizontalLocation;

    if (geometry.kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument(
            "RegularLatLonHorizontalDeformation requires regular latitude-longitude geometry.");
    }

    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);

    RegularLatLonHorizontalDeformationDeviceView result;
    result.h1_at_u = u.contravariant_to_physical.a11;
    result.h2_at_v = v.contravariant_to_physical.a22;
    result.h1_at_v = v.contravariant_to_physical.a11;
    result.h2_at_u = u.contravariant_to_physical.a22;
    result.dq1 = geometry.dq1();
    result.dq2 = geometry.dq2();
    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_DEFORMATION_HPP
