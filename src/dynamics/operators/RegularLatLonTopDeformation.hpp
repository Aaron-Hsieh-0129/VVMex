#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_DEFORMATION_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_DEFORMATION_HPP

#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Borrowed views only. Volumes use (k, j, i).
//
// w:
//     Physical vertical velocity at horizontal T.
//
// xi, eta:
//     Physical eastward and legacy-sign horizontal vorticity divided by
//     rho_up, at V and U respectively.
//
// zeta:
//     Physical relative vertical vorticity divided by rho, at Z.
//
// f_at_z:
//     Planetary vertical vorticity at Z, without density normalization.
//
// inverse_spacing_mid:
//     flex_height_coef_mid / dz.
//
// inverse_spacing_up:
//     flex_height_coef_up / dz.
//
// The caller supplies a rigid lid with w(k_top) = 0 over the required
// horizontal stencil. k_top >= 1, all used extents and halos are valid,
// and used density and spacing values are positive and finite.
// Geometry and field storage must outlive queued work and graph replay.
template<typename VolumeView, typename ProfileView, typename PlaneView>
struct RegularLatLonTopDeformationFields {
    VolumeView w;
    VolumeView xi;
    VolumeView eta;
    VolumeView zeta;
    PlaneView f_at_z;
    ProfileView rho;
    ProfileView rho_up;
    ProfileView inverse_spacing_mid;
    ProfileView inverse_spacing_up;
};

struct TopDeformationTerms {
    VVM::Real stretching = VVM::real(0.0);
    VVM::Real twisting = VVM::real(0.0);
    VVM::Real planetary = VVM::real(0.0);
};

// Rigid-lid CVVM ZETA_3D deformation terms.
//
// Returns physical relative-vorticity tendency contributions at top Z.
// This component does not include transport, impose the rigid lid,
// modify fields, launch kernels, communicate or synchronize.
//
// Planetary stretching is separate from relative stretching. It must
// eventually be combined with planetary transport exactly once; this
// result alone is not a complete top-zeta Coriolis tendency.
struct RegularLatLonTopDeformationDeviceView {
    Core::Geometry::GeometryField2D h1_at_v;
    Core::Geometry::GeometryField2D h2_at_u;

    VVM::Real dq1 = VVM::real(0.0);
    VVM::Real dq2 = VVM::real(0.0);

    template<typename Fields>
    KOKKOS_INLINE_FUNCTION
    TopDeformationTerms calculate_at_z(
        const Fields& fields, int k_top, int j, int i) const noexcept {

        const int lower = k_top - 1;
        const VVM::Real left_w =
            fields.w(lower, j, i) + fields.w(lower, j + 1, i);
        const VVM::Real right_w =
            fields.w(lower, j, i + 1) + fields.w(lower, j + 1, i + 1);

        const VVM::Real relative_stretching =
            left_w * (fields.zeta(k_top, j, i - 1) + fields.zeta(k_top, j, i))
            + right_w * (fields.zeta(k_top, j, i) + fields.zeta(k_top, j, i + 1));

        // rho(k_top) cancels the density normalization of planetary
        // vorticity at this single wind level.
        const VVM::Real planetary_stretching =
            left_w * (fields.f_at_z(j, i - 1) + fields.f_at_z(j, i))
            + right_w * (fields.f_at_z(j, i) + fields.f_at_z(j, i + 1));

        const VVM::Real omega1_sum =
            fields.xi(lower, j, i) / h1_at_v(j, i)
            + fields.xi(lower, j, i + 1) / h1_at_v(j, i + 1);
        const VVM::Real omega2_sum =
            -fields.eta(lower, j, i) / h2_at_u(j, i)
            - fields.eta(lower, j + 1, i) / h2_at_u(j + 1, i);

        const VVM::Real difference_q1 =
            fields.w(lower, j + 1, i + 1) - fields.w(lower, j + 1, i)
            + fields.w(lower, j, i + 1) - fields.w(lower, j, i);
        const VVM::Real difference_q2 =
            fields.w(lower, j + 1, i + 1) - fields.w(lower, j, i + 1)
            + fields.w(lower, j + 1, i) - fields.w(lower, j, i);

        const VVM::Real inverse_mid = fields.inverse_spacing_mid(k_top);
        const VVM::Real lower_factor =
            fields.rho_up(lower) * inverse_mid / fields.inverse_spacing_up(lower);

        TopDeformationTerms result;
        result.stretching =
            -VVM::real(0.125) * fields.rho(k_top) * inverse_mid * relative_stretching;
        result.twisting =
            VVM::real(0.125) * lower_factor
            * (omega1_sum * difference_q1 / dq1 + omega2_sum * difference_q2 / dq2);
        result.planetary =
            -VVM::real(0.125) * inverse_mid * planetary_stretching;
        return result;
    }
};

inline RegularLatLonTopDeformationDeviceView
make_regular_lat_lon_top_deformation_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {

    using Core::Geometry::GeometryKind;
    using Core::Geometry::HorizontalLocation;

    if (geometry.kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument(
            "RegularLatLonTopDeformation requires regular latitude-longitude geometry.");
    }

    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);

    RegularLatLonTopDeformationDeviceView result;
    result.h1_at_v = v.contravariant_to_physical.a11;
    result.h2_at_u = u.contravariant_to_physical.a22;
    result.dq1 = geometry.dq1();
    result.dq2 = geometry.dq2();
    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_DEFORMATION_HPP
