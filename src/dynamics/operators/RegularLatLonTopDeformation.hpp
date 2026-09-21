#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_DEFORMATION_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TOP_DEFORMATION_HPP

#include "dynamics/operators/GeneralizedTopDeformation.hpp"

namespace VVM::Dynamics::Operators {

// Legacy RLL contract: physical w; xi/eta already divided by rho_up;
// zeta already divided by rho; f_at_z is NOT density-normalized.
template <typename VolumeView, typename ProfileView, typename PlaneView>
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

// Temporary representation adapter. No independent deformation stencil.
struct RegularLatLonTopDeformationDeviceView {
    using Base = GeneralizedTopDeformationDeviceView;

    Core::Geometry::GeometryField2D h1_at_v;
    Core::Geometry::GeometryField2D h2_at_u;

    Real dq1 = real(0.0);
    Real dq2 = real(0.0);

    template <typename Fields>
    struct LegacyFields {
        const Fields& fields;
        const RegularLatLonTopDeformationDeviceView& components;

        KOKKOS_INLINE_FUNCTION Real
        w(int k, int j, int i) const noexcept {
            return fields.w(k, j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        omega1_over_rho(int k, int j, int i) const noexcept {
            return fields.xi(k, j, i) / components.h1_at_v(j, i);
        }

        KOKKOS_INLINE_FUNCTION Real
        omega2_over_rho(int k, int j, int i) const noexcept {
            return -fields.eta(k, j, i) / components.h2_at_u(j, i);
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

        KOKKOS_INLINE_FUNCTION Real
        inverse_spacing_up(int k) const noexcept {
            return fields.inverse_spacing_up(k);
        }
    };

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION TopDeformationTerms
    calculate_at_z(const Fields& fields, int k_top, int j, int i) const noexcept {
        return Base{dq1, dq2}.calculate_at_z(LegacyFields<Fields>{fields, *this}, k_top, j, i);
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

    const auto numerical = make_generalized_top_deformation_device_view(geometry);

    RegularLatLonTopDeformationDeviceView result;
    result.dq1 = numerical.dq1;
    result.dq2 = numerical.dq2;
    result.h1_at_v = geometry.device_view(HorizontalLocation::V).contravariant_to_physical.a11;
    result.h2_at_u = geometry.device_view(HorizontalLocation::U).contravariant_to_physical.a22;
    return result;
}

} // namespace VVM::Dynamics::Operators
#endif
