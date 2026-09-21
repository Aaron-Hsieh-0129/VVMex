#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_DEFORMATION_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_HORIZONTAL_DEFORMATION_HPP

#include "dynamics/operators/GeneralizedHorizontalDeformation.hpp"

namespace VVM::Dynamics::Operators {

// Legacy RLL field contract: physical u/v; xi/eta already divided by
// rho_up; zeta already divided by rho; f_at_z is NOT density-normalized.
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

// Temporary representation adapter only. The production deformation path
// bypasses it. All numerical stencils live in the generalized operator.
struct RegularLatLonHorizontalDeformationDeviceView {
    using Base = GeneralizedHorizontalDeformationDeviceView;

    Core::Geometry::GeometryField2D h1_at_u;
    Core::Geometry::GeometryField2D h2_at_v;
    Core::Geometry::GeometryField2D h1_at_v;
    Core::Geometry::GeometryField2D h2_at_u;

    Real dq1 = real(0.0);
    Real dq2 = real(0.0);

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    u1(const Fields& fields, int k, int j, int i) const noexcept {
        return fields.u(k, j, i) / h1_at_u(j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    u2(const Fields& fields, int k, int j, int i) const noexcept {
        return fields.v(k, j, i) / h2_at_v(j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    omega1_over_rho(const Fields& fields, int k, int j, int i) const noexcept {
        return fields.xi(k, j, i) / h1_at_v(j, i);
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    omega2_over_rho(const Fields& fields, int k, int j, int i) const noexcept {
        return -fields.eta(k, j, i) / h2_at_u(j, i);
    }

    // Retained legacy accessor; the deformation adapter below deliberately
    // reads fields.zeta directly because its contract is already normalized.
    template <typename Fields>
    KOKKOS_INLINE_FUNCTION Real
    zeta_over_rho(const Fields& fields, int k, int j, int i) const noexcept {
        return fields.zeta(k, j, i) / fields.rho(k);
    }

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

    KOKKOS_INLINE_FUNCTION static HorizontalDeformationTerms
    to_physical(HorizontalDeformationTerms result, Real scale) noexcept {
        result.stretching *= scale;
        result.twisting *= scale;
        result.planetary *= scale;
        return result;
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION HorizontalDeformationTerms
    calculate_xi_at_v(const Fields& fields, int k, int j, int i) const noexcept {
        return to_physical(
            Base{dq1, dq2}.calculate_omega1_at_v(LegacyFields<Fields>{fields, *this}, k, j, i),
            h1_at_v(j, i));
    }

    template <typename Fields>
    KOKKOS_INLINE_FUNCTION HorizontalDeformationTerms
    calculate_eta_at_u(const Fields& fields, int k, int j, int i) const noexcept {
        return to_physical(
            Base{dq1, dq2}.calculate_omega2_at_u(LegacyFields<Fields>{fields, *this}, k, j, i),
            -h2_at_u(j, i));
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
    const auto numerical = make_generalized_horizontal_deformation_device_view(geometry);

    RegularLatLonHorizontalDeformationDeviceView result;
    result.dq1 = numerical.dq1;
    result.dq2 = numerical.dq2;
    result.h1_at_u = u.contravariant_to_physical.a11;
    result.h2_at_v = v.contravariant_to_physical.a22;
    result.h1_at_v = v.contravariant_to_physical.a11;
    result.h2_at_u = u.contravariant_to_physical.a22;
    return result;
}

} // namespace VVM::Dynamics::Operators
#endif
