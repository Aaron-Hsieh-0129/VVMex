#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_TERRAIN_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_TERRAIN_HPP
#include "core/geometry/StaggeredVectorTransform.hpp"

namespace VVM::Dynamics::Operators {

// Curl of masked physical winds in a stationary horizontal chart. Outputs
// are canonical omega^1@V and -omega^2@U; height remains physical.
struct GeneralizedTerrainDeviceView {
    Core::Geometry::GeometryField2D b11_u, b12_u, b21_v, b22_v;
    Core::Geometry::GeometryField2D inv_j_u, inv_j_v;
    Real rdq1, rdq2;

    template <class U, class V>
    KOKKOS_INLINE_FUNCTION Real
    covariant_q1(const U& u, const V& v, int k, int j, int i) const {
        return b11_u(j, i) * u(k, j, i) + Core::Geometry::weighted_cross<true>(v, b21_v, k, j, i);
    }
    template <class U, class V>
    KOKKOS_INLINE_FUNCTION Real
    covariant_q2(const U& u, const V& v, int k, int j, int i) const {
        return b22_v(j, i) * v(k, j, i) + Core::Geometry::weighted_cross<false>(u, b12_u, k, j, i);
    }
    template <class U, class V, class W>
    KOKKOS_INLINE_FUNCTION Real
    xi(const U& u, const V& v, const W& w, Real rdz, int k, int j, int i) const {
        return inv_j_v(j, i) *
               ((w(k, j + 1, i) - w(k, j, i)) * rdq2 -
                   (covariant_q2(u, v, k + 1, j, i) - covariant_q2(u, v, k, j, i)) * rdz);
    }
    template <class U, class V, class W>
    KOKKOS_INLINE_FUNCTION Real
    eta(const U& u, const V& v, const W& w, Real rdz, int k, int j, int i) const {
        return inv_j_u(j, i) *
               ((w(k, j, i + 1) - w(k, j, i)) * rdq1 -
                   (covariant_q1(u, v, k + 1, j, i) - covariant_q1(u, v, k, j, i)) * rdz);
    }
};

inline GeneralizedTerrainDeviceView
make_generalized_terrain_device_view(const Core::Geometry::HorizontalGeometry& geometry) {
    using Core::Geometry::HorizontalLocation;
    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);
    return {u.contravariant_to_physical.a11,
        u.contravariant_to_physical.a12,
        v.contravariant_to_physical.a21,
        v.contravariant_to_physical.a22,
        u.inv_sqrt_g,
        v.inv_sqrt_g,
        real(1.0) / geometry.dq1(),
        real(1.0) / geometry.dq2()};
}
}
#endif
