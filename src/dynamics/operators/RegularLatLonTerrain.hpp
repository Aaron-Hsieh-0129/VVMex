#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TERRAIN_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_TERRAIN_HPP

#include "core/geometry/HorizontalGeometry.hpp"
#include <stdexcept>

namespace VVM::Dynamics::Operators {

// Original VVMex masked-wind curl, expressed in physical RLL components.
// There is no density normalization here. xi is eastward at V; eta is minus
// northward at U. Geometry has no vertical dependence (shallow atmosphere).
struct RegularLatLonTerrainDeviceView {
    Core::Geometry::GeometryField2D inverse_h1_u, inverse_h2_v;
    Real inverse_dq1, inverse_dq2;

    template <class Volume>
    KOKKOS_INLINE_FUNCTION Real
    xi(const Volume& v, const Volume& w, Real inverse_dz, int k, int j, int i) const {
        return (w(k, j + 1, i) - w(k, j, i)) * inverse_dq2 * inverse_h2_v(j, i) -
               (v(k + 1, j, i) - v(k, j, i)) * inverse_dz;
    }

    template <class Volume>
    KOKKOS_INLINE_FUNCTION Real
    eta(const Volume& u, const Volume& w, Real inverse_dz, int k, int j, int i) const {
        return (w(k, j, i + 1) - w(k, j, i)) * inverse_dq1 * inverse_h1_u(j, i) -
               (u(k + 1, j, i) - u(k, j, i)) * inverse_dz;
    }
};

inline RegularLatLonTerrainDeviceView
make_regular_lat_lon_terrain_device_view(const Core::Geometry::HorizontalGeometry& geometry) {
    using namespace Core::Geometry;
    if (geometry.kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument(
            "RLL terrain curl requires regular latitude-longitude geometry.");
    }
    return {geometry.device_view(HorizontalLocation::U).physical_to_contravariant.a11,
        geometry.device_view(HorizontalLocation::V).physical_to_contravariant.a22,
        real(1.) / geometry.dq1(),
        real(1.) / geometry.dq2()};
}
} // namespace VVM::Dynamics::Operators
#endif
