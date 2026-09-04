#ifndef VVM_CORE_GEOMETRY_HORIZONTAL_GRID_SPEC_HPP
#define VVM_CORE_GEOMETRY_HORIZONTAL_GRID_SPEC_HPP

#include "core/geometry/GeometryKind.hpp"
#include "core/vvm_types.hpp"

namespace VVM {
namespace Core {
namespace Geometry {

struct RegularLatLonParameters {
    // Angular coordinates are in radians. Radius is in metres.
    VVM::Real longitude_west_edge = VVM::real(0.0);
    VVM::Real latitude_south_edge = VVM::real(0.0);
    VVM::Real radius = VVM::real(0.0);
};

struct HorizontalGridSpec {
    GeometryKind kind = GeometryKind::Cartesian;

    // Cartesian: dq1 and dq2 are dx and dy in metres.
    // Regular latitude-longitude: dq1 and dq2 are longitude and latitude increments in radians.
    VVM::Real dq1 = VVM::real(0.0);
    VVM::Real dq2 = VVM::real(0.0);

    RegularLatLonParameters regular_lat_lon;
};

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_HORIZONTAL_GRID_SPEC_HPP
