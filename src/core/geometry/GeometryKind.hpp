#ifndef VVM_CORE_GEOMETRY_GEOMETRY_KIND_HPP
#define VVM_CORE_GEOMETRY_GEOMETRY_KIND_HPP

#include <cstdint>

namespace VVM {
namespace Core {
namespace Geometry {

enum class GeometryKind : std::uint8_t { Cartesian, RegularLatLon, Affine, CubedSphere };

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_GEOMETRY_KIND_HPP
