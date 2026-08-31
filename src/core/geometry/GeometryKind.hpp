#ifndef VVM_CORE_GEOMETRY_GEOMETRY_KIND_HPP
#define VVM_CORE_GEOMETRY_GEOMETRY_KIND_HPP

#include <cstdint>

namespace VVM {
namespace Core {
namespace Geometry {

enum class GeometryKind : std::uint8_t {
    Cartesian,
    RegularLatLon,
    CubedSphere
};

inline const char* geometry_kind_to_string(
    const GeometryKind kind) noexcept {

    switch (kind) {
        case GeometryKind::Cartesian:
            return "cartesian";

        case GeometryKind::RegularLatLon:
            return "regular_latlon";

        case GeometryKind::CubedSphere:
            return "cubed_sphere";
    }

    return "unknown";
}

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_GEOMETRY_KIND_HPP
