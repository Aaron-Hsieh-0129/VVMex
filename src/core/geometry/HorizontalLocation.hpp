#ifndef VVM_CORE_GEOMETRY_HORIZONTAL_LOCATION_HPP
#define VVM_CORE_GEOMETRY_HORIZONTAL_LOCATION_HPP

#include <cstdint>

namespace VVM {
namespace Core {
namespace Geometry {

// Horizontal projection of the full VVMex three-dimensional staggering.
// T: horizontally centred
// U: staggered in q1
// V: staggered in q2
// Z: staggered in both q1 and q2
// "Z" follows the VVM/CVVM convention for the vertical-vorticity
// location. It does not mean vertical staggering.
enum class HorizontalLocation : std::uint8_t {
    T,  // (i,     j)
    U,  // (i+1/2, j)
    V,  // (i,     j+1/2)
    Z   // (i+1/2, j+1/2)
};

inline const char* horizontal_location_to_string(
    const HorizontalLocation location) noexcept {

    switch (location) {
        case HorizontalLocation::T:
            return "T";
        case HorizontalLocation::U:
            return "U";
        case HorizontalLocation::V:
            return "V";
        case HorizontalLocation::Z:
            return "Z";
    }

    return "unknown";
}

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_HORIZONTAL_LOCATION_HPP
