#ifndef VVM_CORE_GRID_STAGGERING_HPP
#define VVM_CORE_GRID_STAGGERING_HPP

#include <cstdint>

namespace VVM {
namespace Core {


enum class GridStaggering : std::uint8_t {
    Unspecified, NotApplicable,

    Centered,       // (i,     j,     k)

    StaggeredX,     // (i+1/2, j,     k)     : u
    StaggeredY,     // (i,     j+1/2, k)     : v
    StaggeredZ,     // (i,     j,     k+1/2) : w

    StaggeredYZ,    // (i,     j+1/2, k+1/2) : xi
    StaggeredXZ,    // (i+1/2, j,     k+1/2) : eta
    StaggeredXY,    // (i+1/2, j+1/2, k)     : zeta

    StaggeredXYZ,   // (i+1/2, j+1/2, k+1/2)
    Surface         // At surface
};

inline const char* grid_staggering_to_string(
    const GridStaggering staggering) noexcept {

    switch (staggering) {
        case GridStaggering::Unspecified:
            return "unspecified";
        case GridStaggering::NotApplicable:
            return "not_applicable";
        case GridStaggering::Centered:
            return "centered";
        case GridStaggering::StaggeredX:
            return "staggered_x";
        case GridStaggering::StaggeredY:
            return "staggered_y";
        case GridStaggering::StaggeredZ:
            return "staggered_z";
        case GridStaggering::StaggeredYZ:
            return "staggered_yz";
        case GridStaggering::StaggeredXZ:
            return "staggered_xz";
        case GridStaggering::StaggeredXY:
            return "staggered_xy";
        case GridStaggering::StaggeredXYZ:
            return "staggered_xyz";
        case GridStaggering::Surface:
            return "surface";
    }

    return "unknown";
}

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GRID_STAGGERING_HPP
