#ifndef VVM_CORE_GEOMETRY_HORIZONTAL_STAGGERING_HPP
#define VVM_CORE_GEOMETRY_HORIZONTAL_STAGGERING_HPP

#include <optional>
#include <stdexcept>
#include <string>

#include "core/GridStaggering.hpp"
#include "core/geometry/HorizontalLocation.hpp"

namespace VVM {
namespace Core {
namespace Geometry {

// Return the horizontal projection of a complete GridStaggering.
// Vertical staggering is deliberately ignored:
//   Centered       -> T
//   StaggeredZ     -> T
//   StaggeredX     -> U
//   StaggeredXZ    -> U
//   StaggeredY     -> V
//   StaggeredYZ    -> V
//   StaggeredXY    -> Z
//   StaggeredXYZ   -> Z
// Surface is assumed to mean a horizontally centred surface field.
//
// Unspecified and NotApplicable have no valid horizontal geometry location.

constexpr std::optional<HorizontalLocation>
try_horizontal_location(const GridStaggering staggering) noexcept {

    switch (staggering) {
        case GridStaggering::Centered:
        case GridStaggering::StaggeredZ:
        case GridStaggering::Surface:
            return HorizontalLocation::T;

        case GridStaggering::StaggeredX:
        case GridStaggering::StaggeredXZ:
            return HorizontalLocation::U;

        case GridStaggering::StaggeredY:
        case GridStaggering::StaggeredYZ:
            return HorizontalLocation::V;

        case GridStaggering::StaggeredXY:
        case GridStaggering::StaggeredXYZ:
            return HorizontalLocation::Z;

        case GridStaggering::Unspecified:
        case GridStaggering::NotApplicable:
            return std::nullopt;
    }

    return std::nullopt;
}

inline HorizontalLocation horizontal_location_or_throw(const GridStaggering staggering, const std::string& field_name = {}) {
    const auto location = try_horizontal_location(staggering);

    if (location.has_value()) return *location;

    std::string message = "No horizontal geometry location exists for staggering '" +
                          std::string(grid_staggering_to_string(staggering)) + "'";

    if (!field_name.empty()) message += " on field '" + field_name + "'";

    message += ".";

    throw std::invalid_argument(message);
}

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_HORIZONTAL_STAGGERING_HPP
