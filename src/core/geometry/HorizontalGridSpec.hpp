#ifndef VVM_CORE_GEOMETRY_HORIZONTAL_GRID_SPEC_HPP
#define VVM_CORE_GEOMETRY_HORIZONTAL_GRID_SPEC_HPP

#include "core/geometry/GeometryKind.hpp"
#include "core/vvm_types.hpp"

namespace VVM {
namespace Core {
namespace Geometry {

struct HorizontalGridSpec {
    GeometryKind kind = GeometryKind::Cartesian;
    VVM::Real dq1 = VVM::real(0.0);
    VVM::Real dq2 = VVM::real(0.0);
};

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_HORIZONTAL_GRID_SPEC_HPP
