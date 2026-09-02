#ifndef VVM_CORE_GEOMETRY_HORIZONTAL_GEOMETRY_FACTORY_HPP
#define VVM_CORE_GEOMETRY_HORIZONTAL_GEOMETRY_FACTORY_HPP

#include <memory>

#include "core/geometry/HorizontalGeometry.hpp"
#include "core/geometry/HorizontalGridSpec.hpp"

namespace VVM {
namespace Core {
namespace Geometry {

class HorizontalGeometryFactory {
public:
    static std::unique_ptr<HorizontalGeometry> create(const HorizontalGridSpec& spec, HorizontalDomainLayout layout);
};

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_HORIZONTAL_GEOMETRY_FACTORY_HPP
