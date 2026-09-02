#include "core/geometry/HorizontalGeometryFactory.hpp"

#include <memory>
#include <stdexcept>

#include "core/geometry/CartesianGeometry.hpp"

namespace VVM {
namespace Core {
namespace Geometry {

std::unique_ptr<HorizontalGeometry> HorizontalGeometryFactory::create(
    const HorizontalGridSpec& spec, HorizontalDomainLayout layout) {

    switch (spec.kind) {
        case GeometryKind::Cartesian:
            return std::make_unique<CartesianGeometry>(layout, spec.dq1, spec.dq2);

        case GeometryKind::RegularLatLon:
            throw std::invalid_argument("Regular latitude-longitude geometry is not implemented yet.");

        case GeometryKind::CubedSphere:
            throw std::invalid_argument("Cubed-sphere geometry is not implemented.");
    }

    throw std::invalid_argument("HorizontalGeometryFactory received an invalid GeometryKind.");
}

} // namespace Geometry
} // namespace Core
} // namespace VVM
