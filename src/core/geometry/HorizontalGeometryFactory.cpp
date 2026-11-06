#include "core/geometry/HorizontalGeometryFactory.hpp"
#include "core/geometry/HorizontalGeometryBuilders.hpp"

#include <memory>
#include <stdexcept>

namespace VVM::Core::Geometry {

std::unique_ptr<HorizontalGeometry>
HorizontalGeometryFactory::create(const HorizontalGridSpec& spec, HorizontalDomainLayout layout) {
    switch (spec.kind) {
    case GeometryKind::Cartesian:
        return std::make_unique<HorizontalGeometry>(
            make_cartesian_geometry(layout, spec.dq1, spec.dq2));
    case GeometryKind::RegularLatLon:
        return std::make_unique<HorizontalGeometry>(make_regular_lat_lon_geometry(layout,
            spec.dq1,
            spec.dq2,
            spec.regular_lat_lon.longitude_west_edge,
            spec.regular_lat_lon.latitude_south_edge,
            spec.regular_lat_lon.radius,
            spec.regular_lat_lon.periodic_latitude));
    case GeometryKind::Affine:
        return std::make_unique<HorizontalGeometry>(make_affine_geometry(layout,
            spec.dq1,
            spec.dq2,
            spec.affine_basis,
            spec.reference_longitude,
            spec.reference_latitude));
    case GeometryKind::CubedSphere:
        throw std::invalid_argument("Cubed-sphere geometry is not implemented.");
    }
    throw std::invalid_argument("HorizontalGeometryFactory received an invalid GeometryKind.");
}

} // namespace VVM::Core::Geometry
