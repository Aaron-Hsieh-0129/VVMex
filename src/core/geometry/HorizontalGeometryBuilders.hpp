#ifndef VVM_CORE_GEOMETRY_HORIZONTAL_GEOMETRY_BUILDERS_HPP
#define VVM_CORE_GEOMETRY_HORIZONTAL_GEOMETRY_BUILDERS_HPP

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM::Core::Geometry {

// These only generate inputs. All return the same metric-driven geometry.
HorizontalGeometry make_cartesian_geometry(HorizontalDomainLayout layout, Real dx, Real dy);

HorizontalGeometry make_regular_lat_lon_geometry(HorizontalDomainLayout layout,
    Real dlongitude,
    Real dlatitude,
    Real longitude_west_edge,
    Real latitude_south_edge,
    Real radius,
    bool periodic_latitude = false);

HorizontalGeometry make_affine_geometry(HorizontalDomainLayout layout,
    Real dq1,
    Real dq2,
    const std::array<Real, 4>& basis,
    Real longitude = real(0.0),
    Real latitude = real(0.0));

// Common layout/spacing checks, also used before builders allocate fields.
void validate_horizontal_geometry_layout(const HorizontalDomainLayout& layout, Real dq1, Real dq2);

} // namespace VVM::Core::Geometry
#endif
