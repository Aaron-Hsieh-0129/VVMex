#ifndef VVM_DYNAMICS_SOLVERS_REGULAR_LAT_LON_ELLIPTIC_METRICS_HPP
#define VVM_DYNAMICS_SOLVERS_REGULAR_LAT_LON_ELLIPTIC_METRICS_HPP

#include <cstddef>
#include <stdexcept>
#include <string>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM {
namespace Dynamics {

// Metric handles needed by the regular latitude-longitude elliptic kernel.
// Every field varies only with j. These views share the existing geometry
// allocations; constructing this object does not allocate or copy metric data.
//
// Keep T/U/V/Z names explicit even where two locations share latitude arrays.
// The Grid and solver must outlive every CUDA graph that uses these views.
struct RegularLatLonEllipticMetrics {
    using View1D = Core::Geometry::GeometryConstView1D;

    View1D sqrt_g_at_t;
    View1D sqrt_g_at_z;
    View1D sqrt_g_at_u;
    View1D sqrt_g_at_v;

    View1D inv_sqrt_g_at_u;
    View1D inv_sqrt_g_at_v;

    View1D sqrt_g_g_contra_11_at_u;
    View1D sqrt_g_g_contra_22_at_u;
    View1D sqrt_g_g_contra_11_at_v;
    View1D sqrt_g_g_contra_22_at_v;

    VVM::Real dq1 = VVM::real(0.0);
    VVM::Real dq2 = VVM::real(0.0);
    VVM::Real inverse_dq1_squared = VVM::real(0.0);
    VVM::Real inverse_dq2_squared = VVM::real(0.0);
};

// Host-side adapter. Call during solver construction, before graph capture.
// Reject a changed storage layout explicitly instead of silently reading an
// empty 1-D view if the geometry implementation changes later.
inline RegularLatLonEllipticMetrics make_regular_lat_lon_elliptic_metrics(
    const Core::Geometry::HorizontalGeometry& geometry) {

    using Core::Geometry::GeometryField2D;
    using Core::Geometry::GeometryFieldLayout;
    using Core::Geometry::GeometryKind;
    using Core::Geometry::HorizontalLocation;

    if (geometry.kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument("RegularLatLonEllipticMetrics requires regular latitude-longitude geometry.");
    }

    const auto t = geometry.device_view(HorizontalLocation::T);
    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);
    const auto z = geometry.device_view(HorizontalLocation::Z);
    const std::size_t ny = static_cast<std::size_t>(geometry.layout().local_total_ny());

    const auto latitude_view = [ny](const GeometryField2D& field, const char* name) {
        if (field.layout != GeometryFieldLayout::VaryingJ ||
            field.one_dimensional.extent(0) != ny ||
            field.one_dimensional.data() == nullptr) {
            throw std::invalid_argument(std::string("RegularLatLonEllipticMetrics requires a latitude array for ") + name);
        }

        return field.one_dimensional;
    };

    RegularLatLonEllipticMetrics result;

    result.sqrt_g_at_t = latitude_view(t.sqrt_g, "sqrt_g_at_t");
    result.sqrt_g_at_z = latitude_view(z.sqrt_g, "sqrt_g_at_z");
    result.sqrt_g_at_u = latitude_view(u.sqrt_g, "sqrt_g_at_u");
    result.sqrt_g_at_v = latitude_view(v.sqrt_g, "sqrt_g_at_v");

    result.inv_sqrt_g_at_u = latitude_view(u.inv_sqrt_g, "inv_sqrt_g_at_u");
    result.inv_sqrt_g_at_v = latitude_view(v.inv_sqrt_g, "inv_sqrt_g_at_v");

    result.sqrt_g_g_contra_11_at_u = latitude_view(u.sqrt_g_g_contra.a11, "sqrt_g_g_contra_11_at_u");
    result.sqrt_g_g_contra_22_at_u = latitude_view(u.sqrt_g_g_contra.a22, "sqrt_g_g_contra_22_at_u");
    result.sqrt_g_g_contra_11_at_v = latitude_view(v.sqrt_g_g_contra.a11, "sqrt_g_g_contra_11_at_v");
    result.sqrt_g_g_contra_22_at_v = latitude_view(v.sqrt_g_g_contra.a22, "sqrt_g_g_contra_22_at_v");

    result.dq1 = t.dq1;
    result.dq2 = t.dq2;
    result.inverse_dq1_squared = VVM::real(1.0) / (result.dq1 * result.dq1);
    result.inverse_dq2_squared = VVM::real(1.0) / (result.dq2 * result.dq2);

    return result;
}

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_SOLVERS_REGULAR_LAT_LON_ELLIPTIC_METRICS_HPP
