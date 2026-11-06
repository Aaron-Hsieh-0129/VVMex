#include "core/geometry/HorizontalGeometryBuilders.hpp"

#include <cmath>
#include <stdexcept>

namespace VVM::Core::Geometry {
namespace {

GeometryField2D
constant(Real value) {
    return GeometryField2D::constant_value(value);
}

GeometryField2D
coordinate(const HorizontalDomainLayout& layout,
    bool first_axis,
    Real edge,
    Real offset,
    Real spacing,
    int periodic_size,
    const std::string& label) {
    const int size = first_axis ? layout.local_total_nx() : layout.local_total_ny();
    const int start = first_axis ? layout.global_start_i : layout.global_start_j;
    const int halo = layout.halo;
    Kokkos::View<Real*> values(label, size);
    Kokkos::parallel_for(label, size, KOKKOS_LAMBDA(const int index) {
        int global = start + index - halo;
        if (periodic_size > 0) {
            global = (global % periodic_size + periodic_size) % periodic_size;
        }
        values(index) = edge + (static_cast<Real>(global) + offset) * spacing;
    });
    return first_axis ? GeometryField2D::varying_i(values) : GeometryField2D::varying_j(values);
}

void
validate_patch(const HorizontalDomainLayout& layout, Real dq1, Real dq2) {
    validate_horizontal_geometry_layout(layout, dq1, dq2);
    if (layout.panel_id != -1) {
        throw std::invalid_argument(
            "Planar and regular latitude-longitude builders require panel_id == -1.");
    }
}

std::array<HorizontalGeometryInput, 4>
coordinate_inputs(const HorizontalDomainLayout& layout,
    Real dq1,
    Real dq2,
    Real west,
    Real south,
    Real offset,
    bool periodic_latitude,
    const std::string& label) {
    const auto q1_centered = coordinate(layout, true, west, offset, dq1, 0, label + "_q1_centered");
    const auto q1_staggered =
        coordinate(layout, true, west, offset + real(.5), dq1, 0, label + "_q1_staggered");
    const auto q2_centered = coordinate(layout,
        false,
        south,
        offset,
        dq2,
        periodic_latitude ? layout.global_ny : 0,
        label + "_q2_centered");
    const auto q2_staggered = coordinate(layout,
        false,
        south,
        offset + real(.5),
        dq2,
        periodic_latitude ? layout.global_ny : 0,
        label + "_q2_staggered");
    std::array<HorizontalGeometryInput, 4> inputs;
    for (int l = 0; l < 4; ++l) {
        inputs[l].q1 = (l == 1 || l == 3) ? q1_staggered : q1_centered;
        inputs[l].q2 = (l == 2 || l == 3) ? q2_staggered : q2_centered;
    }
    return inputs;
}

HorizontalGeometry
constant_mapping_geometry(HorizontalDomainLayout layout,
    Real dq1,
    Real dq2,
    const std::array<Real, 4>& b,
    Real longitude,
    Real latitude,
    GeometryKind kind,
    const std::string& name) {
    validate_patch(layout, dq1, dq2);
    if (!std::isfinite(longitude) || !std::isfinite(latitude) ||
        std::abs(latitude) > std::acos(real(-1)) / real(2)) {
        throw std::invalid_argument(
            "Reference longitude/latitude must be finite with latitude within the poles.");
    }
    const Real g11 = b[0] * b[0] + b[2] * b[2];
    const Real g12 = b[0] * b[1] + b[2] * b[3];
    const Real g22 = b[1] * b[1] + b[3] * b[3];
    auto inputs = coordinate_inputs(layout, dq1, dq2, real(0), real(0), real(0), false, name);
    for (auto& input : inputs) {
        input.g_cov = {constant(g11), constant(g12), constant(g12), constant(g22)};
        input.longitude = constant(longitude);
        input.latitude = constant(latitude);
        input.contravariant_to_physical = {constant(b[0]),
            constant(b[1]),
            constant(b[2]),
            constant(b[3])};
    }
    Kokkos::fence();
    return HorizontalGeometry(layout, dq1, dq2, inputs, kind, name);
}

} // namespace

HorizontalGeometry
make_cartesian_geometry(HorizontalDomainLayout layout, Real dx, Real dy) {
    return constant_mapping_geometry(layout,
        dx,
        dy,
        {real(1), real(0), real(0), real(1)},
        real(0),
        real(0),
        GeometryKind::Cartesian,
        "cartesian");
}

HorizontalGeometry
make_affine_geometry(HorizontalDomainLayout layout,
    Real dq1,
    Real dq2,
    const std::array<Real, 4>& b,
    Real longitude,
    Real latitude) {
    for (const Real value : b) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Affine basis entries must be finite.");
        }
    }
    const Real determinant = b[0] * b[3] - b[1] * b[2];
    if (!std::isfinite(determinant) || determinant <= real(0)) {
        throw std::invalid_argument("Affine basis must be nonsingular and positively oriented.");
    }
    return constant_mapping_geometry(layout,
        dq1,
        dq2,
        b,
        longitude,
        latitude,
        GeometryKind::Affine,
        "affine");
}

HorizontalGeometry
make_regular_lat_lon_geometry(HorizontalDomainLayout layout,
    Real dlongitude,
    Real dlatitude,
    Real west,
    Real south,
    Real radius,
    bool periodic_latitude) {
    validate_patch(layout, dlongitude, dlatitude);
    if (!std::isfinite(west) || !std::isfinite(south) || !std::isfinite(radius) ||
        radius <= real(0)) {
        throw std::invalid_argument(
            "Regular latitude-longitude requires finite edges and positive finite radius.");
    }
    const Real half_pi = std::acos(real(-1)) / real(2);
    const Real minimum = south + (real(.5) - static_cast<Real>(layout.halo)) * dlatitude;
    const Real maximum =
        south + (static_cast<Real>(layout.global_ny) + static_cast<Real>(layout.halo)) * dlatitude;
    if (minimum <= -half_pi || maximum >= half_pi) {
        throw std::invalid_argument("Regular latitude-longitude requires every T/U/V/Z latitude, "
                                    "including halos, to remain strictly between the poles.");
    }
    auto inputs = coordinate_inputs(layout,
        dlongitude,
        dlatitude,
        west,
        south,
        real(.5),
        periodic_latitude,
        "regular_latlon");
    // The chart supplies its metric and physical basis. Every dependent
    // coefficient is constructed by HorizontalGeometry, just as for a skew g.
    for (const int l : {0, 2}) {
        const auto latitude = inputs[l].q2;
        Kokkos::View<Real*> g11("regular_latlon_g11", layout.local_total_ny());
        Kokkos::View<Real*> basis11("regular_latlon_basis11", layout.local_total_ny());
        Kokkos::parallel_for("InitializeRegularLatLonCovariantMetric",
            layout.local_total_ny(),
            KOKKOS_LAMBDA(const int j) {
                const Real scale = radius * Kokkos::cos(latitude(j, 0));
                g11(j) = scale * scale;
                basis11(j) = scale;
            });
        const Matrix2DeviceView g{GeometryField2D::varying_j(g11),
            constant(real(0)),
            constant(real(0)),
            constant(radius * radius)};
        inputs[l].g_cov = g;
        inputs[l + 1].g_cov = g;
        const Matrix2DeviceView basis{GeometryField2D::varying_j(basis11),
            constant(real(0)),
            constant(real(0)),
            constant(radius)};
        inputs[l].contravariant_to_physical = basis;
        inputs[l + 1].contravariant_to_physical = basis;
    }
    for (auto& input : inputs) {
        // Longitude halos stay unwrapped. Experimental periodic latitude
        // wraps native row indices and describes a repeating patch.
        input.longitude = input.q1;
        input.latitude = input.q2;
    }
    Kokkos::fence();
    return HorizontalGeometry(layout,
        dlongitude,
        dlatitude,
        inputs,
        GeometryKind::RegularLatLon,
        "regular_latlon");
}

} // namespace VVM::Core::Geometry
