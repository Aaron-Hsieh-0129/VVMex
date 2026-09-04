#include "core/geometry/RegularLatLonGeometry.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace VVM {
namespace Core {
namespace Geometry {

namespace {

struct InitializeRegularLatLonCoordinate {
    Kokkos::View<VVM::Real*> coordinate;
    int global_start;
    int halo;
    VVM::Real domain_edge;
    VVM::Real point_offset;
    VVM::Real spacing;

    InitializeRegularLatLonCoordinate(const Kokkos::View<VVM::Real*>& coordinate_in,
        const int global_start_in, const int halo_in,
        const VVM::Real domain_edge_in, const VVM::Real point_offset_in, const VVM::Real spacing_in)
        : coordinate(coordinate_in),
          global_start(global_start_in),
          halo(halo_in),
          domain_edge(domain_edge_in),
          point_offset(point_offset_in),
          spacing(spacing_in) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int local_index) const noexcept {
        const int global_index = global_start + local_index - halo;
        coordinate(local_index) = domain_edge + (static_cast<VVM::Real>(global_index) + point_offset) * spacing;
    }
};

struct InitializeRegularLatLonMetrics {
    Kokkos::View<const VVM::Real*> latitude;
    Kokkos::View<VVM::Real*> sqrt_g;
    Kokkos::View<VVM::Real*> inv_sqrt_g;
    Kokkos::View<VVM::Real*> g_cov_11;
    Kokkos::View<VVM::Real*> sqrt_g_g_contra_11;
    Kokkos::View<VVM::Real*> sqrt_g_g_contra_22;
    Kokkos::View<VVM::Real*> physical_to_contravariant_11;
    Kokkos::View<VVM::Real*> contravariant_to_physical_11;
    VVM::Real radius;

    InitializeRegularLatLonMetrics(
        const Kokkos::View<const VVM::Real*>& latitude_in,
        const Kokkos::View<VVM::Real*>& sqrt_g_in,
        const Kokkos::View<VVM::Real*>& inv_sqrt_g_in,
        const Kokkos::View<VVM::Real*>& g_cov_11_in,
        const Kokkos::View<VVM::Real*>& sqrt_g_g_contra_11_in,
        const Kokkos::View<VVM::Real*>& sqrt_g_g_contra_22_in,
        const Kokkos::View<VVM::Real*>& physical_to_contravariant_11_in,
        const Kokkos::View<VVM::Real*>& contravariant_to_physical_11_in,
        const VVM::Real radius_in)
        : latitude(latitude_in),
          sqrt_g(sqrt_g_in),
          inv_sqrt_g(inv_sqrt_g_in),
          g_cov_11(g_cov_11_in),
          sqrt_g_g_contra_11(sqrt_g_g_contra_11_in),
          sqrt_g_g_contra_22(sqrt_g_g_contra_22_in),
          physical_to_contravariant_11(physical_to_contravariant_11_in),
          contravariant_to_physical_11(contravariant_to_physical_11_in),
          radius(radius_in) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int j) const noexcept {
        const VVM::Real cos_latitude = Kokkos::cos(latitude(j));
        const VVM::Real zonal_scale = radius * cos_latitude;
        const VVM::Real jacobian = radius * zonal_scale;

        sqrt_g(j) = jacobian;
        inv_sqrt_g(j) = VVM::real(1.0) / jacobian;

        g_cov_11(j) = zonal_scale * zonal_scale;

        sqrt_g_g_contra_11(j) = VVM::real(1.0) / cos_latitude;
        sqrt_g_g_contra_22(j) = cos_latitude;

        physical_to_contravariant_11(j) = VVM::real(1.0) / zonal_scale;
        contravariant_to_physical_11(j) = zonal_scale;
    }
};

GeometryField2D constant_field(const VVM::Real value) noexcept {
    return GeometryField2D::constant_value(value);
}

} // namespace

RegularLatLonGeometry::RegularLatLonGeometry(HorizontalDomainLayout layout,
    const VVM::Real dlongitude, const VVM::Real dlatitude,
    const VVM::Real longitude_west_edge, const VVM::Real latitude_south_edge,
    const VVM::Real radius)
    : layout_(layout),
      dlongitude_(dlongitude),
      dlatitude_(dlatitude),
      longitude_west_edge_(longitude_west_edge),
      latitude_south_edge_(latitude_south_edge),
      radius_(radius) {

    validate();
    initialize_coordinates();
    initialize_metrics();

    Kokkos::fence();
}

bool RegularLatLonGeometry::is_q1_staggered(const HorizontalLocation location) {

    switch (location) {
        case HorizontalLocation::T:
        case HorizontalLocation::V:
            return false;

        case HorizontalLocation::U:
        case HorizontalLocation::Z:
            return true;
    }

    throw std::invalid_argument("RegularLatLonGeometry received an invalid HorizontalLocation.");
}

bool RegularLatLonGeometry::is_q2_staggered(const HorizontalLocation location) {

    switch (location) {
        case HorizontalLocation::T:
        case HorizontalLocation::U:
            return false;

        case HorizontalLocation::V:
        case HorizontalLocation::Z:
            return true;
    }

    throw std::invalid_argument(
        "RegularLatLonGeometry received an invalid HorizontalLocation.");
}

const Kokkos::View<VVM::Real*>& RegularLatLonGeometry::q1_for(const HorizontalLocation location) const {
    return is_q1_staggered(location) ? q1_staggered_ : q1_centered_;
}

const Kokkos::View<VVM::Real*>& RegularLatLonGeometry::q2_for(const HorizontalLocation location) const {
    return is_q2_staggered(location) ? q2_staggered_ : q2_centered_;
}

const RegularLatLonGeometry::MetricStorage& RegularLatLonGeometry::metrics_for(const HorizontalLocation location) const {
    return is_q2_staggered(location) ? staggered_q2_metrics_ : centered_q2_metrics_;
}

void RegularLatLonGeometry::allocate_metric_storage(MetricStorage& storage, const std::string& label, const int size) {
    storage.sqrt_g = Kokkos::View<VVM::Real*>(label + "_sqrt_g", size);
    storage.inv_sqrt_g = Kokkos::View<VVM::Real*>(label + "_inv_sqrt_g", size);
    storage.g_cov_11 = Kokkos::View<VVM::Real*>(label + "_g_cov_11", size);
    storage.sqrt_g_g_contra_11 = Kokkos::View<VVM::Real*>(label + "_sqrt_g_g_contra_11", size);
    storage.sqrt_g_g_contra_22 = Kokkos::View<VVM::Real*>(label + "_sqrt_g_g_contra_22", size);
    storage.physical_to_contravariant_11 = Kokkos::View<VVM::Real*>(label + "_physical_to_contravariant_11", size);
    storage.contravariant_to_physical_11 = Kokkos::View<VVM::Real*>(label + "_contravariant_to_physical_11", size);
}

void RegularLatLonGeometry::validate() const {
    if (layout_.global_nx <= 0 || layout_.global_ny <= 0) {
        throw std::invalid_argument("RegularLatLonGeometry requires positive global dimensions.");
    }

    if (layout_.local_physical_nx <= 0 || layout_.local_physical_ny <= 0) {
        throw std::invalid_argument("RegularLatLonGeometry requires positive local dimensions.");
    }

    if (layout_.global_start_i < 0 || layout_.global_start_j < 0) {
        throw std::invalid_argument("RegularLatLonGeometry requires nonnegative global starts.");
    }

    if (layout_.global_start_i + layout_.local_physical_nx > layout_.global_nx ||
        layout_.global_start_j + layout_.local_physical_ny > layout_.global_ny) {
        throw std::invalid_argument("RegularLatLonGeometry local range exceeds the global domain.");
    }

    if (layout_.halo < 0) {
        throw std::invalid_argument("RegularLatLonGeometry requires halo >= 0.");
    }

    if (layout_.panel_id != -1) {
        throw std::invalid_argument("RegularLatLonGeometry requires panel_id == -1.");
    }

    if (!std::isfinite(dlongitude_) || dlongitude_ <= VVM::real(0.0) ||
        !std::isfinite(dlatitude_) || dlatitude_ <= VVM::real(0.0)) {
        throw std::invalid_argument("RegularLatLonGeometry requires finite positive angular spacing.");
    }

    if (!std::isfinite(longitude_west_edge_) || !std::isfinite(latitude_south_edge_)) {
        throw std::invalid_argument("RegularLatLonGeometry requires finite domain edges.");
    }

    if (!std::isfinite(radius_) || radius_ <= VVM::real(0.0)) {
        throw std::invalid_argument("RegularLatLonGeometry requires finite radius > 0.");
    }

    const VVM::Real half_pi = std::acos(VVM::real(-1.0)) / VVM::real(2.0);
    const VVM::Real minimum_halo_latitude = latitude_south_edge_ + (VVM::real(0.5) - static_cast<VVM::Real>(layout_.halo)) * dlatitude_;
    const VVM::Real maximum_halo_latitude = latitude_south_edge_ + static_cast<VVM::Real>(layout_.global_ny + layout_.halo) * dlatitude_;

    if (minimum_halo_latitude <= -half_pi || maximum_halo_latitude >= half_pi) {
        throw std::invalid_argument("RegularLatLonGeometry requires every T/U/V/Z latitude, "
            "including halos, to remain strictly between the poles.");
    }
}

void RegularLatLonGeometry::initialize_coordinates() {
    const int nx = layout_.local_total_nx();
    const int ny = layout_.local_total_ny();

    q1_centered_ = Kokkos::View<VVM::Real*>("regular_latlon_q1_centered", nx);
    q1_staggered_ = Kokkos::View<VVM::Real*>("regular_latlon_q1_staggered", nx);
    q2_centered_ = Kokkos::View<VVM::Real*>("regular_latlon_q2_centered", ny);
    q2_staggered_ = Kokkos::View<VVM::Real*>("regular_latlon_q2_staggered", ny);

    Kokkos::parallel_for("InitializeRegularLatLonQ1Centered",
        Kokkos::RangePolicy<>(0, nx),
        InitializeRegularLatLonCoordinate(q1_centered_,
            layout_.global_start_i, layout_.halo,
            longitude_west_edge_, VVM::real(0.5), dlongitude_));

    Kokkos::parallel_for("InitializeRegularLatLonQ1Staggered",
        Kokkos::RangePolicy<>(0, nx),
        InitializeRegularLatLonCoordinate(q1_staggered_,
            layout_.global_start_i, layout_.halo, longitude_west_edge_,
            VVM::real(1.0), dlongitude_));

    Kokkos::parallel_for("InitializeRegularLatLonQ2Centered",
        Kokkos::RangePolicy<>(0, ny),
        InitializeRegularLatLonCoordinate(q2_centered_,
            layout_.global_start_j, layout_.halo, latitude_south_edge_,
            VVM::real(0.5), dlatitude_));

    Kokkos::parallel_for("InitializeRegularLatLonQ2Staggered",
        Kokkos::RangePolicy<>(0, ny),
        InitializeRegularLatLonCoordinate(q2_staggered_,
            layout_.global_start_j, layout_.halo, latitude_south_edge_,
            VVM::real(1.0), dlatitude_));
}

void RegularLatLonGeometry::initialize_metrics() {
    const int ny = layout_.local_total_ny();

    allocate_metric_storage(centered_q2_metrics_, "regular_latlon_centered_q2", ny);
    allocate_metric_storage(staggered_q2_metrics_, "regular_latlon_staggered_q2", ny);

    Kokkos::parallel_for("InitializeRegularLatLonCenteredQ2Metrics", Kokkos::RangePolicy<>(0, ny),
        InitializeRegularLatLonMetrics(q2_centered_,
            centered_q2_metrics_.sqrt_g, centered_q2_metrics_.inv_sqrt_g,
            centered_q2_metrics_.g_cov_11, centered_q2_metrics_.sqrt_g_g_contra_11,
            centered_q2_metrics_.sqrt_g_g_contra_22, centered_q2_metrics_.physical_to_contravariant_11,
            centered_q2_metrics_.contravariant_to_physical_11, radius_));

    Kokkos::parallel_for("InitializeRegularLatLonStaggeredQ2Metrics",
        Kokkos::RangePolicy<>(0, ny),
        InitializeRegularLatLonMetrics(q2_staggered_,
            staggered_q2_metrics_.sqrt_g, staggered_q2_metrics_.inv_sqrt_g,
            staggered_q2_metrics_.g_cov_11, staggered_q2_metrics_.sqrt_g_g_contra_11,
            staggered_q2_metrics_.sqrt_g_g_contra_22, staggered_q2_metrics_.physical_to_contravariant_11,
            staggered_q2_metrics_.contravariant_to_physical_11, radius_));
}

HorizontalGeometryDeviceView
RegularLatLonGeometry::device_view_impl(const HorizontalLocation location) const {
    const auto& q1 = q1_for(location);
    const auto& q2 = q2_for(location);

    const auto& metrics = metrics_for(location);

    const GeometryField2D zero = constant_field(VVM::real(0.0));
    const GeometryField2D radius = constant_field(radius_);
    const GeometryField2D inverse_radius = constant_field(VVM::real(1.0) / radius_);
    const GeometryField2D radius_squared = constant_field(radius_ * radius_);

    HorizontalGeometryDeviceView result;

    result.q1 = GeometryField2D::varying_i(q1);
    result.q2 = GeometryField2D::varying_j(q2);

    // Longitude halos remain unwrapped. This keeps computational-coordinate
    // differences smooth through the periodic seam. Output adapters can
    // normalize geographic longitude to [0, 2*pi) later.
    result.longitude = result.q1;
    result.latitude = result.q2;

    result.sqrt_g = GeometryField2D::varying_j(metrics.sqrt_g);
    result.inv_sqrt_g = GeometryField2D::varying_j(metrics.inv_sqrt_g);

    result.g_cov.a11 = GeometryField2D::varying_j(metrics.g_cov_11);
    result.g_cov.a12 = zero;
    result.g_cov.a22 = radius_squared;

    result.sqrt_g_g_contra.a11 = GeometryField2D::varying_j(metrics.sqrt_g_g_contra_11);
    result.sqrt_g_g_contra.a12 = zero;
    result.sqrt_g_g_contra.a22 = GeometryField2D::varying_j(metrics.sqrt_g_g_contra_22);

    result.physical_to_contravariant.a11 = GeometryField2D::varying_j(metrics.physical_to_contravariant_11);
    result.physical_to_contravariant.a12 = zero;

    result.physical_to_contravariant.a21 = zero;
    result.physical_to_contravariant.a22 = inverse_radius;
    result.contravariant_to_physical.a11 = GeometryField2D::varying_j(metrics.contravariant_to_physical_11);

    result.contravariant_to_physical.a12 = zero;
    result.contravariant_to_physical.a21 = zero;
    result.contravariant_to_physical.a22 = radius;

    result.dq1 = dlongitude_;
    result.dq2 = dlatitude_;

    return result;
}

} // namespace Geometry
} // namespace Core
} // namespace VVM
