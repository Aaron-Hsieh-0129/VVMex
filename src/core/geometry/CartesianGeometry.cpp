#include "core/geometry/CartesianGeometry.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace VVM {
namespace Core {
namespace Geometry {

namespace {

GeometryField2D constant_field(
    const VVM::Real value) noexcept {
    return GeometryField2D::constant_value(value);
}

struct InitializeCartesianQ1 {
    Kokkos::View<VVM::Real*> q1;

    int global_start_i;
    int halo;

    VVM::Real offset_i;
    VVM::Real dx;

    InitializeCartesianQ1(
        const Kokkos::View<VVM::Real*>& q1_in,
        const int global_start_i_in,
        const int halo_in,
        const VVM::Real offset_i_in,
        const VVM::Real dx_in)
        : q1(q1_in),
          global_start_i(global_start_i_in),
          halo(halo_in),
          offset_i(offset_i_in),
          dx(dx_in) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const noexcept {
        const int global_i = global_start_i + i - halo;
        q1(i) = (static_cast<VVM::Real>(global_i) + offset_i) * dx;
    }
};

struct InitializeCartesianQ2 {
    Kokkos::View<VVM::Real*> q2;

    int global_start_j;
    int halo;

    VVM::Real offset_j;
    VVM::Real dy;

    InitializeCartesianQ2(
        const Kokkos::View<VVM::Real*>& q2_in,
        const int global_start_j_in,
        const int halo_in,
        const VVM::Real offset_j_in,
        const VVM::Real dy_in)
        : q2(q2_in),
          global_start_j(global_start_j_in),
          halo(halo_in),
          offset_j(offset_j_in),
          dy(dy_in) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int j) const noexcept {
        const int global_j = global_start_j + j - halo;
        q2(j) = (static_cast<VVM::Real>(global_j) + offset_j) * dy;
    }
};


} // namespace

CartesianGeometry::CartesianGeometry(HorizontalDomainLayout layout, const VVM::Real dx, const VVM::Real dy)
    : layout_(layout), dx_(dx), dy_(dy) {
    validate();

    initialize_location(HorizontalLocation::T);
    initialize_location(HorizontalLocation::U);
    initialize_location(HorizontalLocation::V);
    initialize_location(HorizontalLocation::Z);

    Kokkos::fence();
}

std::size_t CartesianGeometry::location_index(const HorizontalLocation location) {
    switch (location) {
        case HorizontalLocation::T:
            return 0;
        case HorizontalLocation::U:
            return 1;
        case HorizontalLocation::V:
            return 2;
        case HorizontalLocation::Z:
            return 3;
    }

    throw std::invalid_argument("CartesianGeometry received an invalid HorizontalLocation.");
}

VVM::Real CartesianGeometry::q1_offset(const HorizontalLocation location) noexcept {

    switch (location) {
        case HorizontalLocation::T:
        case HorizontalLocation::V:
            return VVM::real(0.0);

        case HorizontalLocation::U:
        case HorizontalLocation::Z:
            return VVM::real(0.5);
    }

    return VVM::real(0.0);
}

VVM::Real CartesianGeometry::q2_offset(const HorizontalLocation location) noexcept {

    switch (location) {
        case HorizontalLocation::T:
        case HorizontalLocation::U:
            return VVM::real(0.0);

        case HorizontalLocation::V:
        case HorizontalLocation::Z:
            return VVM::real(0.5);
    }

    return VVM::real(0.0);
}

void CartesianGeometry::validate() const {
    if (layout_.global_nx <= 0) {
        throw std::invalid_argument("CartesianGeometry requires global_nx > 0.");
    }

    if (layout_.global_ny <= 0) {
        throw std::invalid_argument("CartesianGeometry requires global_ny > 0.");
    }

    if (layout_.local_physical_nx <= 0) {
        throw std::invalid_argument("CartesianGeometry requires local_physical_nx > 0.");
    }

    if (layout_.local_physical_ny <= 0) {
        throw std::invalid_argument("CartesianGeometry requires local_physical_ny > 0.");
    }

    if (layout_.global_start_i < 0) {
        throw std::invalid_argument("CartesianGeometry requires global_start_i >= 0.");
    }

    if (layout_.global_start_j < 0) {
        throw std::invalid_argument("CartesianGeometry requires global_start_j >= 0.");
    }

    if (layout_.global_start_i + layout_.local_physical_nx > layout_.global_nx) {
        throw std::invalid_argument("CartesianGeometry local x range exceeds the global x range.");
    }

    if (layout_.global_start_j + layout_.local_physical_ny > layout_.global_ny) {
        throw std::invalid_argument("CartesianGeometry local y range exceeds the global y range.");
    }

    if (layout_.halo < 0) {
        throw std::invalid_argument("CartesianGeometry requires halo >= 0.");
    }

    if (layout_.panel_id != -1) {
        throw std::invalid_argument("CartesianGeometry requires panel_id == -1.");
    }

    if (!std::isfinite(dx_) || dx_ <= VVM::real(0.0)) {
        throw std::invalid_argument("CartesianGeometry requires finite dx > 0.");
    }

    if (!std::isfinite(dy_) || dy_ <= VVM::real(0.0)) {
        throw std::invalid_argument("CartesianGeometry requires finite dy > 0.");
    }
}

void CartesianGeometry::initialize_location(const HorizontalLocation location) {
    const std::size_t storage_index = location_index(location);
    auto& storage = locations_[storage_index];
    const std::string location_name = horizontal_location_to_string(location);

    storage.q1 = Kokkos::View<VVM::Real*>("cartesian_q1_" + location_name, layout_.local_total_nx());
    storage.q2 = Kokkos::View<VVM::Real*>("cartesian_q2_" + location_name, layout_.local_total_ny());

    const int local_total_nx = layout_.local_total_nx();
    const int local_total_ny = layout_.local_total_ny();

    const InitializeCartesianQ1 initialize_q1(storage.q1, layout_.global_start_i, layout_.halo, q1_offset(location), dx_);
    const InitializeCartesianQ2 initialize_q2(storage.q2, layout_.global_start_j, layout_.halo, q2_offset(location), dy_);

    Kokkos::parallel_for("InitializeCartesianQ1_" + location_name,
        Kokkos::RangePolicy<>(0, local_total_nx),
        initialize_q1);

    Kokkos::parallel_for("InitializeCartesianQ2_" + location_name,
        Kokkos::RangePolicy<>(0, local_total_ny),
        initialize_q2);
}

HorizontalGeometryDeviceView
CartesianGeometry::device_view_impl(
    const HorizontalLocation location) const {
    const auto& storage = locations_[location_index(location)];

    const GeometryField2D zero = constant_field(VVM::real(0.0));
    const GeometryField2D one = constant_field(VVM::real(1.0));

    HorizontalGeometryDeviceView result;

    result.q1 = GeometryField2D::varying_i(storage.q1);
    result.q2 = GeometryField2D::varying_j(storage.q2);

    // Geographic coordinates are not derived from Cartesian x/y here.
    // Existing VVMex State::lon/lat remains the geographic source.
    result.longitude = zero;
    result.latitude = zero;

    result.sqrt_g = one;
    result.inv_sqrt_g = one;

    result.g_cov.a11 = one;
    result.g_cov.a12 = zero;
    result.g_cov.a22 = one;

    result.sqrt_g_g_contra.a11 = one;
    result.sqrt_g_g_contra.a12 = zero;
    result.sqrt_g_g_contra.a22 = one;

    result.physical_to_contravariant.a11 = one;
    result.physical_to_contravariant.a12 = zero;
    result.physical_to_contravariant.a21 = zero;
    result.physical_to_contravariant.a22 = one;

    result.contravariant_to_physical.a11 = one;
    result.contravariant_to_physical.a12 = zero;
    result.contravariant_to_physical.a21 = zero;
    result.contravariant_to_physical.a22 = one;

    result.dq1 = dx_;
    result.dq2 = dy_;

    return result;
}

} // namespace Geometry
} // namespace Core
} // namespace VVM

