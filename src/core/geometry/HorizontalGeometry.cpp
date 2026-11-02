#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM::Core::Geometry {

std::array<VVM::Real, 2>
HorizontalGeometry::physical_spacing(const HorizontalLocation location,
    const int j,
    const int i) const {
    const auto geometry = device_view(location);
    Kokkos::View<VVM::Real*> lengths("horizontal physical spacing", 2);
    Kokkos::parallel_for("HorizontalPhysicalSpacing", 1, KOKKOS_LAMBDA(const int) {
        lengths(0) = geometry.physical_spacing_q1(j, i);
        lengths(1) = geometry.physical_spacing_q2(j, i);
    });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lengths);
    return {host(0), host(1)};
}

} // namespace VVM::Core::Geometry
