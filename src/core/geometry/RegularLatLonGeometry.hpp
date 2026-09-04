#ifndef VVM_CORE_GEOMETRY_REGULAR_LAT_LON_GEOMETRY_HPP
#define VVM_CORE_GEOMETRY_REGULAR_LAT_LON_GEOMETRY_HPP

#include <string>

#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM {
namespace Core {
namespace Geometry {

class RegularLatLonGeometry final : public HorizontalGeometry {
public:
    RegularLatLonGeometry(HorizontalDomainLayout layout,
        VVM::Real dlongitude, VVM::Real dlatitude,
        VVM::Real longitude_west_edge, VVM::Real latitude_south_edge, VVM::Real radius);

    GeometryKind kind() const noexcept override {
        return GeometryKind::RegularLatLon;
    }

    const char* name() const noexcept override {
        return "regular_latlon";
    }

    const HorizontalDomainLayout& layout() const noexcept override {
        return layout_;
    }

    VVM::Real dq1() const noexcept override {
        return dlongitude_;
    }

    VVM::Real dq2() const noexcept override {
        return dlatitude_;
    }

    VVM::Real longitude_west_edge() const noexcept {
        return longitude_west_edge_;
    }

    VVM::Real latitude_south_edge() const noexcept {
        return latitude_south_edge_;
    }

    VVM::Real radius() const noexcept {
        return radius_;
    }

protected:
    HorizontalGeometryDeviceView device_view_impl(
        HorizontalLocation location) const override;

private:
    struct MetricStorage {
        Kokkos::View<VVM::Real*> sqrt_g;
        Kokkos::View<VVM::Real*> inv_sqrt_g;
        Kokkos::View<VVM::Real*> g_cov_11;
        Kokkos::View<VVM::Real*> sqrt_g_g_contra_11;
        Kokkos::View<VVM::Real*> sqrt_g_g_contra_22;
        Kokkos::View<VVM::Real*> physical_to_contravariant_11;
        Kokkos::View<VVM::Real*> contravariant_to_physical_11;
    };

    static bool is_q1_staggered(HorizontalLocation location);
    static bool is_q2_staggered(HorizontalLocation location);

    const Kokkos::View<VVM::Real*>& q1_for(HorizontalLocation location) const;
    const Kokkos::View<VVM::Real*>& q2_for(HorizontalLocation location) const;
    const MetricStorage& metrics_for(HorizontalLocation location) const;

    static void allocate_metric_storage(MetricStorage& storage, const std::string& label, int size);

    void validate() const;
    void initialize_coordinates();
    void initialize_metrics();

    HorizontalDomainLayout layout_;

    VVM::Real dlongitude_;
    VVM::Real dlatitude_;
    VVM::Real longitude_west_edge_;
    VVM::Real latitude_south_edge_;
    VVM::Real radius_;

    Kokkos::View<VVM::Real*> q1_centered_;
    Kokkos::View<VVM::Real*> q1_staggered_;
    Kokkos::View<VVM::Real*> q2_centered_;
    Kokkos::View<VVM::Real*> q2_staggered_;

    MetricStorage centered_q2_metrics_;
    MetricStorage staggered_q2_metrics_;
};

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_REGULAR_LAT_LON_GEOMETRY_HPP
