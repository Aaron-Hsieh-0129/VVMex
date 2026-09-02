#ifndef VVM_CORE_GEOMETRY_CARTESIAN_GEOMETRY_HPP
#define VVM_CORE_GEOMETRY_CARTESIAN_GEOMETRY_HPP

#include <array>
#include <cstddef>
#include <Kokkos_Core.hpp>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM {
namespace Core {
namespace Geometry {

class CartesianGeometry final : public HorizontalGeometry {
public:
    CartesianGeometry(HorizontalDomainLayout layout, VVM::Real dx, VVM::Real dy);

    GeometryKind kind() const noexcept override {
        return GeometryKind::Cartesian;
    }

    const char* name() const noexcept override {
        return "cartesian";
    }

    const HorizontalDomainLayout&
    layout() const noexcept override {
        return layout_;
    }

    VVM::Real dq1() const noexcept override {
        return dx_;
    }

    VVM::Real dq2() const noexcept override {
        return dy_;
    }

protected:
    HorizontalGeometryDeviceView device_view_impl(HorizontalLocation location) const override;

private:
    struct LocationStorage {
        Kokkos::View<VVM::Real*> q1;
        Kokkos::View<VVM::Real*> q2;
    };

    static std::size_t location_index(HorizontalLocation location);
    static VVM::Real q1_offset(HorizontalLocation location) noexcept;
    static VVM::Real q2_offset(HorizontalLocation location) noexcept;

    void validate() const;

    void initialize_location(
        HorizontalLocation location);

    HorizontalDomainLayout layout_;

    VVM::Real dx_;
    VVM::Real dy_;

    std::array<LocationStorage, 4> locations_;
};

}
}
}

#endif
