#ifndef VVM_CORE_GEOMETRY_GEOMETRY_FIELD_2D_HPP
#define VVM_CORE_GEOMETRY_GEOMETRY_FIELD_2D_HPP

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "core/vvm_types.hpp"

namespace VVM {
namespace Core {
namespace Geometry {

using GeometryConstView1D = Kokkos::View<const VVM::Real*>;
using GeometryConstView2D = Kokkos::View<const VVM::Real**>;

enum class GeometryFieldLayout : std::uint8_t {
    Constant,
    VaryingI,
    VaryingJ,
    Full2D
};

struct GeometryField2D {
    GeometryFieldLayout layout = GeometryFieldLayout::Constant;

    VVM::Real constant = VVM::real(0.0);

    GeometryConstView1D one_dimensional;
    GeometryConstView2D two_dimensional;

    KOKKOS_INLINE_FUNCTION
    VVM::Real operator()(const int j, const int i) const noexcept {
        switch (layout) {
            case GeometryFieldLayout::Constant:
                return constant;

            case GeometryFieldLayout::VaryingI:
                return one_dimensional(i);

            case GeometryFieldLayout::VaryingJ:
                return one_dimensional(j);

            case GeometryFieldLayout::Full2D:
                return two_dimensional(j, i);
        }

        return VVM::real(0.0);
    }

    static GeometryField2D constant_value(const VVM::Real value) noexcept {
        GeometryField2D result;

        result.layout = GeometryFieldLayout::Constant;
        result.constant = value;

        return result;
    }

    static GeometryField2D varying_i(const GeometryConstView1D values) noexcept {
        GeometryField2D result;

        result.layout = GeometryFieldLayout::VaryingI;
        result.one_dimensional = values;

        return result;
    }

    static GeometryField2D varying_j(const GeometryConstView1D values) noexcept {
        GeometryField2D result;

        result.layout = GeometryFieldLayout::VaryingJ;
        result.one_dimensional = values;

        return result;
    }

    static GeometryField2D full_2d(const GeometryConstView2D values) noexcept {
        GeometryField2D result;

        result.layout = GeometryFieldLayout::Full2D;
        result.two_dimensional = values;

        return result;
    }
};

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_GEOMETRY_FIELD_2D_HPP
