#ifndef VVM_DYNAMICS_OPERATORS_ORTHOGONAL_ANELASTIC_MASS_FLUX_HPP
#define VVM_DYNAMICS_OPERATORS_ORTHOGONAL_ANELASTIC_MASS_FLUX_HPP

#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "core/geometry/GeometryField2D.hpp"
#include "core/geometry/HorizontalGeometry.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/vvm_types.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Adapts an existing physical mass flux rho*U at U points to rho*u^1.
//
// The density factor is already part of physical_mass_flux. This adapter only
// changes basis and deliberately does not apply the face Jacobian. Conservative
// divergence applies the Jacobian exactly once.
template <typename PhysicalMassFluxView>
struct OrthogonalContravariantMassFluxQ1DeviceView {
    PhysicalMassFluxView physical_mass_flux;
    Core::Geometry::GeometryField2D physical_to_contravariant_11;

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    operator()(const int k, const int j, const int i) const noexcept {

        return physical_to_contravariant_11(j, i) * physical_mass_flux(k, j, i);
    }
};

// Adapts an existing physical mass flux rho*V at V points to rho*u^2.
//
// As for q1, this performs only the orthogonal basis conversion. The native
// V-face Jacobian remains the responsibility of HorizontalFluxDivergence.
template <typename PhysicalMassFluxView>
struct OrthogonalContravariantMassFluxQ2DeviceView {
    PhysicalMassFluxView physical_mass_flux;
    Core::Geometry::GeometryField2D physical_to_contravariant_22;

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    operator()(const int k, const int j, const int i) const noexcept {

        return physical_to_contravariant_22(j, i) * physical_mass_flux(k, j, i);
    }
};

inline bool
is_constant_zero(const Core::Geometry::GeometryField2D& field) noexcept {

    return field.layout == Core::Geometry::GeometryFieldLayout::Constant &&
           field.constant == VVM::real(0.0);
}

inline void
require_orthogonal_mass_flux_geometry(const Core::Geometry::HorizontalGeometry& geometry) {

    const auto u = geometry.device_view(Core::Geometry::HorizontalLocation::U);
    const auto v = geometry.device_view(Core::Geometry::HorizontalLocation::V);

    if (!is_constant_zero(u.physical_to_contravariant.a12) ||
        !is_constant_zero(u.physical_to_contravariant.a21) ||
        !is_constant_zero(v.physical_to_contravariant.a12) ||
        !is_constant_zero(v.physical_to_contravariant.a21)) {

        throw std::invalid_argument("Physical mass-flux adaptation currently requires an "
                                    "orthogonal horizontal geometry.");
    }
}

template <typename PhysicalMassFluxView>
OrthogonalContravariantMassFluxQ1DeviceView<PhysicalMassFluxView>
make_orthogonal_contravariant_mass_flux_q1_device_view(
    const Core::Geometry::HorizontalGeometry& geometry,
    const PhysicalMassFluxView& physical_mass_flux) {

    require_orthogonal_mass_flux_geometry(geometry);

    const auto u = geometry.device_view(Core::Geometry::HorizontalLocation::U);

    OrthogonalContravariantMassFluxQ1DeviceView<PhysicalMassFluxView> result;

    result.physical_mass_flux = physical_mass_flux;
    result.physical_to_contravariant_11 = u.physical_to_contravariant.a11;

    return result;
}

template <typename PhysicalMassFluxView>
OrthogonalContravariantMassFluxQ2DeviceView<PhysicalMassFluxView>
make_orthogonal_contravariant_mass_flux_q2_device_view(
    const Core::Geometry::HorizontalGeometry& geometry,
    const PhysicalMassFluxView& physical_mass_flux) {

    require_orthogonal_mass_flux_geometry(geometry);

    const auto v = geometry.device_view(Core::Geometry::HorizontalLocation::V);

    OrthogonalContravariantMassFluxQ2DeviceView<PhysicalMassFluxView> result;

    result.physical_mass_flux = physical_mass_flux;
    result.physical_to_contravariant_22 = v.physical_to_contravariant.a22;

    return result;
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_ORTHOGONAL_ANELASTIC_MASS_FLUX_HPP
