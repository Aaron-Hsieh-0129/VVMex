#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_SCALAR_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_SCALAR_TRANSPORT_HPP

#include "core/Field.hpp"
#include "core/geometry/GeometryField2D.hpp"
#include "core/geometry/HorizontalGeometry.hpp"
#include "dynamics/operators/TakacsScalarTransport.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Field-level launcher for dry scalar transport on regular latitude-longitude
// geometry.
//
// scalar_q:
//     Density-normalized scalar at T points.
//
// contravariant_mass_flux_q1:
//     rho * u^1 at U points.
//
// contravariant_mass_flux_q2:
//     rho * u^2 at V points.
//
// Horizontal Jacobian factors are NOT included in either field.
// HorizontalFluxDivergence applies the native face Jacobian exactly once.
//
// vertical_mass_flux:
//     rho_up * w at vertical faces, in kg m^-2 s^-1.
//
// vertical_cell_spacing:
//     Physical T-cell thickness, dz_mid, in metres.
//
// out_flux_convergence:
//     Receives the additive, density-weighted flux convergence. No division by
//     rhobar is performed here; AdvectionTerm retains that responsibility.
//
// The caller owns halo exchange, horizontal boundary conditions,
// synchronization, and output initialization.
class RegularLatLonScalarTransport {
public:
    explicit RegularLatLonScalarTransport(const Core::Geometry::HorizontalGeometry& geometry,
        VVM::Real alpha = VVM::real(1.0));

    // Prepare this compilation unit before manual CUDA graph capture.
    // This touches no model fields and is a no-op on CPU.
    static void prepare_execution();

    // Adds transport over [k_begin, k_end) and the physical horizontal domain.
    // Horizontal halos and vertical levels outside this range remain untouched.
    //
    // At least three scalar levels are required because the Takacs boundary
    // reconstruction uses a one-sided four-point stencil.
    //
    // All used vertical_cell_spacing entries must be positive and finite.
    // Input and output storage must not overlap.
    void add_flux_convergence(const Core::Field<3>& scalar_q,
        const Core::Field<3>& contravariant_mass_flux_q1,
        const Core::Field<3>& contravariant_mass_flux_q2,
        const Core::Field<3>& vertical_mass_flux,
        const Core::Field<1>& vertical_cell_spacing,
        Core::Field<3>& out_flux_convergence,
        int k_begin,
        int k_end) const;

private:
    void validate_volume(const Core::Field<3>& field, int nz, const char* role) const;

    Core::Geometry::HorizontalDomainLayout layout_;
    TakacsScalarTransportDeviceView transport_;
    Kokkos::View<TakacsScalarTransportDeviceView> device_transport_;
};

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_SCALAR_TRANSPORT_HPP
