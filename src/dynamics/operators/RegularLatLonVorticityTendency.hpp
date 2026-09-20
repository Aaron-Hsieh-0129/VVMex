#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP

#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "dynamics/operators/RegularLatLonTopTransport.hpp"
#include "dynamics/operators/RegularLatLonTopDeformation.hpp"

namespace VVM::Dynamics::Operators {

// Production launch adapter for canonical RLL dynamics.
//
// Persistent horizontal vorticity is supplied through:
//
//     xi_con  =  omega^1
//     eta_con = -omega^2
//
// zeta remains the physical/vertical component for the current
// horizontal-only generalized coordinate.
//
// The existing CVVM/RLL device stencils historically operate on
// density-normalized physical horizontal-vorticity components. To preserve
// their established arithmetic exactly, this adapter materializes those
// values lazily from the canonical state on device:
//
//     xi/rho_up  = (h1 * xi_con)  / rho_up
//     eta/rho_up = (h2 * eta_con) / rho_up
//     zeta/rho   = zeta / rho
//
// No physical xi/eta State allocation is consumed or modified here.
class RegularLatLonVorticityTendency {
public:
    enum class Term { Transport, Stretching, Twisting, Planetary };
    explicit RegularLatLonVorticityTendency(const Core::Geometry::HorizontalGeometry& geometry);
    static void prepare_execution();

    void add_from_canonical_state(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output,
        const std::string& variable,
        Term term) const;

private:
    Kokkos::View<RegularLatLonTopTransportDeviceView> transport_;
    Kokkos::View<RegularLatLonTopDeformationDeviceView> deformation_;
};

} // namespace VVM::Dynamics::Operators
#endif
