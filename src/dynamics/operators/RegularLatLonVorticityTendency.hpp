#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP

#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "dynamics/operators/GeneralizedHorizontalVorticityTransport.hpp"
#include "dynamics/operators/RegularLatLonTopTransport.hpp"
#include "dynamics/operators/RegularLatLonTopDeformation.hpp"

namespace VVM::Dynamics::Operators {

// Production RLL vorticity-tendency adapter.
//
// Persistent horizontal vorticity is canonical:
//
//     xi_con  =  omega^1
//     eta_con = -omega^2
//
// zeta remains the physical vertical component for the current
// horizontal-only generalized coordinate.
//
// Horizontal transport reads canonical wind/vorticity directly and uses
// the generalized-coordinate operator. The remaining top/deformation
// terms still use temporary physical-component adapters. This production
// boundary returns physical xi/eta tendencies until the complete tendency
// orchestration is migrated. The caller supplies synchronized u_con/v_con,
// including halos, before evaluating horizontal transport.
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
    Kokkos::View<GeneralizedHorizontalVorticityTransportDeviceView> horizontal_transport_;
    Kokkos::View<RegularLatLonTopTransportDeviceView> transport_;
    Kokkos::View<RegularLatLonTopDeformationDeviceView> deformation_;
};

} // namespace VVM::Dynamics::Operators
#endif
