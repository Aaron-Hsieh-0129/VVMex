#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP

#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "dynamics/operators/RegularLatLonTopTransport.hpp"
#include "dynamics/operators/RegularLatLonTopDeformation.hpp"

namespace VVM::Dynamics::Operators {

// Production launch adapter. It borrows the density-normalized vorticity
// used by DynamicalCore during tendency evaluation; no extra normalization.
//
// During normal tendency evaluation DynamicalCore temporarily transforms the
// persistent physical State fields:
//
//     xi   -> xi   / rhobar_up
//     eta  -> eta  / rhobar_up
//     zeta -> zeta / rhobar
//
// before entering this operator.
//
// Therefore this operator consumes density-normalized PHYSICAL / legacy-sign
// State vorticity, not the persistent contravariant shadow fields
// xi_con / eta_con / zeta_con.
class RegularLatLonVorticityTendency {
public:
    enum class Term { Transport, Stretching, Twisting, Planetary };
    explicit RegularLatLonVorticityTendency(const Core::Geometry::HorizontalGeometry& geometry);
    static void prepare_execution();

    void add_from_density_normalized_physical_state(const Core::State& state,
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
