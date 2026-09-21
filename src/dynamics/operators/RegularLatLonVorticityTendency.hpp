#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP

#include <string>

#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "dynamics/operators/GeneralizedHorizontalVorticityTransport.hpp"
#include "dynamics/operators/GeneralizedHorizontalDeformation.hpp"
#include "dynamics/operators/GeneralizedTopTransport.hpp"
#include "dynamics/operators/GeneralizedTopDeformation.hpp"

namespace VVM::Dynamics::Operators {

// Temporary RLL production orchestrator. All transport and deformation
// stencils now use generalized-coordinate operators. Wind inputs are
// synchronized u_con/v_con, including halos; w remains physical vertical wind.
//
// xi_con = omega^1; eta_con = -omega^2. zeta is relative physical vertical
// vorticity, equal to omega^3 for the current horizontal-only mapping.
// The output accumulator remains physical: horizontal tendencies are
// converted once here, and DynamicalCore converts the total tendency.
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
    Kokkos::View<GeneralizedTopTransportDeviceView> transport_;

    GeneralizedHorizontalDeformationDeviceView horizontal_deformation_;
    GeneralizedTopDeformationDeviceView top_deformation_;
};

} // namespace VVM::Dynamics::Operators
#endif
