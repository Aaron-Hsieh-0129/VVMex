#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP

#include <string>

#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "dynamics/operators/GeneralizedHorizontalVorticityTransport.hpp"
#include "dynamics/operators/GeneralizedHorizontalDeformation.hpp"
#include "dynamics/operators/GeneralizedTopDeformation.hpp"
#include "dynamics/operators/RegularLatLonTopTransport.hpp"

namespace VVM::Dynamics::Operators {

// Temporary production RLL adapter. Horizontal transport and all
// deformation read canonical vorticity directly. Horizontal wind inputs
// to those generalized kernels are synchronized u_con/v_con, including
// halos. Only top transport still uses the legacy physical-wind adapter.
//
// xi_con = omega^1; eta_con = -omega^2. zeta is the relative physical
// vertical component, equal to omega^3 for this horizontal-only mapping.
// The output accumulator remains physical until the orchestration moves
// to canonical tendencies; horizontal output is converted exactly once
// here. DynamicalCore still converts the total physical tendency.
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

    // Each deformation operator contains only two Real spacings.
    GeneralizedHorizontalDeformationDeviceView horizontal_deformation_;
    GeneralizedTopDeformationDeviceView top_deformation_;
};

} // namespace VVM::Dynamics::Operators
#endif
