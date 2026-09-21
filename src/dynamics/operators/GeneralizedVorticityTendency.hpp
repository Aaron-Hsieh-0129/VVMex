#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_VORTICITY_TENDENCY_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_VORTICITY_TENDENCY_HPP

#include <string>

#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "dynamics/operators/GeneralizedHorizontalVorticityTransport.hpp"
#include "dynamics/operators/GeneralizedHorizontalDeformation.hpp"
#include "dynamics/operators/GeneralizedTopTransport.hpp"
#include "dynamics/operators/GeneralizedTopDeformation.hpp"

namespace VVM::Dynamics::Operators {

// Orchestration for the stationary, horizontal-only generalized coordinate.
// Inputs: synchronized canonical winds/vorticity, with halos in this chart.
// The supplied Grid must use the construction geometry and its local layout.
// Outputs: xi_con = omega^1, eta_con = -omega^2, zeta = omega^3 tendencies.
// The equation names remain "xi", "eta", "zeta"; they are not State aliases.
//
// Horizontal tendencies cover [halo, top); zeta is evaluated only at top.
// The caller owns rigid-lid/terrain masks, halo exchange and topology.
// No geometry-kind dispatch or physical-component conversion occurs here.
// Construction/prepare_execution must precede CUDA graph capture.
class GeneralizedVorticityTendency {
public:
    enum class Term { Transport, Stretching, Twisting, Planetary };

    explicit GeneralizedVorticityTendency(const Core::Geometry::HorizontalGeometry& geometry);

    static void prepare_execution();

    void add_from_canonical_state(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output,
        const std::string& variable,
        Term term) const;

private:
    Kokkos::View<GeneralizedHorizontalVorticityTransportDeviceView> horizontal_transport_;
    Kokkos::View<GeneralizedTopTransportDeviceView> top_transport_;

    GeneralizedHorizontalDeformationDeviceView horizontal_deformation_;
    GeneralizedTopDeformationDeviceView top_deformation_;
};

} // namespace VVM::Dynamics::Operators
#endif
