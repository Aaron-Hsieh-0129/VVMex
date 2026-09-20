#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP

#include "core/State.hpp"
#include "core/Parameters.hpp"
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
// The underlying CVVM stencils retain their established arithmetic.
// Density-normalized physical quantities are reconstructed locally inside
// the device operators from the canonical state without modifying xi/eta
// State fields.
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
