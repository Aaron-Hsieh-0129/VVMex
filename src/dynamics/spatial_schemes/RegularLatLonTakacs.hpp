#ifndef VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP
#define VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP

#include "dynamics/operators/RegularLatLonScalarTransport.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"

namespace VVM {
namespace Dynamics {

// Guarded Takacs implementation for ordinary density-normalized scalars on
// regular latitude-longitude geometry.
//
// This class does not implement the RLL vorticity advection, stretching,
// twisting, buoyancy, or Coriolis terms. NumericalMethodFactory rejects those
// combinations before constructing this scheme.
class RegularLatLonTakacs final : public SpatialScheme {
public:
    explicit RegularLatLonTakacs(
        const Core::Geometry::HorizontalGeometry& geometry);

    bool handles_multidimensional_advection()
        const override {

        return true;
    }

    bool produces_anelastic_scalar_flux_divergence()
        const override {

        return true;
    }

    void calculate_advection_tendency(
        const Core::State& state,
        const Core::Field<3>& scalar,
        const Core::Field<3>& physical_mass_flux_q1,
        const Core::Field<3>& physical_mass_flux_q2,
        const Core::Field<3>& vertical_mass_flux,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency,
        const std::string& var_name,
        VVM::Real stage_dt) const override;

private:
    Operators::RegularLatLonScalarTransport
        scalar_transport_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP
