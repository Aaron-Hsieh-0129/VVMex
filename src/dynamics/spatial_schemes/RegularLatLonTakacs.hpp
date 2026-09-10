#ifndef VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP
#define VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP

#include "dynamics/operators/RegularLatLonDryBuoyancy.hpp"
#include "dynamics/operators/RegularLatLonScalarTransport.hpp"
#include "dynamics/operators/RegularLatLonVorticityTendency.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"

namespace VVM {
namespace Dynamics {

// Guarded Takacs implementation for ordinary density-normalized scalars
// and explicitly selected dry horizontal-vorticity buoyancy on regular
// latitude-longitude geometry.
//
// xi is physical eastward vorticity at V. eta is negative physical
// northward vorticity at U, preserving the existing State convention.
//
// Flat RLL vorticity transport and deformation use the shared CVVM operators.
// Moisture and terrain remain unsupported. Full-model execution remains guarded.
class RegularLatLonTakacs final : public SpatialScheme {
public:
    // Enabling dry buoyancy declares that the caller supplies a dry State.
    // Construction and backend preparation must occur before graph capture.
    explicit RegularLatLonTakacs(const Core::Geometry::HorizontalGeometry& geometry,
        bool enable_dry_buoyancy = false);

    bool
    handles_multidimensional_advection() const override {

        return true;
    }

    bool
    produces_anelastic_scalar_flux_divergence() const override {

        return true;
    }

    void calculate_advection_tendency(const Core::State& state,
        const Core::Field<3>& scalar,
        const Core::Field<3>& physical_mass_flux_q1,
        const Core::Field<3>& physical_mass_flux_q2,
        const Core::Field<3>& vertical_mass_flux,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency,
        const std::string& var_name,
        VVM::Real stage_dt) const override;

    void calculate_buoyancy_tendency_x(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency) const override;

    void calculate_buoyancy_tendency_y(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency) const override;

    void calculate_stretching_tendency_x(const Core::State&,
        const Core::Grid&,
        const Core::Parameters&,
        Core::Field<3>&,
        const std::string&) const override;
    void calculate_stretching_tendency_y(const Core::State&,
        const Core::Grid&,
        const Core::Parameters&,
        Core::Field<3>&,
        const std::string&) const override;
    void calculate_stretching_tendency_z(const Core::State&,
        const Core::Grid&,
        const Core::Parameters&,
        Core::Field<3>&,
        const std::string&) const override;
    void calculate_twisting_tendency_x(const Core::State&,
        const Core::Grid&,
        const Core::Parameters&,
        Core::Field<3>&,
        const std::string&) const override;
    void calculate_twisting_tendency_y(const Core::State&,
        const Core::Grid&,
        const Core::Parameters&,
        Core::Field<3>&,
        const std::string&) const override;
    void calculate_twisting_tendency_z(const Core::State&,
        const Core::Grid&,
        const Core::Parameters&,
        Core::Field<3>&,
        const std::string&) const override;

    void
    calculate_coriolis_tendency_x(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output) const override {
        vorticity_.add(state,
            grid,
            params,
            output,
            "xi",
            Operators::RegularLatLonVorticityTendency::Term::Planetary);
    }
    void
    calculate_coriolis_tendency_y(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output) const override {
        vorticity_.add(state,
            grid,
            params,
            output,
            "eta",
            Operators::RegularLatLonVorticityTendency::Term::Planetary);
    }
    void
    calculate_coriolis_tendency_z(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output) const override {
        vorticity_.add(state,
            grid,
            params,
            output,
            "zeta",
            Operators::RegularLatLonVorticityTendency::Term::Planetary);
    }

private:
    void validate_dry_buoyancy(
        const Core::State& state, const Core::Grid& grid, const Core::Parameters& params) const;

    Operators::RegularLatLonScalarTransport scalar_transport_;
    Operators::RegularLatLonDryBuoyancy dry_buoyancy_;
    Operators::RegularLatLonVorticityTendency vorticity_;
    bool enable_dry_buoyancy_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP
