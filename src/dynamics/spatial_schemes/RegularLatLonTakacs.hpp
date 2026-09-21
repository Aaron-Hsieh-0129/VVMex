#ifndef VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP
#define VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP

#include "dynamics/operators/GeneralizedBuoyancy.hpp"
#include "dynamics/operators/GeneralizedScalarTransport.hpp"
#include "dynamics/operators/GeneralizedVorticityTendency.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"

namespace VVM {
namespace Dynamics {

// RLL configuration and physical-tendency boundary for generalized scalar,
// vorticity and buoyancy operators. The numerical kernels consume geometry
// data, not latitude-longitude-specific equations. This adapter still
// returns physical xi/eta increments to the existing tendency accumulator.
class RegularLatLonTakacs final : public SpatialScheme {
public:
    // Dry mode requires a dry State; moist mode requires physical qv, qp and
    // initialized native terrain masks. The factory derives this from P3,
    // not a new user-facing dry/moist configuration key.
    // Construction and backend preparation must occur before graph capture.
    explicit RegularLatLonTakacs(const Core::Geometry::HorizontalGeometry& geometry,
        bool enable_dry_buoyancy = false,
        bool enable_moist_buoyancy = false);

    static bool
    is_moist_scalar(const std::string& name) {
        return name == "qv" || name == "qc" || name == "qr" || name == "qi" || name == "qm" ||
               name == "nc" || name == "nr" || name == "ni" || name == "bm";
    }

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
        const Core::Field<3>& contravariant_mass_flux_q1,
        const Core::Field<3>& contravariant_mass_flux_q2,
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
        add_vorticity_tendency(state,
            grid,
            params,
            output,
            "xi",
            Operators::GeneralizedVorticityTendency::Term::Planetary);
    }

    void
    calculate_coriolis_tendency_y(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output) const override {
        add_vorticity_tendency(state,
            grid,
            params,
            output,
            "eta",
            Operators::GeneralizedVorticityTendency::Term::Planetary);
    }

    void
    calculate_coriolis_tendency_z(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output) const override {
        add_vorticity_tendency(state,
            grid,
            params,
            output,
            "zeta",
            Operators::GeneralizedVorticityTendency::Term::Planetary);
    }

private:
    void add_vorticity_tendency(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output,
        const std::string& variable,
        Operators::GeneralizedVorticityTendency::Term term) const;

    void validate_dry_buoyancy(
        const Core::State& state, const Core::Grid& grid, const Core::Parameters& params) const;

    void add_buoyancy_tendency(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output,
        bool xi_component) const;

    Operators::GeneralizedScalarTransport scalar_transport_;
    Operators::GeneralizedBuoyancy buoyancy_;
    Operators::GeneralizedVorticityTendency vorticity_;

    bool enable_dry_buoyancy_;
    bool enable_moist_buoyancy_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_REGULAR_LAT_LON_TAKACS_HPP
