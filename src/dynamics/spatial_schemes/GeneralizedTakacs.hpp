#ifndef VVM_DYNAMICS_GENERALIZED_TAKACS_HPP
#define VVM_DYNAMICS_GENERALIZED_TAKACS_HPP

#include "dynamics/operators/GeneralizedBuoyancy.hpp"
#include "dynamics/operators/GeneralizedScalarTransport.hpp"
#include "dynamics/operators/GeneralizedVorticityTendency.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"

namespace VVM {
namespace Dynamics {

// Host-side output/validation policy. Weights multiply only the new
// canonical increment; they are NOT a general vector-basis transform.
// A nonorthogonal physical conversion needs both staggered components.
// The optional validator must perform host-only checks without allocating
// device memory or synchronizing a stream during CUDA graph capture.
struct GeneralizedTakacsBoundary {
    using Weight = Core::Geometry::GeometryField2D;
    using Validator = void (*)(const Core::State&, const Core::Grid&, const Core::Parameters&);

    Weight xi_weight = Weight::constant_value(real(1.0));
    Weight eta_weight = Weight::constant_value(real(1.0));
    Validator validate_vorticity = nullptr;
};

// Assembly of generalized scalar, vorticity and buoyancy operators for a
// stationary horizontal chart with unchanged physical z. No geometry-kind
// dispatch or RLL field names belong here. Caller owns topology and halos.
//
// Default horizontal output: xi_con=omega^1, eta_con=-omega^2 tendencies.
// zeta remains omega^3. A physical-accumulator caller must explicitly supply
// a valid output boundary. Factory-selected Cartesian production continues
// to use legacy Takacs, not this class.
//
// Construct using grid.geometry(); that geometry object must outlive the
// scheme. Construction and backend preparation must precede graph capture.
class GeneralizedTakacs final : public SpatialScheme {
public:
    enum class BuoyancyMode { Disabled, Dry, Moist };

    explicit GeneralizedTakacs(const Core::Geometry::HorizontalGeometry& geometry,
        BuoyancyMode buoyancy_mode = BuoyancyMode::Disabled,
        const GeneralizedTakacsBoundary& boundary = {});

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
    void validate_grid(const Core::Grid& grid) const;

    void validate_dry_buoyancy(
        const Core::State& state, const Core::Grid& grid, const Core::Parameters& params) const;

    void add_vorticity_tendency(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output,
        const std::string& variable,
        Operators::GeneralizedVorticityTendency::Term term) const;

    void add_buoyancy_tendency(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output,
        bool xi_component) const;

    const Core::Geometry::HorizontalGeometry* geometry_;
    GeneralizedTakacsBoundary boundary_;
    BuoyancyMode buoyancy_mode_;

    Operators::GeneralizedScalarTransport scalar_transport_;
    Operators::GeneralizedBuoyancy buoyancy_;
    Operators::GeneralizedVorticityTendency vorticity_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_GENERALIZED_TAKACS_HPP
