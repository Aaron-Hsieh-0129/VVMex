#include "dynamics/spatial_schemes/GeneralizedTakacs.hpp"

#include <stdexcept>

namespace VVM::Dynamics {

GeneralizedTakacs::GeneralizedTakacs(const Core::Geometry::HorizontalGeometry& geometry,
    BuoyancyMode buoyancy_mode,
    const GeneralizedTakacsBoundary& boundary)
    : geometry_(&geometry), boundary_(boundary), buoyancy_mode_(buoyancy_mode),
      scalar_transport_(geometry), buoyancy_(geometry), vorticity_(geometry) {
    if (buoyancy_mode_ != BuoyancyMode::Disabled && buoyancy_mode_ != BuoyancyMode::Dry &&
        buoyancy_mode_ != BuoyancyMode::Moist) {
        throw std::invalid_argument("Invalid generalized buoyancy mode.");
    }

    Operators::GeneralizedScalarTransport::prepare_execution();

    if (buoyancy_mode_ != BuoyancyMode::Disabled) {
        Operators::GeneralizedBuoyancy::prepare_execution();
    }

    // GeneralizedVorticityTendency prepares its kernels in its constructor.
}

void
GeneralizedTakacs::validate_grid(const Core::Grid& grid) const {
    if (&grid.geometry() != geometry_) {
        throw std::invalid_argument(
            "GeneralizedTakacs must use the Grid geometry supplied at construction.");
    }
}

void
GeneralizedTakacs::add_vorticity_tendency(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable,
    Operators::GeneralizedVorticityTendency::Term term) const {
    validate_grid(grid);

    if (boundary_.validate_vorticity != nullptr) {
        boundary_.validate_vorticity(state, grid, params);
    }

    vorticity_.add_from_canonical_state(state, grid, params, output, variable, term);
}

void
GeneralizedTakacs::calculate_advection_tendency(const Core::State& state,
    const Core::Field<3>& scalar,
    const Core::Field<3>& contravariant_mass_flux_q1,
    const Core::Field<3>& contravariant_mass_flux_q2,
    const Core::Field<3>& vertical_mass_flux,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency,
    const std::string& var_name,
    Real stage_dt) const {
    (void)stage_dt;
    validate_grid(grid);

    if (var_name == "xi" || var_name == "eta" || var_name == "zeta") {
        add_vorticity_tendency(state,
            grid,
            params,
            out_tendency,
            var_name,
            Operators::GeneralizedVorticityTendency::Term::Transport);
        return;
    }

    if (var_name != "th" && !is_moist_scalar(var_name) && !state.is_tracer(var_name)) {
        throw std::runtime_error(
            "GeneralizedTakacs does not support advection of field '" + var_name + "'.");
    }

    const int h = grid.get_halo_cells();

    scalar_transport_.add_flux_convergence(scalar,
        contravariant_mass_flux_q1,
        contravariant_mass_flux_q2,
        vertical_mass_flux,
        params.dz_mid,
        out_tendency,
        h,
        grid.get_local_total_points_z() - h);
}

void
GeneralizedTakacs::validate_dry_buoyancy(
    const Core::State& state, const Core::Grid& grid, const Core::Parameters& params) const {
    for (const char* name : {"qp", "qc", "qr", "qi", "qm", "nc", "nr", "ni", "bm"}) {
        if (state.has_field(name)) {
            throw std::runtime_error(
                "GeneralizedTakacs dry buoyancy does not support moist State fields.");
        }
    }

    // Retain the existing dry-mode capability restriction. Moist mode
    // retains its native face-mask behavior inside GeneralizedBuoyancy.
    if (params.max_topo_idx != grid.get_halo_cells()) {
        throw std::runtime_error(
            "GeneralizedTakacs dry buoyancy requires initialized flat terrain.");
    }
}

void
GeneralizedTakacs::add_buoyancy_tendency(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    bool xi_component) const {
    validate_grid(grid);

    if (buoyancy_mode_ == BuoyancyMode::Disabled) {
        throw std::runtime_error("GeneralizedTakacs buoyancy was not enabled.");
    }

    if (buoyancy_mode_ == BuoyancyMode::Dry) {
        validate_dry_buoyancy(state, grid, params);
    }

    const int h = grid.get_halo_cells();
    const int end = grid.get_local_total_points_z() - h - 1;

    // The operator's default is canonical xi_con/eta_con output.
    // The moist face mask still clears the complete accumulated tendency
    // at masked points. Removing physical weights does not change that rule.
    if (buoyancy_mode_ == BuoyancyMode::Moist) {
        buoyancy_.add_moist_tendency(state.get_field<3>("th"),
            state.get_field<1>("thbar"),
            params.gravity,
            state.get_field<3>("qv"),
            state.get_field<3>("qp"),
            state.get_field<3>(xi_component ? "ITYPEV" : "ITYPEU"),
            output,
            h,
            end,
            params.max_topo_idx,
            xi_component);
    }
    else if (xi_component) {
        buoyancy_.add_xi_tendency(state.get_field<3>("th"),
            state.get_field<1>("thbar"),
            params.gravity,
            output,
            h,
            end);
    }
    else {
        buoyancy_.add_eta_tendency(state.get_field<3>("th"),
            state.get_field<1>("thbar"),
            params.gravity,
            output,
            h,
            end);
    }
}

void
GeneralizedTakacs::calculate_buoyancy_tendency_x(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output) const {
    add_buoyancy_tendency(state, grid, params, output, true);
}

void
GeneralizedTakacs::calculate_buoyancy_tendency_y(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output) const {
    add_buoyancy_tendency(state, grid, params, output, false);
}

void
GeneralizedTakacs::calculate_stretching_tendency_x(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    add_vorticity_tendency(state,
        grid,
        params,
        output,
        variable,
        Operators::GeneralizedVorticityTendency::Term::Stretching);
}

void
GeneralizedTakacs::calculate_stretching_tendency_y(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    calculate_stretching_tendency_x(state, grid, params, output, variable);
}

void
GeneralizedTakacs::calculate_stretching_tendency_z(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    calculate_stretching_tendency_x(state, grid, params, output, variable);
}

void
GeneralizedTakacs::calculate_twisting_tendency_x(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    add_vorticity_tendency(state,
        grid,
        params,
        output,
        variable,
        Operators::GeneralizedVorticityTendency::Term::Twisting);
}

void
GeneralizedTakacs::calculate_twisting_tendency_y(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    calculate_twisting_tendency_x(state, grid, params, output, variable);
}

void
GeneralizedTakacs::calculate_twisting_tendency_z(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    calculate_twisting_tendency_x(state, grid, params, output, variable);
}

} // namespace VVM::Dynamics
