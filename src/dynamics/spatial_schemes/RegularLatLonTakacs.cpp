#include "dynamics/spatial_schemes/RegularLatLonTakacs.hpp"
#include "dynamics/operators/RegularLatLonVorticityTendencyBoundary.hpp"

#include <stdexcept>
#include <string>

#include "core/geometry/GeometryKind.hpp"

namespace VVM {
namespace Dynamics {

RegularLatLonTakacs::RegularLatLonTakacs(const Core::Geometry::HorizontalGeometry& geometry,
    const bool enable_dry_buoyancy,
    const bool enable_moist_buoyancy)
    : scalar_transport_(geometry), buoyancy_(geometry), vorticity_(geometry),
      enable_dry_buoyancy_(enable_dry_buoyancy), enable_moist_buoyancy_(enable_moist_buoyancy) {
    if (geometry.kind() != Core::Geometry::GeometryKind::RegularLatLon) {
        throw std::invalid_argument("RegularLatLonTakacs requires RLL geometry.");
    }

    // Numerical schemes are constructed during model initialization, before
    // the time-integrator graph is captured. Prepare the exact operator
    // launch functors without evaluating a model tendency.
    Operators::GeneralizedScalarTransport::prepare_execution();

    if (enable_dry_buoyancy_ || enable_moist_buoyancy_) {
        Operators::GeneralizedBuoyancy::prepare_execution();
    }
}

void
RegularLatLonTakacs::add_vorticity_tendency(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable,
    Operators::GeneralizedVorticityTendency::Term term) const {
    Operators::add_regular_lat_lon_physical_vorticity_tendency(vorticity_,
        state,
        grid,
        params,
        output,
        variable,
        term);
}

void
RegularLatLonTakacs::calculate_advection_tendency(const Core::State& state,
    const Core::Field<3>& scalar,
    const Core::Field<3>& contravariant_mass_flux_q1,
    const Core::Field<3>& contravariant_mass_flux_q2,
    const Core::Field<3>& vertical_mass_flux,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency,
    const std::string& var_name,
    const VVM::Real stage_dt) const {

    (void)stage_dt;

    if (grid.geometry().kind() != Core::Geometry::GeometryKind::RegularLatLon) {

        throw std::logic_error("RegularLatLonTakacs received a non-RLL Grid.");
    }

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

        throw std::runtime_error("RegularLatLonTakacs currently supports only "
                                 "potential-temperature, moisture and passive-tracer advection; "
                                 "field '" +
                                 var_name + "' is not supported.");
    }

    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();

    scalar_transport_.add_flux_convergence(scalar,
        contravariant_mass_flux_q1,
        contravariant_mass_flux_q2,
        vertical_mass_flux,
        params.dz_mid,
        out_tendency,
        h,
        nz - h);
}

void
RegularLatLonTakacs::validate_dry_buoyancy(
    const Core::State& state, const Core::Grid& grid, const Core::Parameters& params) const {

    if (grid.geometry().kind() != Core::Geometry::GeometryKind::RegularLatLon) {

        throw std::logic_error("RegularLatLonTakacs received a non-RLL Grid.");
    }

    if (!enable_dry_buoyancy_) {
        throw std::runtime_error("RegularLatLonTakacs dry buoyancy was not explicitly enabled.");
    }

    if (state.has_field("qp") || state.has_field("qc") || state.has_field("qr") ||
        state.has_field("qi") || state.has_field("qm") || state.has_field("nc") ||
        state.has_field("nr") || state.has_field("ni") || state.has_field("bm")) {

        throw std::runtime_error(
            "RegularLatLonTakacs dry buoyancy does not support moist State fields.");
    }

    // Initializer stores maximum(topo) + h. Require its initialized
    // flat-terrain value; do not interpret terrain masks in this operator.
    if (params.max_topo_idx != grid.get_halo_cells()) {
        throw std::runtime_error(
            "RegularLatLonTakacs dry buoyancy requires initialized flat terrain.");
    }
}

void
RegularLatLonTakacs::calculate_stretching_tendency_x(const Core::State& state,
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
RegularLatLonTakacs::calculate_stretching_tendency_y(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    calculate_stretching_tendency_x(state, grid, params, output, variable);
}
void
RegularLatLonTakacs::calculate_stretching_tendency_z(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    calculate_stretching_tendency_x(state, grid, params, output, variable);
}
void
RegularLatLonTakacs::calculate_twisting_tendency_x(const Core::State& state,
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
RegularLatLonTakacs::calculate_twisting_tendency_y(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    calculate_twisting_tendency_x(state, grid, params, output, variable);
}
void
RegularLatLonTakacs::calculate_twisting_tendency_z(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable) const {
    calculate_twisting_tendency_x(state, grid, params, output, variable);
}

void
RegularLatLonTakacs::add_buoyancy_tendency(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    bool xi_component) const {
    using Core::Geometry::GeometryKind;
    using Core::Geometry::HorizontalLocation;

    if (grid.geometry().kind() != GeometryKind::RegularLatLon) {
        throw std::logic_error("RLL buoyancy output conversion requires RLL geometry.");
    }

    if (!enable_moist_buoyancy_) {
        validate_dry_buoyancy(state, grid, params);
    }

    const int h = grid.get_halo_cells();
    const int end = grid.get_local_total_points_z() - h - 1;

    const auto location = xi_component ? HorizontalLocation::V : HorizontalLocation::U;

    const auto geometry = grid.geometry().device_view(location);

    const auto weight = xi_component ? geometry.contravariant_to_physical.a11
                                     : geometry.contravariant_to_physical.a22;

    // xi/eta and xi_con/eta_con use the same VVM sign convention. Only a
    // positive RLL scale is needed here. Weight the new increment, never
    // rescale contributions already present in the physical accumulator.
    if (enable_moist_buoyancy_) {
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
            xi_component,
            weight);
    }
    else if (xi_component) {
        buoyancy_.add_xi_tendency(state.get_field<3>("th"),
            state.get_field<1>("thbar"),
            params.gravity,
            output,
            h,
            end,
            weight);
    }
    else {
        buoyancy_.add_eta_tendency(state.get_field<3>("th"),
            state.get_field<1>("thbar"),
            params.gravity,
            output,
            h,
            end,
            weight);
    }
}

void
RegularLatLonTakacs::calculate_buoyancy_tendency_x(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output) const {
    add_buoyancy_tendency(state, grid, params, output, true);
}

void
RegularLatLonTakacs::calculate_buoyancy_tendency_y(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output) const {
    add_buoyancy_tendency(state, grid, params, output, false);
}

} // namespace Dynamics
} // namespace VVM
