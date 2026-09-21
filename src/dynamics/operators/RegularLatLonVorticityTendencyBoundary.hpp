#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_BOUNDARY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_BOUNDARY_HPP

#include <stdexcept>

#include "dynamics/operators/GeneralizedVorticityTendency.hpp"

namespace VVM::Dynamics::Operators {

// Temporary physical-accumulator boundary, NOT a numerical backend.
// RLL permits independent scaling at the two native locations:
//     physical xi  = h1_at_v * xi_con
//     physical eta = h2_at_u * eta_con
// Both eta representations already have the VVM minus sign.
// Do not reuse this diagonal conversion for nonorthogonal coordinates.
inline void
add_regular_lat_lon_physical_vorticity_tendency(const GeneralizedVorticityTendency& tendency,
    const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable,
    GeneralizedVorticityTendency::Term term) {
    using Core::Geometry::GeometryField2D;
    using Core::Geometry::GeometryKind;
    using Core::Geometry::HorizontalLocation;

    if (grid.geometry().kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument("The RLL physical-vorticity boundary requires RLL geometry.");
    }

    // Preserve the existing RLL terrain-initialization guard outside the
    // generalized orchestrator. Mask application remains with the caller.
    const int h = grid.get_halo_cells();
    const int top = grid.get_local_total_points_z() - h - 1;
    const bool terrain = state.has_field("rll_terrain_height");

    if ((!terrain && params.max_topo_idx != h) ||
        (terrain && (params.max_topo_idx < h || params.max_topo_idx >= top))) {
        throw std::invalid_argument(
            "RLL vorticity tendencies require flat or initialized RLL terrain below the lid.");
    }

    auto weight = GeometryField2D::constant_value(real(1.0));

    if (variable == "xi") {
        weight = grid.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a11;
    }
    else if (variable == "eta") {
        weight = grid.geometry().device_view(HorizontalLocation::U).contravariant_to_physical.a22;
    }

    // Weight the new increment only. Never rescale previously accumulated
    // buoyancy, diffusion, turbulence or other physical contributions.
    tendency.add_weighted_from_canonical_state(state, grid, params, output, variable, term, weight);
}

} // namespace VVM::Dynamics::Operators
#endif
