#include "dynamics/spatial_schemes/RegularLatLonTakacs.hpp"

#include <stdexcept>
#include <string>

#include "core/geometry/GeometryKind.hpp"

namespace VVM {
namespace Dynamics {

RegularLatLonTakacs::RegularLatLonTakacs(
    const Core::Geometry::HorizontalGeometry& geometry)
    : scalar_transport_(geometry) {

    // Numerical schemes are constructed during model initialization, before
    // the time-integrator graph is captured. Preparing here guarantees that
    // the first RLL scalar tendency launch can safely occur inside capture.
    Operators::RegularLatLonScalarTransport::
        prepare_execution();
}

void RegularLatLonTakacs::calculate_advection_tendency(
    const Core::State& state,
    const Core::Field<3>& scalar,
    const Core::Field<3>& physical_mass_flux_q1,
    const Core::Field<3>& physical_mass_flux_q2,
    const Core::Field<3>& vertical_mass_flux,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency,
    const std::string& var_name,
    const VVM::Real stage_dt) const {

    (void)stage_dt;

    if (grid.geometry().kind()
        != Core::Geometry::GeometryKind::RegularLatLon) {

        throw std::logic_error(
            "RegularLatLonTakacs received a non-RLL Grid.");
    }

    if (var_name != "th"
        && !state.is_tracer(var_name)) {

        throw std::runtime_error(
            "RegularLatLonTakacs currently supports only "
            "potential-temperature and passive-tracer advection; "
            "field '"
            + var_name
            + "' is not supported.");
    }

    const int h = grid.get_halo_cells();
    const int nz =
        grid.get_local_total_points_z();

    scalar_transport_.add_flux_convergence(
        scalar,
        physical_mass_flux_q1,
        physical_mass_flux_q2,
        vertical_mass_flux,
        params.dz_mid,
        out_tendency,
        h,
        nz - h);
}

} // namespace Dynamics
} // namespace VVM
