#include "core/geometry/GeometryKind.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "dynamics/operators/HorizontalVectorConversion.hpp"
#include "dynamics/solvers/GeneralizedWindDiagnostic.hpp"
#include "dynamics/solvers/WindSolver.hpp"

#include <stdexcept>
#include <vector>

namespace VVM {
namespace Dynamics {

namespace {

void
commit_regular_latlon_covariant_wind_to_physical(const Core::Grid& grid,
    const Core::Field<3>& covariant_q1,
    const Core::Field<3>& covariant_q2,
    Core::Field<3>& u,
    Core::Field<3>& v,
    const int bottom,
    const int top) {

    const int h = grid.get_halo_cells();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const auto inverse_h1_at_u = grid.geometry()
                                     .device_view(Core::Geometry::HorizontalLocation::U)
                                     .physical_to_contravariant.a11;

    const auto inverse_h2_at_v = grid.geometry()
                                     .device_view(Core::Geometry::HorizontalLocation::V)
                                     .physical_to_contravariant.a22;

    const auto q1 = covariant_q1.get_device_data();

    const auto q2 = covariant_q2.get_device_data();

    auto physical_u = u.get_mutable_device_data();

    auto physical_v = v.get_mutable_device_data();

    Kokkos::parallel_for("CommitRegularLatLonPhysicalWindFromCovariant",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            physical_u(k, j, i) =
                Operators::HorizontalVectorConversion::covariant_to_physical(q1(k, j, i),
                    inverse_h1_at_u(j, i));

            physical_v(k, j, i) =
                Operators::HorizontalVectorConversion::covariant_to_physical(q2(k, j, i),
                    inverse_h2_at_v(j, i));
        });
}

} // namespace

void
WindSolver::prepare_regular_latlon_diagnostic_execution() {

    GeneralizedWindDiagnostic::prepare_execution();
}

void
WindSolver::diagnose_regular_latlon_wind(const Core::Grid& grid,
    Core::HaloExchanger& halo,
    VerticalEllipticSolver& vertical_solver,
    HorizontalEllipticSolver& horizontal_solver,
    const RegularLatLonDiagnosticFields& fields,
    const HorizontalDiagnosticWorkspace& workspace,
    const RegularLatLonDiagnosticOptions& options) {

    if (grid.geometry().kind() != Core::Geometry::GeometryKind::RegularLatLon) {

        throw std::invalid_argument("Regular latitude-longitude wind wrapper "
                                    "requires RegularLatLon geometry.");
    }

    const GeneralizedWindDiagnosticFields generalized_fields{fields.psi,
        fields.psi_previous,
        fields.chi,
        fields.chi_previous,

        fields.zeta,
        fields.w,
        fields.w_previous,

        fields.xi_con,
        fields.eta_con,

        fields.covariant_q1_wind,
        fields.covariant_q2_wind,

        fields.rhobar,
        fields.rhobar_up,
        fields.flex_mid,
        fields.spacing,

        fields.zonal_covariant_increment};

    GeneralizedWindDiagnostic::diagnose(grid,
        halo,
        vertical_solver,
        horizontal_solver,
        generalized_fields,
        workspace,
        options);

    const int h = grid.get_halo_cells();

    const int nz = grid.get_local_total_points_z();

    const int bottom = h - 1;

    const int top = nz - h - 1;

    // Representation boundary:
    //
    //     generalized covariant wind
    //              ↓
    //     RLL physical east/north wind
    //
    // This conversion intentionally remains outside
    // GeneralizedWindDiagnostic.
    commit_regular_latlon_covariant_wind_to_physical(grid,
        fields.covariant_q1_wind,
        fields.covariant_q2_wind,
        fields.u,
        fields.v,
        bottom,
        top);

    halo.exchange_multiple_halos(std::vector<Core::Field<3>*>{&fields.u, &fields.v});

    const bool free_slip_boundary =
        options.boundary_policy == HorizontalDiagnosticBoundaryPolicy::FreeSlipBoundedQ2;

    const bool reference_boundary =
        options.boundary_policy == HorizontalDiagnosticBoundaryPolicy::CvvmMode2Reference;

    if (free_slip_boundary || reference_boundary) {

        Core::Boundary::HorizontalBoundaryStencils boundary(grid);

        if (free_slip_boundary) {
            boundary.fill_regular_lat_lon_free_slip_physical_wind_halos(fields.u, fields.v);
        }
        else {
            boundary.fill_constant_q2_halos(fields.u);

            boundary.fill_constant_q2_halos(fields.v);
        }
    }
}

} // namespace Dynamics
} // namespace VVM
