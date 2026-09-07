#include "dynamics/solvers/WindSolver.hpp"

#include "core/geometry/GeometryKind.hpp"
#include "dynamics/solvers/VerticalEllipticSolver.hpp"
#include "dynamics/solvers/VerticalWindDiagnostic.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace VVM {
namespace Dynamics {

void WindSolver::prepare_regular_latlon_diagnostic_execution() {
    VerticalEllipticSolver::prepare_execution();
    prepare_horizontal_diagnostic_execution();

#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::parallel_for("PrepareWindSolverRegularLatLonDiagnostic",
        Kokkos::RangePolicy<Kokkos::Cuda>(0, 1),
        KOKKOS_LAMBDA(const int) {});

    Kokkos::Cuda().fence("Prepare WindSolver regular latitude-longitude diagnostic");

    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
#endif
}

void WindSolver::diagnose_regular_latlon_wind(const Core::Grid& grid, Core::HaloExchanger& halo,
    VerticalEllipticSolver& vertical_solver, HorizontalEllipticSolver& horizontal_solver,
    const RegularLatLonDiagnosticFields& fields, const HorizontalDiagnosticWorkspace& workspace,
    const RegularLatLonDiagnosticOptions& options) {

    if (grid.geometry().kind() != Core::Geometry::GeometryKind::RegularLatLon) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic requires RegularLatLon geometry.");
    }

    const auto& horizontal = grid.horizontal_specification();

    if (horizontal.topology.q1 != Core::HorizontalEdgeTopology::Periodic) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic requires periodic q1.");
    }

    if (horizontal.topology.q2 != Core::HorizontalEdgeTopology::Bounded) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic requires bounded q2.");
    }

    if (options.vertical_iterations <= 0) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic requires positive fixed vertical iterations.");
    }

    if (!std::isfinite(options.inverse_dz) || options.inverse_dz <= real(0.0)) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic requires a finite positive inverse reference dz.");
    }

    if (options.horizontal.iterations <= 0 ||
        !std::isfinite(options.horizontal.diagonal_shift) ||
        options.horizontal.diagonal_shift < real(0.0) ||
        !options.horizontal.refresh_initial_halos) {

        throw std::invalid_argument(
            "Regular latitude-longitude wind diagnostic requires positive fixed horizontal iterations, "
            "a nonnegative shift, and initial halo refresh.");
    }

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const int bottom = h - 1;
    const int top = nz - h - 1;

    if (h < 1 || top <= bottom || top + 1 >= nz) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic received an invalid vertical layout.");
    }

    const std::array<const Core::Field<3>*, 7> volumes = {
        &fields.zeta, &fields.w, &fields.w_previous, &fields.xi,
        &fields.eta, &fields.u, &fields.v
    };

    for (const auto* field : volumes) {
        const auto& data = field->get_device_data();

        if (static_cast<int>(data.extent(0)) != nz ||
            static_cast<int>(data.extent(1)) != ny ||
            static_cast<int>(data.extent(2)) != nx) {

            throw std::invalid_argument("Regular latitude-longitude diagnostic volume extent mismatch.");
        }
    }

    if (static_cast<int>(fields.spacing.get_device_data().extent(0)) <= top) {
        throw std::invalid_argument("Regular latitude-longitude diagnostic has insufficient spacing entries.");
    }

    vertical_solver.solve(
        fields.xi,
        fields.eta,
        fields.w,
        fields.w_previous,
        options.vertical_iterations);

    const auto operation =
        make_vertical_wind_diagnostic_device_view(grid.geometry());

    const auto xi = fields.xi.get_device_data();
    const auto eta = fields.eta.get_device_data();
    const auto spacing = fields.spacing.get_device_data();
    const auto zeta = fields.zeta.get_mutable_device_data();

    const auto zeta_policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {h, h},
            {ny - h, nx - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for(
        "DiagnoseRegularLatLonZetaColumn",
        zeta_policy,
        KOKKOS_LAMBDA(const int j, const int i) {
            operation.integrate_zeta_column(
                xi,
                eta,
                spacing,
                zeta,
                bottom,
                top,
                j,
                i,
                true);
        });

    halo.exchange_halos(fields.zeta);

    Core::Boundary::HorizontalBoundaryStencils boundary(grid);
    boundary.fill_constant_q2_halos(fields.zeta);

    const HorizontalDiagnosticFields horizontal_fields{
        fields.psi,
        fields.psi_previous,
        fields.chi,
        fields.chi_previous,
        fields.zeta,
        fields.w,
        fields.xi,
        fields.eta,
        fields.u,
        fields.v,
        fields.rhobar,
        fields.rhobar_up,
        fields.flex_mid,
        fields.spacing,
        fields.zonal_covariant_increment
    };

    diagnose_horizontal_wind(
        grid,
        halo,
        horizontal_solver,
        horizontal_fields,
        workspace,
        options.horizontal,
        options.inverse_dz,
        bottom,
        top);

    halo.exchange_multiple_halos(
        std::vector<Core::Field<3>*>{&fields.u, &fields.v});

    // This deliberately retains the currently qualified CVVM MODE=2 reference
    // stencil. It is not presented as a complete free-slip physical boundary.
    boundary.fill_constant_q2_halos(fields.u);
    boundary.fill_constant_q2_halos(fields.v);
}

} // namespace Dynamics
} // namespace VVM
