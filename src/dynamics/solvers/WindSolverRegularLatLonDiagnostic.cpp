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

void
WindSolver::prepare_regular_latlon_diagnostic_execution() {
    VerticalEllipticSolver::prepare_execution();
    prepare_horizontal_diagnostic_execution();

#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::parallel_for("PrepareWindSolverRegularLatLonDiagnostic",
        Kokkos::RangePolicy<Kokkos::Cuda>(0, 1),
        KOKKOS_LAMBDA(const int){});

    Kokkos::Cuda().fence("Prepare WindSolver regular latitude-longitude diagnostic");

    const auto result = cudaGetLastError();
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
#endif
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
        throw std::invalid_argument(
            "Regular latitude-longitude wind diagnostic requires RegularLatLon geometry.");
    }

    const auto& horizontal = grid.horizontal_specification();

    if (horizontal.topology.q1 != Core::HorizontalEdgeTopology::Periodic) {
        throw std::invalid_argument(
            "Regular latitude-longitude wind diagnostic requires periodic q1.");
    }

    const bool periodic_boundary =
        options.boundary_policy == HorizontalDiagnosticBoundaryPolicy::RegularLatLonPeriodic;
    if ((horizontal.topology.q2 == Core::HorizontalEdgeTopology::Periodic) != periodic_boundary) {
        throw std::invalid_argument(
            "Regular latitude-longitude wind diagnostic requires bounded q2.");
    }

    const bool reference_boundary =
        options.boundary_policy == HorizontalDiagnosticBoundaryPolicy::CvvmMode2Reference;

    const bool free_slip_boundary =
        options.boundary_policy == HorizontalDiagnosticBoundaryPolicy::RegularLatLonFreeSlipChannel;

    if (!reference_boundary && !free_slip_boundary && !periodic_boundary) {
        throw std::invalid_argument("Unknown regular latitude-longitude boundary policy.");
    }

    if (options.vertical_iterations <= 0) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic requires positive "
                                    "fixed vertical iterations.");
    }

    if (!std::isfinite(options.inverse_dz) || options.inverse_dz <= real(0.0)) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic requires a finite "
                                    "positive inverse reference dz.");
    }

    if (options.horizontal.iterations <= 0 || !std::isfinite(options.horizontal.diagonal_shift) ||
        options.horizontal.diagonal_shift < real(0.0) ||
        !options.horizontal.refresh_initial_halos) {
        throw std::invalid_argument("Regular latitude-longitude wind diagnostic requires positive "
                                    "fixed horizontal iterations, "
                                    "a nonnegative shift, and initial halo refresh.");
    }

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const int bottom = h - 1;
    const int top = nz - h - 1;

    if (h < 1 || top <= bottom || top + 1 >= nz) {
        throw std::invalid_argument(
            "Regular latitude-longitude wind diagnostic received an invalid vertical layout.");
    }

    const std::array<const Core::Field<3>*, 7> volumes = {&fields.zeta,
        &fields.w,
        &fields.w_previous,
        &fields.xi,
        &fields.eta,
        &fields.u,
        &fields.v};

    for (const auto* field : volumes) {
        const auto& data = field->get_device_data();

        if (static_cast<int>(data.extent(0)) != nz || static_cast<int>(data.extent(1)) != ny ||
            static_cast<int>(data.extent(2)) != nx) {
            throw std::invalid_argument(
                "Regular latitude-longitude diagnostic volume extent mismatch.");
        }
    }

    if (static_cast<int>(fields.spacing.get_device_data().extent(0)) <= top) {
        throw std::invalid_argument(
            "Regular latitude-longitude diagnostic has insufficient spacing entries.");
    }

    std::unique_ptr<Core::Boundary::HorizontalBoundaryStencils> boundary;
    if (!periodic_boundary) {
        boundary = std::make_unique<Core::Boundary::HorizontalBoundaryStencils>(grid);
    }

    vertical_solver.solve(fields.xi,
        fields.eta,
        fields.w,
        fields.w_previous,
        options.vertical_iterations);

    if (free_slip_boundary) {
        halo.exchange_multiple_halos(std::vector<Core::Field<3>*>{&fields.w, &fields.w_previous});

        boundary->fill_centered_q2_neumann_halos(fields.w);
        boundary->fill_centered_q2_neumann_halos(fields.w_previous);
    }

    const auto operation = make_vertical_wind_diagnostic_device_view(grid.geometry());

    const auto xi = fields.xi.get_device_data();
    const auto eta = fields.eta.get_device_data();
    const auto spacing = fields.spacing.get_device_data();
    const auto zeta = fields.zeta.get_mutable_device_data();

    const bool owns_north_wall = grid.get_local_physical_end_y() == grid.get_global_points_y() - 1;

    const int zeta_end_j = free_slip_boundary && owns_north_wall ? ny - h - 1 : ny - h;

    const auto zeta_policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {zeta_end_j, nx - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("DiagnoseRegularLatLonZetaColumn",
        zeta_policy,
        KOKKOS_LAMBDA(const int j, const int i) {
            operation.integrate_zeta_column(xi, eta, spacing, zeta, bottom, top, j, i, true);
        });

    halo.exchange_halos(fields.zeta);

    if (free_slip_boundary) {
        boundary->fill_positive_face_q2_homogeneous_dirichlet_halos(fields.zeta);
    }
    else if (reference_boundary) {
        boundary->fill_constant_q2_halos(fields.zeta);
    }

    const HorizontalDiagnosticFields horizontal_fields{fields.psi,
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
        fields.zonal_covariant_increment};

    diagnose_horizontal_wind(grid,
        halo,
        horizontal_solver,
        horizontal_fields,
        workspace,
        options.horizontal,
        options.inverse_dz,
        bottom,
        top,
        options.boundary_policy);

    halo.exchange_multiple_halos(std::vector<Core::Field<3>*>{&fields.u, &fields.v});

    if (free_slip_boundary) {
        boundary->fill_regular_lat_lon_free_slip_physical_wind_halos(fields.u, fields.v);
    }
    else if (reference_boundary) {
        boundary->fill_constant_q2_halos(fields.u);
        boundary->fill_constant_q2_halos(fields.v);
    }
}

void
WindSolver::diagnose_horizontal_potentials(const Core::Grid& grid,
    Core::HaloExchanger& halo,
    HorizontalEllipticSolver& solver,
    const HorizontalDiagnosticFields& fields,
    const HorizontalDiagnosticWorkspace& workspace,
    const HorizontalEllipticSolver::Options& options,
    const Real inverse_dz,
    const int top,
    const HorizontalDiagnosticBoundaryPolicy boundary_policy) {

    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int h = grid.get_halo_cells();

    const auto& horizontal = grid.horizontal_specification();

    const bool free_slip_boundary =
        boundary_policy == HorizontalDiagnosticBoundaryPolicy::RegularLatLonFreeSlipChannel;

    const auto zeta = fields.zeta.get_device_data();
    const auto w = fields.w.get_device_data();

    const auto rho = fields.rhobar.get_device_data();
    const auto rho_up = fields.rhobar_up.get_device_data();
    const auto flex = fields.flex_mid.get_device_data();
    const auto rhs_psi = workspace.rhs_psi.get_mutable_device_data();
    const auto rhs_chi = workspace.rhs_chi.get_mutable_device_data();

    // HorizontalEllipticSolver supplies J internally,
    // exactly once.
    Kokkos::parallel_for("BuildHorizontalDiagnosticRHS",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            rhs_psi(j, i) = zeta(top, j, i);
            rhs_chi(j, i) = flex(top) * rho_up(top - 1) * w(top - 1, j, i) * inverse_dz / rho(top);
        });

    solver.make_extrapolated_guess(fields.psi, fields.psi_previous, workspace.solution_psi);

    solver.make_extrapolated_guess(fields.chi, fields.chi_previous, workspace.solution_chi);

    if (free_slip_boundary) {
        solver.solve_regular_lat_lon_channel_at_z_and_t(workspace.rhs_psi,
            workspace.solution_psi,
            workspace.rhs_chi,
            workspace.solution_chi,
            options);
    }
    else {
        solver.solve_at_z_and_t(workspace.rhs_psi,
            workspace.solution_psi,
            workspace.rhs_chi,
            workspace.solution_chi,
            options);
    }

    const auto psi = fields.psi.get_mutable_device_data();
    const auto chi = fields.chi.get_mutable_device_data();
    const auto psi_previous = fields.psi_previous.get_mutable_device_data();
    const auto chi_previous = fields.chi_previous.get_mutable_device_data();
    const auto solved_psi = workspace.solution_psi.get_device_data();
    const auto solved_chi = workspace.solution_chi.get_device_data();

    Kokkos::parallel_for("CommitHorizontalPotentialHistory",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            psi_previous(j, i) = psi(j, i);
            chi_previous(j, i) = chi(j, i);
            psi(j, i) = solved_psi(j, i);
            chi(j, i) = solved_chi(j, i);
        });

    halo.exchange_multiple_halos(std::vector<Core::Field<2>*>{&fields.psi,
        &fields.chi,
        &fields.psi_previous,
        &fields.chi_previous});

    if (horizontal.ny > 1 && horizontal.topology.q2 == Core::HorizontalEdgeTopology::Bounded) {
        Core::Boundary::HorizontalBoundaryStencils boundary(grid);

        if (free_slip_boundary) {
            boundary.fill_positive_face_q2_dirichlet_halos(fields.psi,
                options.channel_psi_south,
                options.channel_psi_north);
            boundary.fill_positive_face_q2_dirichlet_halos(fields.psi_previous,
                options.channel_psi_south,
                options.channel_psi_north);

            boundary.fill_centered_q2_neumann_halos(fields.chi);
            boundary.fill_centered_q2_neumann_halos(fields.chi_previous);
        }
        else {
            boundary.fill_constant_q2_halos(fields.psi);
            boundary.fill_constant_q2_halos(fields.chi);
            boundary.fill_constant_q2_halos(fields.psi_previous);
            boundary.fill_constant_q2_halos(fields.chi_previous);
        }
    }
}

} // namespace Dynamics
} // namespace VVM
