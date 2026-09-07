#include "dynamics/solvers/WindSolver.hpp"
#include "dynamics/solvers/HorizontalWindStateAdapter.hpp"
#include "core/geometry/HorizontalLocation.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace VVM {
namespace Dynamics {

void WindSolver::prepare_horizontal_diagnostic_execution() {
    // This checks capture state before doing any backend preparation.
    HorizontalWindStateAdapter::prepare_execution();

#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::parallel_for("PrepareWindSolverHorizontalDiagnostic",
        Kokkos::RangePolicy<Kokkos::Cuda>(0, 1), KOKKOS_LAMBDA(const int) {});

    Kokkos::Cuda().fence("Prepare WindSolver horizontal diagnostic");

    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
#endif
}

void WindSolver::diagnose_horizontal_wind(const Core::Grid& grid, Core::HaloExchanger& halo,
    HorizontalEllipticSolver& solver, const HorizontalDiagnosticFields& fields,
    const HorizontalDiagnosticWorkspace& workspace, const HorizontalEllipticSolver::Options& options,
    Real inverse_dz, int bottom, int top) {

    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int nz = static_cast<int>(fields.w.get_device_data().extent(0));
    const int h = grid.get_halo_cells();
    const auto& horizontal = grid.horizontal_specification();

    if (bottom < 0 || top < 1 || top < bottom || top >= nz) {
        throw std::invalid_argument("Invalid horizontal diagnostic levels.");
    }

    if (!std::isfinite(inverse_dz) || inverse_dz <= real(0.0)) {
        throw std::invalid_argument("Invalid inverse reference dz.");
    }

    if (options.iterations <= 0 || !std::isfinite(options.diagonal_shift)
        || options.diagonal_shift < real(0.0) || !options.refresh_initial_halos) {
        throw std::invalid_argument("Horizontal diagnostic requires positive fixed iterations, nonnegative shift, and initial halo refresh.");
    }

    if (horizontal.nx > 1 && horizontal.topology.q1 == Core::HorizontalEdgeTopology::Bounded) {
        throw std::invalid_argument("Bounded q1 is not supported.");
    }

    const HorizontalWindStateAdapter adapter(grid.geometry());

    const std::array<const Core::Field<2>*, 8> planes = {
        &fields.psi, &fields.psi_previous, &fields.chi, &fields.chi_previous,
        &workspace.rhs_psi, &workspace.rhs_chi, &workspace.solution_psi, &workspace.solution_chi
    };

    const std::array<const Core::Field<3>*, 6> volumes = {
        &fields.zeta, &fields.w, &fields.xi, &fields.eta, &fields.u, &fields.v
    };

    for (const auto* field : planes) {
        const auto& d = field->get_device_data();

        if (static_cast<int>(d.extent(0)) != ny || static_cast<int>(d.extent(1)) != nx) {
            throw std::invalid_argument("Incorrect diagnostic plane extents.");
        }
    }

    for (const auto* field : volumes) {
        const auto& d = field->get_device_data();

        if (static_cast<int>(d.extent(0)) != nz || static_cast<int>(d.extent(1)) != ny || static_cast<int>(d.extent(2)) != nx) {
            throw std::invalid_argument("Incorrect diagnostic volume extents.");
        }
    }

    for (const auto* field : {&fields.rhobar, &fields.rhobar_up, &fields.flex_mid}) {
        if (static_cast<int>(field->get_device_data().extent(0)) <= top) {
            throw std::invalid_argument("Insufficient diagnostic profile entries.");
        }
    }

    if (static_cast<int>(fields.spacing.get_device_data().extent(0)) < top) {
        throw std::invalid_argument("Insufficient diagnostic spacing entries.");
    }

    std::array<const Real*, 19> addresses = {};
    int count = 0;

    for (const auto* field : planes) addresses[count++] = field->get_device_data().data();
    for (const auto* field : volumes) addresses[count++] = field->get_device_data().data();

    addresses[count++] = fields.rhobar.get_device_data().data();
    addresses[count++] = fields.rhobar_up.get_device_data().data();
    addresses[count++] = fields.flex_mid.get_device_data().data();
    addresses[count++] = fields.spacing.get_device_data().data();
    addresses[count++] = fields.zonal_covariant_increment.get_device_data().data();

    for (int a = 0; a < count; ++a) {
        if (!addresses[a]) throw std::invalid_argument("Unallocated diagnostic storage.");

        for (int b = 0; b < a; ++b) {
            if (addresses[a] == addresses[b]) {
                throw std::invalid_argument("Diagnostic fields must have distinct storage.");
            }
        }
    }

    const auto zeta = fields.zeta.get_device_data();
    const auto w = fields.w.get_device_data();
    const auto rho = fields.rhobar.get_device_data();
    const auto rho_up = fields.rhobar_up.get_device_data();
    const auto flex = fields.flex_mid.get_device_data();
    const auto rhs_psi = workspace.rhs_psi.get_mutable_device_data();
    const auto rhs_chi = workspace.rhs_chi.get_mutable_device_data();

    // Physical RHS: HorizontalEllipticSolver supplies J internally, exactly once.
    Kokkos::parallel_for("BuildHorizontalDiagnosticRHS",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            rhs_psi(j, i) = zeta(top, j, i);
            rhs_chi(j, i) = flex(top) * rho_up(top - 1) * w(top - 1, j, i) * inverse_dz / rho(top);
        });

    solver.make_extrapolated_guess(fields.psi, fields.psi_previous, workspace.solution_psi);
    solver.make_extrapolated_guess(fields.chi, fields.chi_previous, workspace.solution_chi);
    solver.solve_at_z_and_t(workspace.rhs_psi, workspace.solution_psi, workspace.rhs_chi, workspace.solution_chi, options);

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

    // Refresh current AND previous potentials, without swapping their allocations.
    halo.exchange_multiple_halos({&fields.psi, &fields.chi, &fields.psi_previous, &fields.chi_previous});

    if (horizontal.ny > 1 && horizontal.topology.q2 == Core::HorizontalEdgeTopology::Bounded) {
        Core::Boundary::HorizontalBoundaryStencils boundary(grid);

        boundary.fill_constant_q2_halos(fields.psi);
        boundary.fill_constant_q2_halos(fields.chi);
        boundary.fill_constant_q2_halos(fields.psi_previous);
        boundary.fill_constant_q2_halos(fields.chi_previous);
    }

    adapter.reconstruct_top(fields.psi, fields.chi, fields.u, fields.v, top);

    const auto inverse_h1 = grid.geometry().device_view(Core::Geometry::HorizontalLocation::U).physical_to_contravariant.a11;
    const auto increment = fields.zonal_covariant_increment.get_device_data();
    const auto u = fields.u.get_mutable_device_data();

    Kokkos::parallel_for("AddPrescribedZonalCovariantIncrement",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            u(top, j, i) += increment() * inverse_h1(j, i);
        });

    adapter.integrate_from_top(fields.w, fields.xi, fields.eta, fields.spacing, fields.u, fields.v, bottom, top);
}

} // namespace Dynamics
} // namespace VVM
