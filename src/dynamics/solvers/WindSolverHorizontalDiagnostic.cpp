#include "dynamics/solvers/WindSolver.hpp"
#include "dynamics/solvers/HorizontalWindStateAdapter.hpp"
#include "core/geometry/GeometryKind.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "dynamics/operators/HorizontalVectorConversion.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace VVM {
namespace Dynamics {

void
WindSolver::prepare_horizontal_diagnostic_execution() {
    HorizontalWindStateAdapter::prepare_execution();

#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::parallel_for("PrepareWindSolverHorizontalDiagnostic",
        Kokkos::RangePolicy<Kokkos::Cuda>(0, 1),
        KOKKOS_LAMBDA(const int){});

    Kokkos::Cuda().fence("Prepare WindSolver horizontal diagnostic");

    const auto result = cudaGetLastError();
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
#endif
}

void
WindSolver::reconstruct_horizontal_top_wind(const HorizontalWindStateAdapter& adapter,
    const HorizontalDiagnosticFields& fields,
    const int top) {

    adapter.reconstruct_top(fields.psi, fields.chi, fields.u, fields.v, top);
}

void
WindSolver::apply_prescribed_zonal_covariant_increment(
    const Core::Grid& grid, const HorizontalDiagnosticFields& fields, const int top) {

    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int h = grid.get_halo_cells();

    const auto inverse_h1 = grid.geometry()
                                .device_view(Core::Geometry::HorizontalLocation::U)
                                .physical_to_contravariant.a11;

    const auto increment = fields.zonal_covariant_increment.get_device_data();

    const auto u = fields.u.get_mutable_device_data();

    Kokkos::parallel_for("AddPrescribedZonalCovariantIncrement",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            u(top, j, i) +=
                Operators::HorizontalVectorConversion::covariant_to_physical(increment(),
                    inverse_h1(j, i));
        });
}

void
WindSolver::integrate_horizontal_wind_from_top(const HorizontalWindStateAdapter& adapter,
    const HorizontalDiagnosticFields& fields,
    const int bottom,
    const int top) {

    adapter.integrate_from_top(fields.w,
        fields.xi,
        fields.eta,
        fields.spacing,
        fields.u,
        fields.v,
        bottom,
        top);
}

void
WindSolver::diagnose_horizontal_wind(const Core::Grid& grid,
    Core::HaloExchanger& halo,
    HorizontalEllipticSolver& solver,
    const HorizontalDiagnosticFields& fields,
    const HorizontalDiagnosticWorkspace& workspace,
    const HorizontalEllipticSolver::Options& options,
    Real inverse_dz,
    int bottom,
    int top,
    HorizontalDiagnosticBoundaryPolicy boundary_policy) {

    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int nz = static_cast<int>(fields.w.get_device_data().extent(0));
    const int h = grid.get_halo_cells();
    const auto& horizontal = grid.horizontal_specification();

    const bool reference_boundary =
        boundary_policy == HorizontalDiagnosticBoundaryPolicy::CvvmMode2Reference;

    const bool free_slip_boundary =
        boundary_policy == HorizontalDiagnosticBoundaryPolicy::RegularLatLonFreeSlipChannel;
    const bool periodic_boundary =
        boundary_policy == HorizontalDiagnosticBoundaryPolicy::RegularLatLonPeriodic;

    if (!reference_boundary && !free_slip_boundary && !periodic_boundary) {
        throw std::invalid_argument("Unknown horizontal diagnostic boundary policy.");
    }
    if (periodic_boundary &&
        (grid.geometry().kind() != Core::Geometry::GeometryKind::RegularLatLon ||
            horizontal.topology.q1 != Core::HorizontalEdgeTopology::Periodic ||
            horizontal.topology.q2 != Core::HorizontalEdgeTopology::Periodic)) {
        throw std::invalid_argument("Periodic RLL diagnostic requires periodic q1 and q2.");
    }

    if (bottom < 0 || top < 1 || top < bottom || top >= nz) {
        throw std::invalid_argument("Invalid horizontal diagnostic levels.");
    }

    if (!std::isfinite(inverse_dz) || inverse_dz <= real(0.0)) {
        throw std::invalid_argument("Invalid inverse reference dz.");
    }

    if (options.iterations <= 0 || !std::isfinite(options.diagonal_shift) ||
        options.diagonal_shift < real(0.0) || !options.refresh_initial_halos) {
        throw std::invalid_argument("Horizontal diagnostic requires positive fixed iterations, "
                                    "nonnegative shift, and initial halo refresh.");
    }

    if (horizontal.nx > 1 && horizontal.topology.q1 == Core::HorizontalEdgeTopology::Bounded) {
        throw std::invalid_argument("Bounded q1 is not supported.");
    }

    if (free_slip_boundary &&
        (grid.geometry().kind() != Core::Geometry::GeometryKind::RegularLatLon ||
            horizontal.topology.q1 != Core::HorizontalEdgeTopology::Periodic ||
            horizontal.topology.q2 != Core::HorizontalEdgeTopology::Bounded)) {
        throw std::invalid_argument(
            "Free-slip channel diagnostics require RLL geometry, periodic q1, and bounded q2.");
    }

    const HorizontalWindStateAdapter adapter(grid.geometry());

    const std::array<const Core::Field<2>*, 8> planes = {&fields.psi,
        &fields.psi_previous,
        &fields.chi,
        &fields.chi_previous,
        &workspace.rhs_psi,
        &workspace.rhs_chi,
        &workspace.solution_psi,
        &workspace.solution_chi};

    const std::array<const Core::Field<3>*, 6> volumes =
        {&fields.zeta, &fields.w, &fields.xi, &fields.eta, &fields.u, &fields.v};

    for (const auto* field : planes) {
        const auto& data = field->get_device_data();

        if (static_cast<int>(data.extent(0)) != ny || static_cast<int>(data.extent(1)) != nx) {
            throw std::invalid_argument("Incorrect diagnostic plane extents.");
        }
    }

    for (const auto* field : volumes) {
        const auto& data = field->get_device_data();

        if (static_cast<int>(data.extent(0)) != nz || static_cast<int>(data.extent(1)) != ny ||
            static_cast<int>(data.extent(2)) != nx) {
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

    for (const auto* field : planes) {
        addresses[count++] = field->get_device_data().data();
    }
    for (const auto* field : volumes) {
        addresses[count++] = field->get_device_data().data();
    }

    addresses[count++] = fields.rhobar.get_device_data().data();
    addresses[count++] = fields.rhobar_up.get_device_data().data();
    addresses[count++] = fields.flex_mid.get_device_data().data();
    addresses[count++] = fields.spacing.get_device_data().data();
    addresses[count++] = fields.zonal_covariant_increment.get_device_data().data();

    for (int a = 0; a < count; ++a) {
        if (!addresses[a]) {
            throw std::invalid_argument("Unallocated diagnostic storage.");
        }

        for (int b = 0; b < a; ++b) {
            if (addresses[a] == addresses[b]) {
                throw std::invalid_argument("Diagnostic fields must have distinct storage.");
            }
        }
    }

    diagnose_horizontal_potentials(grid,
        halo,
        solver,
        fields,
        workspace,
        options,
        inverse_dz,
        top,
        boundary_policy);

    reconstruct_horizontal_top_wind(adapter, fields, top);

    apply_prescribed_zonal_covariant_increment(grid, fields, top);

    integrate_horizontal_wind_from_top(adapter, fields, bottom, top);
}

} // namespace Dynamics
} // namespace VVM
