#ifndef VVM_DYNAMICS_VERTICAL_ELLIPTIC_SOLVER_HPP
#define VVM_DYNAMICS_VERTICAL_ELLIPTIC_SOLVER_HPP

#include <memory>
#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "dynamics/solvers/VerticalWindDiagnostic.hpp"

namespace VVM {
namespace Dynamics {

// RLL port of the existing fixed local iteration with a vertical Thomas solve.
// No convergence stopping, mean removal, RHS projection, or terrain support.
// Existing WindSolver::solve_w and its Cartesian graph are not replaced.
class VerticalEllipticSolver {
public:
    // Initialization only, outside capture. Density/stretching values are frozen
    // into private coefficients. Construct another solver if they change.
    // Grid and HaloExchanger must describe the same decomposition and stream.
    VerticalEllipticSolver(const Core::Grid& grid,
        Core::HaloExchanger& halo,
        const Core::Field<1>& rhobar,
        const Core::Field<1>& rhobar_up,
        const Core::Field<1>& flex_mid,
        const Core::Field<1>& flex_up,
        Real inverse_dz,
        Real diagonal_shift);

    VerticalEllipticSolver(const VerticalEllipticSolver&) = delete;
    VerticalEllipticSolver& operator=(const VerticalEllipticSolver&) = delete;

    static void prepare_execution();

    // Inputs are physical xi and legacy-sign eta, with valid horizontal halos.
    // Save old w into previous_w, extrapolate 2*w-previous_w, then perform
    // exactly iterations line-relaxation sweeps. No source fields are modified.
    // Interior k = halo ... nz-halo-2; rigid surfaces at halo-1 and nz-halo-1.
    // Final w halos are refreshed through the full horizontal halo depth.
    // Bounded q2 uses CVVM MODE=2 copying, not a complete free-slip policy.
    // All inputs/outputs must be distinct, full Grid-sized allocations.
    // Solver, geometry, halo, and field allocations must outlive queued work
    // and captured graphs. Do not call this solver concurrently on two streams.
    void solve(const Core::Field<3>& xi,
        const Core::Field<3>& eta,
        Core::Field<3>& w,
        Core::Field<3>& previous_w,
        int iterations);

private:
    using WorkField = Core::Field<3, Kokkos::LayoutRight>;
    void refresh(WorkField& field, int depth);
    void validate(const Core::Field<3>& field) const;

    const Core::Grid& grid_;
    Core::HaloExchanger& halo_;
    VerticalWindDiagnosticDeviceView operation_;
    Real shift_;
    WorkField rhs_, first_, second_;
    Kokkos::View<Real*> density_up_;
    Kokkos::View<Real**> lower_, pivot_, upper_normalized_, temporary_;
    std::unique_ptr<Core::Boundary::HorizontalBoundaryStencils> boundary_;
};

} // namespace Dynamics
} // namespace VVM

#endif
