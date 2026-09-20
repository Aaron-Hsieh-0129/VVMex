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

    // Legacy physical/terrain-compatible path.
    //
    // Inputs:
    //
    //     xi  = physical omega_1
    //     eta = -physical omega_2
    //
    // This path remains required by terrain-adjusted xi_topo / eta_topo.
    void solve(const Core::Field<3>& xi,
        const Core::Field<3>& eta,
        Core::Field<3>& w,
        Core::Field<3>& previous_w,
        int iterations);

    // Canonical persistent-vorticity path.
    //
    // Inputs follow the VVM State convention:
    //
    //     xi_con  =  omega^1
    //     eta_con = -omega^2
    //
    // The solver equation, iteration order, Thomas coefficients and w/history
    // semantics are identical to solve(). Only weighted-RHS construction differs.
    void solve_from_vvm_contravariant_state(const Core::Field<3>& xi_con,
        const Core::Field<3>& eta_con,
        Core::Field<3>& w,
        Core::Field<3>& previous_w,
        int iterations);

    enum class VorticityInputRepresentation { PhysicalLegacy, VvmContravariant };

    void solve_impl(const Core::Field<3>& q1_vorticity,
        const Core::Field<3>& q2_vorticity,
        Core::Field<3>& w,
        Core::Field<3>& previous_w,
        int iterations,
        VorticityInputRepresentation representation);

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
