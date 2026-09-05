#ifndef VVM_DYNAMICS_SOLVERS_HORIZONTAL_ELLIPTIC_SOLVER_HPP
#define VVM_DYNAMICS_SOLVERS_HORIZONTAL_ELLIPTIC_SOLVER_HPP

#include <memory>

#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/vvm_types.hpp"
#include "dynamics/operators/HorizontalLaplaceBeltrami.hpp"

namespace VVM {
namespace Dynamics {

// Fixed-iteration shited-Jacobi solver for
//
//     LaplaceBeltrami(solution) = right_hand_side.
//
// The caller must currently provide a compatible right-hand side with zero
// area-weighted mean. The initial solution mean selects the constant gauge.
// Explicit null-space projection and residual-based stopping will be added
// separately after the basic metric solver is validated.
class HorizontalEllipticSolver {
public:
    struct Options {
        int iterations = 0;
        VVM::Real diagonal_shift = VVM::real(0.0);
    };

    HorizontalEllipticSolver(const Core::Grid& grid, Core::HaloExchanger& halo_exchanger);

    void make_extrapolated_guess(const Core::Field<2>& current, const Core::Field<2>& previous, Core::Field<2>& guess) const;

    // Solve LaplaceBeltrami(solution) = right_hand_side at T points.
    // The caller supplies the physical right-hand side. The solver internally
    // uses the equivalent Jacobian-weighted equation.
    void solve_at_t(const Core::Field<2>& right_hand_side, Core::Field<2>& solution, const Options& options);

    // Solve LaplaceBeltrami(solution) = right_hand_side at Z points.
    // This is the streamfunction placement used for psi.
    void solve_at_z(const Core::Field<2>& right_hand_side, Core::Field<2>& solution, const Options& options);

    // Solve the Z-point streamfunction and T-point velocity potential together.
    // They use different metric stencils but share one kernel launch and one
    // batched halo exchange per iteration, preserving the current WindSolver
    // communication pattern.
    void solve_at_z_and_t(const Core::Field<2>& right_hand_side_at_z, Core::Field<2>& solution_at_z,
        const Core::Field<2>& right_hand_side_at_t, Core::Field<2>& solution_at_t, const Options& options);

private:
    void validate_field_extents(const Core::Field<2>& field, const char* role) const;
    void validate_solve_arguments(const Core::Field<2>& right_hand_side, const Core::Field<2>& solution, const Options& options) const;
    void refresh_solution_halos(Core::Field<2>& field);
    void refresh_solution_halos(Core::Field<2>& first, Core::Field<2>& second);

    const Core::Grid& grid_;
    Core::HaloExchanger& halo_exchanger_;
    Operators::HorizontalLaplaceBeltramiDeviceView laplace_beltrami_;
    Core::Field<2> scratch_at_z_;
    Core::Field<2> scratch_at_t_;

    std::unique_ptr<Core::Boundary::HorizontalBoundaryStencils> bounded_q2_stencils_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_SOLVERS_HORIZONTAL_ELLIPTIC_SOLVER_HPP
