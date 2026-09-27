#ifndef VVM_DYNAMICS_SOLVERS_GENERALIZED_WIND_DIAGNOSTIC_HPP
#define VVM_DYNAMICS_SOLVERS_GENERALIZED_WIND_DIAGNOSTIC_HPP

#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/vvm_types.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"

namespace VVM {
namespace Dynamics {

class VerticalEllipticSolver;

enum class HorizontalDiagnosticBoundaryPolicy {
    CvvmMode2Reference,
    FreeSlipBoundedQ2,
    PeriodicQ2,
};

struct HorizontalPotentialDiagnosticFields {
    Core::Field<2>& psi;
    Core::Field<2>& psi_previous;
    Core::Field<2>& chi;
    Core::Field<2>& chi_previous;

    const Core::Field<3>& zeta;
    const Core::Field<3>& w;

    const Core::Field<1>& rhobar;
    const Core::Field<1>& rhobar_up;
    const Core::Field<1>& flex_mid;
};

struct HorizontalDiagnosticWorkspace {
    Core::Field<2>& rhs_psi;
    Core::Field<2>& rhs_chi;
    Core::Field<2>& solution_psi;
    Core::Field<2>& solution_chi;
};

struct GeneralizedWindDiagnosticFields {
    Core::Field<2>& psi;
    Core::Field<2>& psi_previous;
    Core::Field<2>& chi;
    Core::Field<2>& chi_previous;

    Core::Field<3>& zeta;
    Core::Field<3>& w;
    Core::Field<3>& w_previous;

    // Canonical VVM horizontal vorticity:
    //
    //     xi_con  =  omega^1
    //     eta_con = -omega^2
    const Core::Field<3>& xi_con;
    const Core::Field<3>& eta_con;

    // Solver-private covariant wind.
    Core::Field<3>& covariant_q1_wind;
    Core::Field<3>& covariant_q2_wind;

    const Core::Field<1>& rhobar;
    const Core::Field<1>& rhobar_up;
    const Core::Field<1>& flex_mid;
    const Core::Field<1>& spacing;

    const Core::Field<0>& zonal_covariant_increment;
};

struct GeneralizedWindDiagnosticOptions {
    int vertical_iterations = 0;

    HorizontalEllipticSolver::Options horizontal;

    VVM::Real inverse_dz = VVM::real(0.0);

    HorizontalDiagnosticBoundaryPolicy boundary_policy =
        HorizontalDiagnosticBoundaryPolicy::FreeSlipBoundedQ2;
};

class GeneralizedWindDiagnostic {
public:
    static void prepare_execution();

    // Numerical generalized-coordinate wind diagnostic.
    //
    // Produces:
    //
    //     w
    //     zeta
    //     psi / chi
    //     covariant q1/q2 wind
    //
    // It deliberately does NOT convert wind to physical components.
    static void diagnose(const Core::Grid& grid,
        Core::HaloExchanger& halo,
        VerticalEllipticSolver& vertical_solver,
        HorizontalEllipticSolver& horizontal_solver,
        const GeneralizedWindDiagnosticFields& fields,
        const HorizontalDiagnosticWorkspace& workspace,
        const GeneralizedWindDiagnosticOptions& options);

    static void diagnose_horizontal_potentials(const Core::Grid& grid,
        Core::HaloExchanger& halo,
        HorizontalEllipticSolver& solver,
        const HorizontalPotentialDiagnosticFields& fields,
        const HorizontalDiagnosticWorkspace& workspace,
        const HorizontalEllipticSolver::Options& options,
        VVM::Real inverse_dz,
        int top,
        HorizontalDiagnosticBoundaryPolicy boundary_policy);
};

} // namespace Dynamics
} // namespace VVM

#endif
