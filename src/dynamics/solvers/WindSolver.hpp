#ifndef VVM_DYNAMICS_WIND_SOLVER_HPP
#define VVM_DYNAMICS_WIND_SOLVER_HPP

#include <memory>
#include <map>
#include <vector>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/vvm_types.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindTopologyConstraint.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Dynamics {

enum class WSolverMethod { TRIDIAGONAL, JACOBI };

class VerticalEllipticSolver;
class HorizontalWindStateAdapter;

class WindSolver {
public:
    enum class HorizontalDiagnosticBoundaryPolicy {
        CvvmMode2Reference,
        RegularLatLonFreeSlipChannel,
        RegularLatLonPeriodic
    };

    WindSolver(const Core::Grid& grid,
        const Utils::ConfigurationManager& config,
        const Core::Parameters& params,
        VVM::Core::HaloExchanger& halo_exchanger,
        VVM::Core::State& state);
    ~WindSolver();

    WindSolver(const WindSolver&) = delete;
    WindSolver& operator=(const WindSolver&) = delete;

    void solve(const Core::BoundaryConditionManager& bc_manager);

    void solve_w();
    void recover_cartesian_horizontal_wind(); // original solve_uv
    void solve_regular_latlon();

    void relax_2d_batched();

    // Prepares the Cartesian terrain-adjusted wind/vorticity inputs used by
    // solve_w() and integrate_uv_from_top().
    void prepare_cartesian_wind_recovery_inputs(const Core::BoundaryConditionManager& bc_manager);

    // Integrate existing State u/v from their prescribed top physical level.
    // Uses xi_topo and legacy-sign eta_topo. Does not solve potentials,
    // modify top means, or exchange horizontal halos.
    // Currently implemented only for Cartesian geometry.
    void integrate_uv_from_top();

    struct HorizontalDiagnosticFields {
        Core::Field<2>& psi;
        Core::Field<2>& psi_previous;
        Core::Field<2>& chi;
        Core::Field<2>& chi_previous;
        const Core::Field<3>& zeta;
        const Core::Field<3>& w;
        const Core::Field<3>& xi;
        const Core::Field<3>& eta;
        Core::Field<3>& u;
        Core::Field<3>& v;
        const Core::Field<1>& rhobar;
        const Core::Field<1>& rhobar_up;
        const Core::Field<1>& flex_mid;
        const Core::Field<1>& spacing;
        const Core::Field<0>& zonal_covariant_increment;
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

    static void reconstruct_horizontal_top_wind(const HorizontalWindStateAdapter& adapter,
        const HorizontalDiagnosticFields& fields,
        int top);

    static void apply_prescribed_zonal_covariant_increment(
        const Core::Grid& grid, const HorizontalDiagnosticFields& fields, int top);

    static void integrate_horizontal_wind_from_top(const HorizontalWindStateAdapter& adapter,
        const HorizontalDiagnosticFields& fields,
        int bottom,
        int top);

    static void prepare_horizontal_diagnostic_execution();

    static void diagnose_horizontal_wind(const Core::Grid& grid,
        Core::HaloExchanger& halo,
        HorizontalEllipticSolver& solver,
        const HorizontalDiagnosticFields& fields,
        const HorizontalDiagnosticWorkspace& workspace,
        const HorizontalEllipticSolver::Options& options,
        VVM::Real inverse_dz,
        int bottom,
        int top,
        HorizontalDiagnosticBoundaryPolicy boundary_policy =
            HorizontalDiagnosticBoundaryPolicy::CvvmMode2Reference);

    static void diagnose_horizontal_potentials(const Core::Grid& grid,
        Core::HaloExchanger& halo,
        HorizontalEllipticSolver& solver,
        const HorizontalPotentialDiagnosticFields& fields,
        const HorizontalDiagnosticWorkspace& workspace,
        const HorizontalEllipticSolver::Options& options,
        VVM::Real inverse_dz,
        int top,
        HorizontalDiagnosticBoundaryPolicy boundary_policy);

    struct RegularLatLonDiagnosticFields {
        Core::Field<2>& psi;
        Core::Field<2>& psi_previous;
        Core::Field<2>& chi;
        Core::Field<2>& chi_previous;

        Core::Field<3>& zeta;
        Core::Field<3>& w;
        Core::Field<3>& w_previous;

        // Persistent canonical representation:
        //
        //     xi_con  =  omega^1
        //     eta_con = -omega^2
        //
        // Used by the generalized flat-RLL horizontal wind-column recovery.
        const Core::Field<3>& xi_con;
        const Core::Field<3>& eta_con;
        // Solver-private covariant wind scratch.
        //
        // These are not model State fields.
        Core::Field<3>& covariant_q1_wind;
        Core::Field<3>& covariant_q2_wind;
        Core::Field<3>& u;
        Core::Field<3>& v;
        const Core::Field<1>& rhobar;
        const Core::Field<1>& rhobar_up;
        const Core::Field<1>& flex_mid;
        const Core::Field<1>& spacing;
        const Core::Field<0>& zonal_covariant_increment;
    };

    struct RegularLatLonDiagnosticOptions {
        int vertical_iterations = 0;
        HorizontalEllipticSolver::Options horizontal;
        VVM::Real inverse_dz = VVM::real(0.0);
        HorizontalDiagnosticBoundaryPolicy boundary_policy =
            HorizontalDiagnosticBoundaryPolicy::RegularLatLonFreeSlipChannel;
    };

    static void prepare_regular_latlon_diagnostic_execution();

    // This remains a guarded diagnostic component. It does not enable complete
    // RLL time stepping or select/evolve the prescribed channel circulation.
    static void diagnose_regular_latlon_wind(const Core::Grid& grid,
        Core::HaloExchanger& halo,
        VerticalEllipticSolver& vertical_solver,
        HorizontalEllipticSolver& horizontal_solver,
        const RegularLatLonDiagnosticFields& fields,
        const HorizontalDiagnosticWorkspace& workspace,
        const RegularLatLonDiagnosticOptions& options,
        bool terrain);

    static void diagnose_regular_latlon_wind(const Core::Grid& grid,
        Core::HaloExchanger& halo,
        VerticalEllipticSolver& vertical_solver,
        HorizontalEllipticSolver& horizontal_solver,
        const RegularLatLonDiagnosticFields& fields,
        const HorizontalDiagnosticWorkspace& workspace,
        const RegularLatLonDiagnosticOptions& options);

    void finalize_regular_latlon_wind(
        Core::Field<3>& u_field, Core::Field<3>& v_field, bool terrain);

    // RLL diagnostic execution and CUDA graph capture/replay.
    void execute_regular_latlon_diagnostic(bool initial,
        RegularLatLonDiagnosticFields& fields,
        HorizontalDiagnosticWorkspace& workspace,
        const RegularLatLonDiagnosticOptions& options);

    void diagnose_cartesian_horizontal_potentials();

    void reconstruct_cartesian_top_wind();
    void apply_cartesian_top_wind_closure();

    void snapshot_regular_latlon_top_vertical_vorticity();

private:
    void initialize_regular_latlon_solver(bool periodic, int nz);

    RegularLatLonDiagnosticFields prepare_regular_latlon_wind_recovery(
        bool initial, bool terrain, RegularLatLonDiagnosticOptions& options);

    void fill_bounded_q2_potential_halos(Core::Field<2>& first, Core::Field<2>& second) const;
    void exchange_2d_solver_halos(Core::Field<2>& first, Core::Field<2>& second, int depth);

    void finalize_cartesian_wind();

    void recover_regular_latlon_horizontal_wind(bool initial,
        bool terrain,
        RegularLatLonDiagnosticFields& fields,
        HorizontalDiagnosticWorkspace& workspace,
        const RegularLatLonDiagnosticOptions& options);

    const Core::Grid& grid_;
    const Utils::ConfigurationManager& config_;
    const Core::Parameters& params_;
    Core::State& state_;
    WSolverMethod w_solver_method_;

    using DeepField = Core::Field<3, Kokkos::LayoutRight>;
    void exchange_w_solver_halos(DeepField& field, int depth);

    mutable DeepField YTEM_field_;
    mutable DeepField w_deep_field_;
    mutable DeepField W3DN_field_;

    mutable Core::Field<2> rhs_psi_field_;
    mutable Core::Field<2> rhs_chi_field_;
    mutable Core::Field<2> psi_out_field_;
    mutable Core::Field<2> chi_out_field_;
    mutable Core::Field<2> psi_tmp_field_;
    mutable Core::Field<2> chi_tmp_field_;

    Kokkos::View<VVM::Real**> tri_tmp_;

    Core::HaloExchanger& halo_exchanger_;

    HorizontalEllipticSolver horizontal_elliptic_solver_;
    HorizontalEllipticSolver::Options horizontal_elliptic_options_;

    std::unique_ptr<Core::Boundary::HorizontalBoundaryStencils> bounded_q2_stencils_;
    std::unique_ptr<VerticalEllipticSolver> rll_vertical_solver_;
    std::unique_ptr<Core::Field<1>> rll_spacing_;
    std::unique_ptr<Core::Field<0>> rll_prescribed_zonal_covariant_increment_;

    std::unique_ptr<Core::Field<3>> rll_covariant_q1_wind_;
    std::unique_ptr<Core::Field<3>> rll_covariant_q2_wind_;

    // Solver-private canonical representation of terrain-adjusted horizontal
    // vorticity.
    //
    // xi_topo / eta_topo remain physical/legacy State fields. These scratch
    // fields provide the representation consumed by the generalized RLL
    // wind-recovery backend.
    std::unique_ptr<Core::Field<3>> rll_terrain_xi_con_;
    std::unique_ptr<Core::Field<3>> rll_terrain_eta_con_;

    std::unique_ptr<HorizontalWindTopologyConstraint> horizontal_wind_constraint_;

    VVM::Real rll_inverse_dz_ = VVM::real(0.0);
    VVM::Real rll_psi_north_ = VVM::real(0.0);
    bool rll_initialized_ = false;

    VVM::Real h_inv_C0_;

    Core::FieldRef<0> utopmn_ref_;
    Core::FieldRef<0> vtopmn_ref_;
    Core::FieldRef<0> utop_mean_tmp_ref_;
    Core::FieldRef<0> vtop_mean_tmp_ref_;
    Core::FieldRef<1> rhobar_ref_;
    Core::FieldRef<1> rhobar_up_ref_;
    Core::FieldRef<2> psi_ref_;
    Core::FieldRef<2> psinm1_ref_;
    Core::FieldRef<2> chi_ref_;
    Core::FieldRef<2> chinm1_ref_;
    Core::FieldRef<2> utop_ref_;
    Core::FieldRef<2> vtop_ref_;
    Core::FieldRef<3> u_ref_;
    Core::FieldRef<3> v_ref_;
    Core::FieldRef<3> w_ref_;

    Core::FieldRef<3> xi_con_ref_;
    Core::FieldRef<3> eta_con_ref_;
    Core::FieldRef<3> zeta_con_ref_;

    Core::FieldRef<3> xi_topo_ref_;
    Core::FieldRef<3> eta_topo_ref_;

    Core::FieldRef<3> W3DNM1_ref_;

    Core::FieldRef<3> u_topo_ref_;
    Core::FieldRef<3> v_topo_ref_;
    Core::FieldRef<3> w_topo_ref_;

    Core::FieldRef<3> ITYPEU_ref_;
    Core::FieldRef<3> ITYPEV_ref_;
    Core::FieldRef<3> ITYPEW_ref_;

    mutable std::vector<Core::Field<3>*> uv_fields_;

#if defined(ENABLE_NCCL)
    std::map<const VVM::Real*, cudaGraphExec_t> rll_graphs_;
    bool solve_w_graph_created_ = false;
    cudaGraphExec_t solve_w_graph_exec_ = nullptr;

    bool relax_2d_graph_created_ = false;
    cudaGraphExec_t relax_2d_graph_exec_ = nullptr;
#endif
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_WIND_SOLVER_HPP
