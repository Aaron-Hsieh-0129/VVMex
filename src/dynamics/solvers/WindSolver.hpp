#ifndef VVM_DYNAMICS_WIND_SOLVER_HPP
#define VVM_DYNAMICS_WIND_SOLVER_HPP

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/vvm_types.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"
#include "utils/ConfigurationManager.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include <memory>
#include <map>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace VVM {
namespace Dynamics {

enum class WSolverMethod {
    TRIDIAGONAL, // Original method
    JACOBI       // 3D Jacobi iteration
};

class WindSolver {
public:
    WindSolver(const Core::Grid& grid, const Utils::ConfigurationManager& config, const Core::Parameters& params, VVM::Core::HaloExchanger& halo_exchanger, VVM::Core::State& state);
    ~WindSolver();

    WindSolver(const WindSolver&) = delete;
    WindSolver& operator=(const WindSolver&) = delete;

    void solve_w();
    void solve_uv();

    // psi and chi are relaxed together: same operator, same iteration count,
    // independent right-hand sides, so one batched exchange per sweep covers both
    // instead of two. Public because CUDA extended lambdas may not appear in a
    // private member.
    void relax_2d_batched();

private:
    const Core::Grid& grid_;
    const Utils::ConfigurationManager& config_;
    const Core::Parameters& params_;
    Core::State& state_;
    WSolverMethod w_solver_method_;

    // LayoutRight (x contiguous), not the Kokkos default. The per-sweep kernel puts
    // consecutive threads on consecutive i; under the default LayoutLeft that is a
    // stride of nz*ny -- ~300 KB apart at 2048^2/8 ranks -- so a warp scattered its
    // accesses across 32 cache lines per array, five of them for the stencil. With x
    // contiguous a warp reads 32 consecutive doubles instead. These three arrays are
    // solver-private, so nothing outside WindSolver sees the layout.
    using DeepField = Core::Field<3, Kokkos::LayoutRight>;
    mutable DeepField YTEM_field_;
    mutable DeepField w_deep_field_;
    mutable DeepField W3DN_field_;

    // Default layout: these are threaded (j,i) with j fastest, already coalesced.
    mutable Core::Field<2> rhs_psi_field_;   // was RIP1
    mutable Core::Field<2> rhs_chi_field_;   // was RIP2
    mutable Core::Field<2> psi_out_field_;   // was ROP1
    mutable Core::Field<2> chi_out_field_;   // was ROP2
    mutable Core::Field<2> psi_tmp_field_;   // was ATEMP, now one per field
    mutable Core::Field<2> chi_tmp_field_;

    // Default layout coalesces across columns on CUDA and along levels on host.
    Kokkos::View<VVM::Real**> tri_tmp_;

    Core::HaloExchanger& halo_exchanger_;

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
    Core::FieldRef<3> zeta_ref_;
    Core::FieldRef<3> xi_topo_ref_;
    Core::FieldRef<3> eta_topo_ref_;
    Core::FieldRef<3> W3DNM1_ref_;
    mutable std::vector<Core::Field<3>*> uv_fields_;

    VVM::Real h_inv_C0_;

#if defined(ENABLE_NCCL)
    bool solve_w_graph_created_ = false;
    cudaGraphExec_t solve_w_graph_exec_ = nullptr;

    // One graph now, not one per field name: psi and chi share a single batched loop.
    bool relax_2d_graph_created_ = false;
    cudaGraphExec_t relax_2d_graph_exec_ = nullptr;
#endif
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_WIND_SOLVER_HPP
