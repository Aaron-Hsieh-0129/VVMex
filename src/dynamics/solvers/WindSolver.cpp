#include "WindSolver.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#if defined(KOKKOS_ENABLE_CUDA)
#include <nvtx3/nvToolsExt.h>
#endif
#include <algorithm>
#include <utility>
#include <limits>

namespace VVM {
namespace Dynamics {

WindSolver::~WindSolver() {
#if defined(ENABLE_NCCL)
    int device = -1;
    if (cudaGetDevice(&device) == cudaSuccess) {
        cudaDeviceSynchronize(); 

        if (solve_w_graph_exec_) {
            cudaGraphExecDestroy(solve_w_graph_exec_);
        }
        if (relax_2d_graph_exec_) {
            cudaGraphExecDestroy(relax_2d_graph_exec_);
        }
    }
#endif
}

WindSolver::WindSolver(const Core::Grid& grid, const Utils::ConfigurationManager& config, const Core::Parameters& params, VVM::Core::HaloExchanger& halo_exchanger, VVM::Core::State& state)
    : grid_(grid), config_(config), halo_exchanger_(halo_exchanger), params_(params), state_(state),
      YTEM_field_("YTEM", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      w_deep_field_("w_deep", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      W3DN_field_("W3DN", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      rhs_psi_field_("rhs_psi", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      rhs_chi_field_("rhs_chi", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      psi_out_field_("psi_out", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      chi_out_field_("chi_out", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      psi_tmp_field_("psi_tmp", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      chi_tmp_field_("chi_tmp", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}) {

    std::string solver_method_str = config.get_value<std::string>("dynamics.solver.w_solver_method");
    if (solver_method_str == "tridiagonal") {
        w_solver_method_ = WSolverMethod::TRIDIAGONAL;
    } 
    else {
        w_solver_method_ = WSolverMethod::JACOBI;
    }

    VVM::Real h_WRXMU, h_rdx2, h_rdy2;
    Kokkos::deep_copy(h_WRXMU, params_.WRXMU);
    Kokkos::deep_copy(h_rdx2,  params_.rdx2);
    Kokkos::deep_copy(h_rdy2,  params_.rdy2);
    h_inv_C0_ = real(1.0) / (h_WRXMU + real(2.0) * h_rdx2 + real(2.0) * h_rdy2);

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();

    if (w_solver_method_ == WSolverMethod::TRIDIAGONAL) {
        const int h = grid_.get_halo_cells();
        const size_t ncols = static_cast<size_t>(ny - 2 * h) * static_cast<size_t>(nx - 2 * h);
        tri_tmp_ = Kokkos::View<VVM::Real**>("tri_tmp", ncols, nz);
    }

    if (!state_.has_field("utop_mean_tmp")) state_.add_field<0>("utop_mean_tmp", {});
    if (!state_.has_field("vtop_mean_tmp")) state_.add_field<0>("vtop_mean_tmp", {});
    if (!state_.has_field("W3DNM1")) state_.add_field<3>("W3DNM1", {nz, ny, nx}, Core::FieldMetadata{Core::GridStaggering::StaggeredZ, "m s-1", "previous-step vertical wind"});
    if (!state_.has_field("u_topo")) state_.add_field<3>("u_topo", {nz, ny, nx}, Core::FieldMetadata{Core::GridStaggering::StaggeredX, "m s-1", "topography-adjusted x wind"});
    if (!state_.has_field("v_topo")) state_.add_field<3>("v_topo", {nz, ny, nx}, Core::FieldMetadata{Core::GridStaggering::StaggeredY, "m s-1", "topography-adjusted y wind"});
    if (!state_.has_field("w_topo")) state_.add_field<3>("w_topo", {nz, ny, nx}, Core::FieldMetadata{Core::GridStaggering::StaggeredZ, "m s-1", "topography-adjusted vertical wind"});
    if (!state_.has_field("xi_topo")) state_.add_field<3>("xi_topo", {nz, ny, nx}, Core::FieldMetadata{Core::GridStaggering::StaggeredYZ, "s-1", "topography-adjusted x vorticity"});
    if (!state_.has_field("eta_topo")) state_.add_field<3>("eta_topo", {nz, ny, nx}, Core::FieldMetadata{Core::GridStaggering::StaggeredXZ, "s-1", "topography-adjusted y vorticity"});
}

void WindSolver::solve_w() {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    // The private work arrays carry the Grid halo, so they share extents and index
    // origin with the State fields: no shift between the two.
    const auto& iter_num = params_.solver_iteration;
    const auto& rdx = params_.rdx;
    const auto& rdy = params_.rdy;
    const auto& rdx2 = params_.rdx2;
    const auto& rdy2 = params_.rdy2;
    // const auto& rdz2 = params.rdz2;
    const auto& WRXMU = params_.WRXMU;
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& AGAU = params_.AGAU.get_device_data();
    const auto& BGAU = params_.BGAU.get_device_data();
    const auto& CGAU = params_.CGAU.get_device_data();

    const auto& rhobar_up = rhobar_up_ref_.get(state_, "rhobar_up").get_device_data();
    const auto& xi = xi_topo_ref_.get(state_, "xi_topo").get_device_data();
    const auto& eta = eta_topo_ref_.get(state_, "eta_topo").get_device_data();

    auto& w = w_ref_.get(state_, "w").get_mutable_device_data();

    auto YTEM = YTEM_field_.get_mutable_device_data();

    Kokkos::parallel_for("Poisson", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h-1,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
             YTEM(k,j,i)=-(eta(k,j,i) - eta(k,j,i-1))*rdx()
                         -( xi(k,j,i) -  xi(k,j-1,i))*rdy();
        }
    );

    // Linear extrapolation of initial guess
    auto& W3DNM1 = W3DNM1_ref_.get(state_, "W3DNM1").get_mutable_device_data();
    Kokkos::parallel_for("W3DNP1", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            VVM::Real w_val = w(k,j,i);
            
            if (k >= h && k < nz-h-1) {
                VVM::Real w_np1 = real(2.0) * w_val - W3DNM1(k,j,i);
                W3DNM1(k,j,i) = w_val;
                w(k,j,i) = w_np1;
            }
            else {
                W3DNM1(k,j,i) = w_val;
            }
        }
    );

    const auto& bn_new = params_.bn_new.get_device_data();
    const auto& cn_new = params_.cn_new.get_device_data();

#if defined(ENABLE_NCCL)
    cudaStream_t stream = Kokkos::Cuda().cuda_stream();
    if (solve_w_graph_created_) {
        cudaGraphLaunch(solve_w_graph_exec_, stream);
        return;
    }
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
#endif

    // Copy w into the solver-private iterate buffer. The buffers are private purely so
    // their layout can be chosen for this kernel (see DeepField in the header); w itself
    // is a State field and must keep the model-wide default layout. Halos included, so
    // the first sweep sees exactly what iterating on w directly would have seen.
    {
        auto w_deep = w_deep_field_.get_mutable_device_data();
        Kokkos::parallel_for("gather_w_priv", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
            KOKKOS_LAMBDA(int k, int j, int i) { w_deep(k,j,i) = w(k,j,i); }
        );
    }

    // cur holds the newest iterate. Swapping Field pointers (rather than mutating the
    // Views in place) keeps the sequence deterministic, which is what makes the
    // captured graph valid for replay.
    DeepField* cur = &w_deep_field_;
    DeepField* prv = &W3DN_field_;

    if (w_solver_method_ == WSolverMethod::TRIDIAGONAL) {
        for (int iter = 0; iter < iter_num; iter++) {
            {
                const int jlo = h, jhi = ny - h;
                const int ilo = h, ihi = nx - h;

                std::swap(cur, prv);
                auto P = prv->get_mutable_device_data();
                auto C = cur->get_mutable_device_data();

                const int nj = jhi - jlo, ni = ihi - ilo;
                const int ncols = nj * ni;
                auto tmp = tri_tmp_;

                Kokkos::parallel_for("fused_tridiagonal_solver",
                    Kokkos::RangePolicy<>(0, ncols),
                    KOKKOS_LAMBDA(const int idx) {
                        const int j = jlo + idx / ni;
                        const int i = ilo + idx % ni;

                        // Forward elimination)
                        tmp(idx,h) = (WRXMU() * P(h,j,i)
                                   + (P(h,j,i+1)+P(h,j,i-1))*rdx2()
                                   + (P(h,j+1,i)+P(h,j-1,i))*rdy2()
                                   + YTEM(h,j,i)) / BGAU(h);
                        for (int k = h+1; k <= nz-h-2; k++) {
                            const VVM::Real rhs_k = WRXMU() * P(k,j,i)
                                                  + (P(k,j,i+1)+P(k,j,i-1))*rdx2()
                                                  + (P(k,j+1,i)+P(k,j-1,i))*rdy2()
                                                  + YTEM(k,j,i);
                            tmp(idx,k) = (rhs_k - AGAU(k) * tmp(idx,k-1)) / bn_new(k);
                        }

                        // Backward substitution
                        VVM::Real pm_next = tmp(idx,nz-h-2);
                        C(nz-h-2,j,i) = pm_next / rhobar_up(nz-h-2);
                        for (int k = nz-h-3; k >= h; k--) {
                            const VVM::Real pm_k = tmp(idx,k) - cn_new(k) * pm_next;
                            C(k,j,i) = pm_k / rhobar_up(k);
                            pm_next = pm_k;
                        }

                        // Rigid lid / surface, as the old copy_w_to_w3dn pass did.
                        C(h-1,j,i)    = real(0.0);
                        C(nz-h-1,j,i) = real(0.0);
                    }
                );
            }
            halo_exchanger_.exchange_halos(*cur, 1);
        }
    }
    else {
        // Jacobi writes only where diagonal_term != 0, so the iterate must stay in one
        // buffer (as in the original) rather than alternating between two.
        auto C = cur->get_mutable_device_data();
        auto P = prv->get_mutable_device_data();
        for (int iter = 0; iter < iter_num; iter++) {
            {
                const int jlo = h, jhi = ny - h;
                const int ilo = h, ihi = nx - h;

                Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), P, C);

                Kokkos::parallel_for("jacobi_w_solver", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,jlo,ilo}, {nz-h-1,jhi,ihi}),
                    KOKKOS_LAMBDA(int k, int j, int i) {
                        const VVM::Real horizontal_terms = (P(k,j,i+1)+P(k,j,i-1))*rdx2()
                                                      + (P(k,j+1,i)+P(k,j-1,i))*rdy2();

                        const VVM::Real vertical_terms = -AGAU(k)*P(k-1,j,i)*rhobar_up(k-1)
                                                      -CGAU(k)*P(k+1,j,i)*rhobar_up(k+1);

                        const VVM::Real diagonal_term = BGAU(k)*rhobar_up(k) - WRXMU();

                        if (diagonal_term != real(0.0)) {
                            C(k,j,i) = (YTEM(k,j,i) + horizontal_terms + vertical_terms) / diagonal_term;
                        }
                    }
                );
            }
            // Full-depth here, matching the original Jacobi path (the tridiagonal
            // path exchanged depth 1).
            halo_exchanger_.exchange_halos(*cur);
        }
    }

    // Back into the State field, then the same depth-1 refresh the original did.
    {
        auto RES = cur->get_mutable_device_data();
        Kokkos::parallel_for("scatter_w_priv", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1,h,h}, {nz-h,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i) { w(k,j,i) = RES(k,j,i); }
        );
    }
    halo_exchanger_.exchange_halos(w_ref_.get(state_, "w"), 1);

#if defined(ENABLE_NCCL)
    cudaGraph_t graph = nullptr;
    cudaStreamEndCapture(stream, &graph);
    
    if (graph != nullptr) {
        cudaGraphInstantiate(&solve_w_graph_exec_, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        solve_w_graph_created_ = true;
        
        cudaGraphLaunch(solve_w_graph_exec_, stream);
    }
#endif
    return;
}


void WindSolver::solve_uv() {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& rhobar = rhobar_ref_.get(state_, "rhobar").get_device_data();
    const auto& rhobar_up = rhobar_up_ref_.get(state_, "rhobar_up").get_device_data();
    const auto& rdz = params_.rdz;

    auto& psi_field = psi_ref_.get(state_, "psi");
    auto& psinm1_field = psinm1_ref_.get(state_, "psinm1");
    auto& psi = psi_field.get_mutable_device_data();
    const auto& psinm1 = psinm1_field.get_device_data();
    const auto& zeta = zeta_ref_.get(state_, "zeta").get_device_data();

    auto& w = w_ref_.get(state_, "w").get_mutable_device_data();
    auto& chi_field = chi_ref_.get(state_, "chi");
    auto& chi = chi_field.get_mutable_device_data();
    auto& chinm1_field = chinm1_ref_.get(state_, "chinm1");
    const auto& chinm1 = chinm1_field.get_device_data();

    {
        auto rhs_psi = rhs_psi_field_.get_mutable_device_data();
        auto rhs_chi = rhs_chi_field_.get_mutable_device_data();
        auto psi_out = psi_out_field_.get_mutable_device_data();
        auto chi_out = chi_out_field_.get_mutable_device_data();

        // Right-hand sides and the time-extrapolated initial guesses, built on the
        // deep grid over the physical region only. The ring values then come from the
        // exchange below, and equal what the neighbour computes for the same cell --
        // which is what keeps this bitwise identical to relaxing each field alone.
        Kokkos::parallel_for("build_rhs_and_guess_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny,nx}),
            KOKKOS_LAMBDA(int j, int i) {
                rhs_psi(j,i) = zeta(nz-h-1,j,i);
                rhs_chi(j,i) = flex_height_coef_mid(nz-h-1)*rhobar_up(nz-h-2)*w(nz-h-2,j,i)*rdz() / rhobar(nz-h-1);
                psi_out(j,i) = real(2.)*psi(j,i) - psinm1(j,i);
                chi_out(j,i) = real(2.)*chi(j,i) - chinm1(j,i);
            }
        );
    }

    relax_2d_batched();

    // Rotate the State buffers, then refresh the halos the rest of solve_uv reads
    // (calculate_uvtop reaches one ring out in psi and chi).
    {
        auto psi_out = psi_out_field_.get_mutable_device_data();
        auto chi_out = chi_out_field_.get_mutable_device_data();
        auto psinm1_w = psinm1_field.get_mutable_device_data();
        auto chinm1_w = chinm1_field.get_mutable_device_data();
        Kokkos::parallel_for("scatter_psi_chi", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
            KOKKOS_LAMBDA(int j, int i) {
                psinm1_w(j,i) = psi(j,i);
                chinm1_w(j,i) = chi(j,i);
                psi(j,i) = psi_out(j,i);
                chi(j,i) = chi_out(j,i);
            }
        );
    }
    halo_exchanger_.exchange_halos(psi_field);
    halo_exchanger_.exchange_halos(chi_field);
    halo_exchanger_.exchange_halos(psinm1_field);
    halo_exchanger_.exchange_halos(chinm1_field);

    // Calculate utop, vtop
    auto& utop_field = utop_ref_.get(state_, "utop");
    auto& vtop_field = vtop_ref_.get(state_, "vtop");
    auto& utop = utop_field.get_mutable_device_data();
    auto& vtop = vtop_field.get_mutable_device_data();
    const auto& rdx = params_.rdx;
    const auto& rdy = params_.rdy;
    Kokkos::parallel_for("calculate_uvtop", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(int j, int i) {
            utop(j,i) = -(psi(j,i) - psi(j-1,i)) * rdy() + (chi(j,i+1) - chi(j,i)) * rdx();
            vtop(j,i) = (psi(j,i) - psi(j,i-1)) * rdx() + (chi(j+1,i) - chi(j,i)) * rdy();
        }
    );
    
    // calculate u
    auto& u_field = u_ref_.get(state_, "u");
    auto& u = u_field.get_mutable_device_data();
    auto& v_field = v_ref_.get(state_, "v");
    auto& v = v_field.get_mutable_device_data();

    auto& utopm = utop_mean_tmp_ref_.get(state_, "utop_mean_tmp").get_mutable_device_data();
    auto& vtopm = vtop_mean_tmp_ref_.get(state_, "vtop_mean_tmp").get_mutable_device_data();
    state_.calculate_horizontal_mean(utop_field, utopm);
    state_.calculate_horizontal_mean(vtop_field, vtopm);

    auto& utopmn = utopmn_ref_.get(state_, "utopmn").get_device_data();
    auto& vtopmn = vtopmn_ref_.get(state_, "vtopmn").get_device_data();

    // Note: this data clipping is necessary to prevent too small values and this makes CPU and GPU VVM same.
    Kokkos::parallel_for("DataClipZero", 1, KOKKOS_LAMBDA(const int i) {
        if (Kokkos::abs(utopm()) < 1e-15) {
            utopm() = real(0.0);
        }
        if (Kokkos::abs(vtopm()) < 1e-15) {
            vtopm() = real(0.0);
        }
    });


    Kokkos::parallel_for("uvtop_process", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(int j, int i) {
            u(nz-h-1,j,i) = utopmn() + utop(j,i) - utopm();
            v(nz-h-1,j,i) = vtopmn() + vtop(j,i) - vtopm();
        }
    );

    const auto& xi = xi_topo_ref_.get(state_, "xi_topo").get_mutable_device_data();
    const auto& eta = eta_topo_ref_.get(state_, "eta_topo").get_mutable_device_data();
    const auto& dz = params_.dz;
    Kokkos::parallel_for("u_downward_integration",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            // The for-loop inside is to prevent racing condition because lower layers depend on upper layers.
            for (int k = nz-h-2; k >= h-1; --k) {
                // WARNING: The eta is with negative because of the eta definition in original VVM
                // NOTE: Need to fix it if the definition is reversed.
                u(k,j,i) = u(k+1,j,i) 
                         - ((w(k,j,i+1) - w(k,j,i))*rdx() - eta(k,j,i)) * dz() / flex_height_coef_up(k); 
                v(k,j,i) = v(k+1,j,i) 
                         - ((w(k,j+1,i) - w(k,j,i))*rdy() - xi(k,j,i)) * dz() / flex_height_coef_up(k); 
            }
            // WARNING: NK3 has a upward integration in original VVM code.
            u(nz-h,j,i) = u(nz-h-1,j,i)
                      + ((w(nz-h-1,j,i+1) - w(nz-h-1,j,i))*rdx() - eta(nz-h-1,j,i)) * dz() / flex_height_coef_up(nz-h-1); 
            v(nz-h,j,i) = v(nz-h-1,j,i)
                      + ((w(nz-h-1,j+1,i) - w(nz-h-1,j,i))*rdy() -  xi(nz-h-1,j,i)) * dz() / flex_height_coef_up(nz-h-1); 
        }
    );
    if (uv_fields_.empty()) {
        uv_fields_ = {&u_ref_.get(state_, "u"), &v_ref_.get(state_, "v")};
    }
    halo_exchanger_.exchange_multiple_halos(uv_fields_);
    return;
}

// Relax psi and chi together on the deep 2-D grid. Callers must have filled the
// physical region of rhs_psi/rhs_chi and the initial guesses in psi_out/chi_out.
// On return the converged iterates are in psi_out_field_/chi_out_field_.
void WindSolver::relax_2d_batched() {

#if defined(ENABLE_NCCL)
    cudaStream_t stream = Kokkos::Cuda().cuda_stream();
    if (relax_2d_graph_created_) {
        cudaGraphLaunch(relax_2d_graph_exec_, stream);
        return;
    }
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
#endif

    const auto& WRXMU = params_.WRXMU;
    const int h = grid_.get_halo_cells();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const auto& iter_num = params_.solver_iteration;
    const auto& rdx2 = params_.rdx2;
    const auto& rdy2 = params_.rdy2;
    const VVM::Real inv_C0 = h_inv_C0_;

    auto rhs_psi = rhs_psi_field_.get_mutable_device_data();
    auto rhs_chi = rhs_chi_field_.get_mutable_device_data();


    Core::Field<2>* psi_cur = &psi_out_field_;
    Core::Field<2>* psi_prv = &psi_tmp_field_;
    Core::Field<2>* chi_cur = &chi_out_field_;
    Core::Field<2>* chi_prv = &chi_tmp_field_;

    for (int iter = 0; iter < iter_num; iter++) {
        {
            const int jlo = h, jhi = ny - h;
            const int ilo = h, ihi = nx - h;

            std::swap(psi_cur, psi_prv);
            std::swap(chi_cur, chi_prv);
            auto Pp = psi_prv->get_mutable_device_data();
            auto Cp = psi_cur->get_mutable_device_data();
            auto Pc = chi_prv->get_mutable_device_data();
            auto Cc = chi_cur->get_mutable_device_data();

            Kokkos::parallel_for("AOUT", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({jlo,ilo}, {jhi,ihi}),
                KOKKOS_LAMBDA(int j, int i) {
                    Cp(j,i) = (WRXMU()*Pp(j,i) + rdx2()*(Pp(j,i-1)+Pp(j,i+1))
                            + rdy2()*(Pp(j-1,i)+Pp(j+1,i)) - rhs_psi(j,i)) * inv_C0;
                    Cc(j,i) = (WRXMU()*Pc(j,i) + rdx2()*(Pc(j,i-1)+Pc(j,i+1))
                            + rdy2()*(Pc(j-1,i)+Pc(j+1,i)) - rhs_chi(j,i)) * inv_C0;
                }
            );
        }
        // psi and chi ride in one NCCL group: same operator, independent right-hand sides.
        halo_exchanger_.exchange_multiple_halos({psi_cur, chi_cur}, 1);
    }

    // Land the result in psi_out_field_/chi_out_field_ regardless of swap parity.
    if (psi_cur != &psi_out_field_) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), psi_out_field_.get_mutable_device_data(), psi_cur->get_mutable_device_data());
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), chi_out_field_.get_mutable_device_data(), chi_cur->get_mutable_device_data());
    }

#if defined(ENABLE_NCCL)
    cudaGraph_t graph = nullptr;
    cudaStreamEndCapture(stream, &graph);

    if (graph != nullptr) {
        cudaGraphInstantiate(&relax_2d_graph_exec_, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        relax_2d_graph_created_ = true;

        cudaGraphLaunch(relax_2d_graph_exec_, stream);
    }
#endif
    return;
}

} // namespace Dynamics
} // namespace VVM
