#include "dynamics/solvers/WindSolver.hpp"
#include "dynamics/solvers/VerticalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindTopologyConstraint.hpp"
#include "dynamics/solvers/GeneralizedWindDiagnostic.hpp"
#include "core/geometry/HorizontalLocation.hpp"

#include "dynamics/operators/GeneralizedTerrain.hpp"
#include "dynamics/operators/StaggeredVectorTransform.hpp"

#include <array>

namespace VVM::Dynamics {
namespace {
struct CovariantPotentialVolume {
    Kokkos::View<Operators::GeneralizedHorizontalWindReconstructionDeviceView> reconstruction;
    Core::Field<2>::ViewType psi, chi;
    bool first;
    int ny, nx;
    KOKKOS_INLINE_FUNCTION int
    extent(int dimension) const {
        return dimension == 1 ? ny : nx;
    }
    KOKKOS_INLINE_FUNCTION Real
    operator()(int, int j, int i) const {
        return first ? reconstruction().calculate_covariant_q1_at_u(psi, chi, j, i)
                     : reconstruction().calculate_covariant_q2_at_v(psi, chi, j, i);
    }
};

// Outside solver capture, like the existing Cartesian terrain adaptation.
// The elliptic matrices, sweep counts and graph replay are unchanged.
void
adapt_terrain(Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Real inverse_dz,
    Core::HaloExchanger& halo,
    Core::Boundary::HorizontalBoundaryStencils* boundary,
    Core::Field<3>& canonical_xi,
    Core::Field<3>& canonical_eta) {
    const int nz = grid.get_local_total_points_z(), ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x(), h = grid.get_halo_cells();
    const auto u = state.get_field<3>("u").get_device_data();
    const auto v = state.get_field<3>("v").get_device_data();
    const auto w = state.get_field<3>("w").get_device_data();
    const auto mu = state.get_field<3>("ITYPEU").get_device_data();
    const auto mv = state.get_field<3>("ITYPEV").get_device_data();
    const auto mw = state.get_field<3>("ITYPEW").get_device_data();
    auto& uf = state.get_field<3>("u_topo");
    auto& vf = state.get_field<3>("v_topo");
    auto& wf = state.get_field<3>("w_topo");
    const auto ut = uf.get_mutable_device_data(), vt = vf.get_mutable_device_data(),
               wt = wf.get_mutable_device_data();
    auto& xf = state.get_field<3>("xi_topo");
    auto& ef = state.get_field<3>("eta_topo");
    const auto xt = canonical_xi.get_mutable_device_data(),
               et = canonical_eta.get_mutable_device_data();
    const auto xi = state.get_field<3>("xi_con").get_device_data();
    const auto eta = state.get_field<3>("eta_con").get_device_data();
    // Copy vorticity in the same full-volume pass as the masked winds. The
    // following terrain correction overwrites only the blocked faces.
    Kokkos::parallel_for("GeneralizedTerrainMaskedWinds",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            xt(k, j, i) = xi(k, j, i);
            et(k, j, i) = eta(k, j, i);
            ut(k, j, i) = mu(k, j, i) == real(1.) ? u(k, j, i) : real(0.);
            vt(k, j, i) = mv(k, j, i) == real(1.) ? v(k, j, i) : real(0.);
            wt(k, j, i) = mw(k, j, i) == real(1.) ? w(k, j, i) : real(0.);
        });
    halo.exchange_multiple_halos(std::vector<Core::Field<3>*>{&uf, &vf, &wf});
    if (boundary) {
        boundary->fill_free_slip_physical_wind_halos(uf, vf);
        boundary->fill_physical_q2_neumann_halos(wf);
    }
    const auto flex = params.flex_height_coef_up.get_device_data();
    const Real rdz = inverse_dz;
    const auto operation = Operators::make_generalized_terrain_device_view(grid.geometry());
    Kokkos::parallel_for("GeneralizedTerrainAdaptedVorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h - 1, h, h}, {nz - h - 1, ny - h, nx - h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            if (mv(k, j, i) != real(1.)) {
                xt(k, j, i) = operation.xi(ut, vt, wt, rdz * flex(k), k, j, i);
            }
            if (mu(k, j, i) != real(1.)) {
                et(k, j, i) = operation.eta(ut, vt, wt, rdz * flex(k), k, j, i);
            }
        });
    halo.exchange_multiple_halos(std::vector<Core::Field<3>*>{&canonical_xi, &canonical_eta});
    if (boundary) {
        boundary->fill_contravariant_vorticity_halos(canonical_xi, canonical_eta);
    }
    using Core::Geometry::HorizontalLocation;
    const auto gu = grid.geometry().device_view(HorizontalLocation::U);
    const auto gv = grid.geometry().device_view(HorizontalLocation::V);
    Operators::transform_staggered_vector<false>(xt,
        et,
        xf.get_mutable_device_data(),
        ef.get_mutable_device_data(),
        gv.contravariant_to_physical,
        gu.contravariant_to_physical,
        real(-1.0));
}

#if defined(ENABLE_NCCL)
void
require_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}
#endif
}

void
WindSolver::initialize_generalized_solver(const int nz) {

    using Core::Geometry::GeometryFieldLayout;
    using Core::Geometry::HorizontalLocation;
    const auto gu = grid_.geometry().device_view(HorizontalLocation::U);
    const auto gv = grid_.geometry().device_view(HorizontalLocation::V);
    const auto zero = [](const auto& field) {
        return field.layout == GeometryFieldLayout::Constant && field.constant == real(0.0);
    };
    if (!zero(gu.sqrt_g_g_contra.a12) || !zero(gv.sqrt_g_g_contra.a12)) {
        const std::array<int, 2> plane_shape{grid_.get_local_total_points_y(),
            grid_.get_local_total_points_x()};
        mapped_top_correction_q1_ =
            std::make_unique<Core::Field<2>>("canonical top q1 correction", plane_shape);
        mapped_top_correction_q2_ =
            std::make_unique<Core::Field<2>>("canonical top q2 correction", plane_shape);
        mapped_reconstruction_ =
            decltype(mapped_reconstruction_)("mapped canonical top reconstruction");
        auto host = Kokkos::create_mirror_view(mapped_reconstruction_);
        host() = Operators::make_generalized_horizontal_wind_reconstruction_device_view(
            grid_.geometry());
        Kokkos::deep_copy(mapped_reconstruction_, host);
    }

    mapped_inverse_dz_ = params_.get_value_host(params_.rdz);

    if (state_.has_field("rll_psi_north")) {
        Kokkos::deep_copy(mapped_psi_north_,
            state_.get_field<0>("rll_psi_north").get_device_data());
    }
    else {
        mapped_psi_north_ = config_.get_value<Real>("dynamics.solver.psi_q2_plus", real(0.0));
    }

    mapped_spacing_ =
        std::make_unique<Core::Field<1>>("Generalized wind spacing", std::array<int, 1>{nz});

    const std::array<int, 3> volume_shape{nz,
        grid_.get_local_total_points_y(),
        grid_.get_local_total_points_x()};

    mapped_covariant_q1_wind_ =
        std::make_unique<Core::Field<3>>("Generalized covariant q1 wind scratch", volume_shape);

    mapped_covariant_q2_wind_ =
        std::make_unique<Core::Field<3>>("Generalized covariant q2 wind scratch", volume_shape);

    mapped_terrain_xi_con_ =
        std::make_unique<Core::Field<3>>("Generalized terrain xi contravariant scratch",
            volume_shape);

    mapped_terrain_eta_con_ =
        std::make_unique<Core::Field<3>>("Generalized terrain eta contravariant scratch",
            volume_shape);

    mapped_prescribed_zonal_covariant_increment_ =
        std::make_unique<Core::Field<0>>("Generalized prescribed zonal covariant increment",
            std::array<int, 0>{});

    horizontal_wind_constraint_ = make_generalized_circulation_constraint(grid_,
        state_,
        *mapped_covariant_q1_wind_,
        *mapped_covariant_q2_wind_,
        halo_exchanger_,
        horizontal_elliptic_solver_,
        config_.get_value<int>("dynamics.solver.initial_iterations"));

    if (config_.get_value<bool>("dynamics.forcings.areamn.enable", false)) {
        horizontal_wind_constraint_->prepare_physical_mean_constraint(halo_exchanger_);
        const Real tau = config_.get_value<Real>("dynamics.forcings.areamn.uvtau", real(0.0));
        mapped_mean_nudging_fraction_ =
            tau == real(0.0) ? real(1.0) : config_.get_value<Real>("simulation.dt_s") / tau;
    }

    const auto spacing = mapped_spacing_->get_mutable_device_data();
    const auto z = params_.z_mid.get_device_data(),
               flex = params_.flex_height_coef_up.get_device_data();
    const Real dz = params_.get_value_host(params_.dz);
    Kokkos::parallel_for("MappedWindPhysicalHeightSpacing", nz, KOKKOS_LAMBDA(int k) {
        const Real distance = k + 1 < nz ? z(k + 1) - z(k) : real(0.0);
        spacing(k) = distance > real(0.0) ? distance : (flex(k) > real(0.0) ? dz / flex(k) : dz);
    });

    mapped_vertical_solver_ = std::make_unique<VerticalEllipticSolver>(grid_,
        halo_exchanger_,
        state_.get_field<1>("rhobar"),
        state_.get_field<1>("rhobar_up"),
        params_.flex_height_coef_mid,
        params_.flex_height_coef_up,
        mapped_inverse_dz_,
        params_.get_value_host(params_.WRXMU));

    GeneralizedWindDiagnostic::prepare_execution();
}

void
WindSolver::solve_generalized() {
    const bool initial = !mapped_initialized_;
    const bool q2_periodic =
        grid_.horizontal_specification().topology.q2 == Core::HorizontalEdgeTopology::Periodic;
    const bool terrain = params_.max_topo_idx > grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    if (initial) {
        initialize_generalized_solver(nz);
    }

    if (!horizontal_wind_constraint_) {
        throw std::logic_error("Generalized wind recovery requires an initialized "
                               "topology constraint.");
    }

    // Ordinary channels inherit their incoming circulation. Analytic cases
    // and explicit streamfunction wall values retain their prescribed gauge.
    if (initial && !q2_periodic && !state_.has_field("rll_psi_north") &&
        !config_.has_key("dynamics.solver.psi_q2_plus")) {
        horizontal_wind_constraint_->seed_from_physical_wind();
    }
    horizontal_wind_constraint_->before_recovery(initial);

    GeneralizedWindDiagnosticOptions options;

    if (q2_periodic) {
        options.boundary_policy = HorizontalDiagnosticBoundaryPolicy::PeriodicQ2;
    }

    auto fields = prepare_generalized_wind_recovery(initial, terrain, options);

    HorizontalDiagnosticWorkspace workspace{rhs_psi_field_,
        rhs_chi_field_,
        psi_out_field_,
        chi_out_field_};

    recover_generalized_horizontal_wind(initial, terrain, fields, workspace, options);

    mapped_initialized_ = true;
}

void
WindSolver::initialize_generalized_restart_state(bool canonical_horizontal_state) {
    if (grid_.geometry().kind() == Core::Geometry::GeometryKind::Cartesian || mapped_initialized_) {
        return;
    }
    initialize_generalized_solver(grid_.get_local_total_points_z());
    if (canonical_horizontal_state) {
        if (bounded_q2_stencils_) {
            bounded_q2_stencils_->fill_contravariant_wind_halos(
                state_.get_field<3>("u_con"), state_.get_field<3>("v_con"));
        }
        horizontal_wind_constraint_->seed_from_contravariant_wind();
    }
    else {
        horizontal_wind_constraint_->seed_from_physical_wind();
    }
    mapped_initialized_ = true;
}

GeneralizedWindDiagnosticFields
WindSolver::prepare_generalized_wind_recovery(
    const bool initial, const bool terrain, GeneralizedWindDiagnosticOptions& options) {

    options.vertical_iterations = config_.get_value<int>("dynamics.solver.vertical_iterations");

    options.horizontal = horizontal_elliptic_options_;

    options.horizontal.psi_q2_plus = mapped_psi_north_;

    options.horizontal.iterations =
        initial ? config_.get_value<int>("dynamics.solver.initial_iterations")
                : params_.solver_iteration;

    options.inverse_dz = mapped_inverse_dz_;

    if (terrain) {
        adapt_terrain(state_,
            grid_,
            params_,
            mapped_inverse_dz_,
            halo_exchanger_,
            bounded_q2_stencils_.get(),
            *mapped_terrain_xi_con_,
            *mapped_terrain_eta_con_);
    }

    const Core::Field<3>& active_xi_con =
        terrain ? *mapped_terrain_xi_con_ : state_.get_field<3>("xi_con");

    const Core::Field<3>& active_eta_con =
        terrain ? *mapped_terrain_eta_con_ : state_.get_field<3>("eta_con");

    return GeneralizedWindDiagnosticFields{state_.get_field<2>("psi"),
        state_.get_field<2>("psinm1"),
        state_.get_field<2>("chi"),
        state_.get_field<2>("chinm1"),
        state_.get_field<3>("zeta"),
        state_.get_field<3>("w"),
        state_.get_field<3>("W3DNM1"),
        active_xi_con,
        active_eta_con,
        *mapped_covariant_q1_wind_,
        *mapped_covariant_q2_wind_,
        state_.get_field<1>("rhobar"),
        state_.get_field<1>("rhobar_up"),
        params_.flex_height_coef_mid,
        *mapped_spacing_,
        *mapped_prescribed_zonal_covariant_increment_};
}

void
WindSolver::snapshot_generalized_top_vertical_vorticity() {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const int top = nz - h - 1;

    const auto zeta = state_.get_field<3>("zeta").get_device_data();
    if (!state_.has_field("rll_zeta_top")) {
        return;
    }
    const auto snapshot = state_.get_field<2>("rll_zeta_top").get_mutable_device_data();

    Kokkos::parallel_for("SnapshotGeneralizedTopVerticalVorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) { snapshot(j, i) = zeta(top, j, i); });
}

void
WindSolver::commit_generalized_recovered_wind(const GeneralizedWindDiagnosticFields& fields) {
    using Core::Geometry::HorizontalLocation;

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const int bottom = h - 1;
    const int top = nz - h - 1;

    // Refresh covariant scratch halos before native-face mixed terms are
    // interpolated. The column recovery only writes physical columns.
    halo_exchanger_.exchange_multiple_halos(
        std::vector<Core::Field<3>*>{&fields.covariant_q1_wind, &fields.covariant_q2_wind});
    if (bounded_q2_stencils_) {
        bounded_q2_stencils_->fill_centered_q2_neumann_halos(fields.covariant_q1_wind);
        // Normal contravariant velocity is constrained after raising.
        bounded_q2_stencils_->fill_centered_q2_neumann_halos(fields.covariant_q2_wind);
    }
    Operators::raise_staggered_wind(fields.covariant_q1_wind.get_device_data(),
        fields.covariant_q2_wind.get_device_data(),
        state_.get_field<3>("u_con").get_mutable_device_data(),
        state_.get_field<3>("v_con").get_mutable_device_data(),
        grid_.geometry());
    if (mapped_reconstruction_.data()) {
        // Raising an already lowered rotational wind would interpolate mixed
        // terms twice. Preserve the direct Helmholtz top reconstruction and
        // apply only the raised column shear and harmonic correction.
        halo_exchanger_.exchange_multiple_halos(
            std::vector<Core::Field<2>*>{&fields.psi, &fields.chi},
            2);
        if (bounded_q2_stencils_) {
            bounded_q2_stencils_->fill_positive_face_q2_dirichlet_halos(fields.psi,
                real(0.0),
                mapped_psi_north_);
            bounded_q2_stencils_->fill_physical_q2_neumann_halos(fields.chi);
        }
        const auto gu = grid_.geometry().device_view(HorizontalLocation::U);
        const auto gv = grid_.geometry().device_view(HorizontalLocation::V);
        const auto c11 = gu.sqrt_g_g_contra.a11, c12v = gv.sqrt_g_g_contra.a12,
                   inv_ju = gu.inv_sqrt_g;
        const auto c22 = gv.sqrt_g_g_contra.a22, c12u = gu.sqrt_g_g_contra.a12,
                   inv_jv = gv.inv_sqrt_g;
        const auto reconstruction = mapped_reconstruction_;
        const auto psi = fields.psi.get_device_data(), chi = fields.chi.get_device_data();
        const CovariantPotentialVolume original1{reconstruction, psi, chi, true, ny, nx};
        const CovariantPotentialVolume original2{reconstruction, psi, chi, false, ny, nx};
        const auto correction1 = mapped_top_correction_q1_->get_mutable_device_data();
        const auto correction2 = mapped_top_correction_q2_->get_mutable_device_data();
        const auto q1 = state_.get_field<3>("u_con").get_mutable_device_data();
        const auto q2 = state_.get_field<3>("v_con").get_mutable_device_data();
        Kokkos::parallel_for("PreserveDirectCanonicalTopQ1",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(int j, int i) {
                const Real raised = inv_ju(j, i) * c11(j, i) * original1(top, j, i) +
                                    Operators::weighted_cross<true>(original2,
                                        Core::Geometry::InverseMetricCoefficient{c12v, inv_jv},
                                        top,
                                        j,
                                        i);
                const Real direct =
                    reconstruction().calculate_contravariant_q1_at_u(psi, chi, j, i);
                correction1(j, i) = direct - raised;
            });
        Kokkos::parallel_for("PreserveDirectCanonicalTopQ2",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(int j, int i) {
                const Real raised = inv_jv(j, i) * c22(j, i) * original2(top, j, i) +
                                    Operators::weighted_cross<false>(original1,
                                        Core::Geometry::InverseMetricCoefficient{c12u, inv_ju},
                                        top,
                                        j,
                                        i);
                const Real direct =
                    reconstruction().calculate_contravariant_q2_at_v(psi, chi, j, i);
                correction2(j, i) = direct - raised;
            });
        // Volume kernels follow the State layout's fastest dimension instead
        // of assigning one GPU thread a strided serial height loop.
        Kokkos::parallel_for("ApplyCanonicalColumnTopCorrection",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                q1(k, j, i) += correction1(j, i);
                q2(k, j, i) += correction2(j, i);
            });
    }
}

void
WindSolver::finalize_generalized_wind(
    Core::Field<3>& u_field, Core::Field<3>& v_field, const bool terrain) {
    auto& first = state_.get_field<3>("u_con");
    auto& second = state_.get_field<3>("v_con");
    const auto q1 = first.get_mutable_device_data(), q2 = second.get_mutable_device_data();
    const int h = grid_.get_halo_cells(), nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y(), nx = grid_.get_local_total_points_x();
    const int top = nz - h - 1;
    Kokkos::parallel_for("GeneralizedWindVerticalGhosts",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            if (k < h - 1) {
                q1(k, j, i) = q1(h - 1, j, i);
                q2(k, j, i) = q2(h - 1, j, i);
            }
            if (k > top) {
                q1(k, j, i) = q1(top, j, i);
                q2(k, j, i) = q2(top, j, i);
            }
        });
    halo_exchanger_.exchange_multiple_halos(std::vector<Core::Field<3>*>{&first, &second});
    if (bounded_q2_stencils_) {
        bounded_q2_stencils_->fill_contravariant_wind_halos(first, second);
    }
    using Core::Geometry::HorizontalLocation;
    const auto gu = grid_.geometry().device_view(HorizontalLocation::U);
    const auto gv = grid_.geometry().device_view(HorizontalLocation::V);
    Operators::transform_staggered_vector<true>(q1,
        q2,
        u_field.get_mutable_device_data(),
        v_field.get_mutable_device_data(),
        gu.contravariant_to_physical,
        gv.contravariant_to_physical);
    // Exchanging the exported physical shadows also supplies the outermost
    // allocation edges where the local cross-component average is clamped.
    halo_exchanger_.exchange_multiple_halos(std::vector<Core::Field<3>*>{&u_field, &v_field});
    if (terrain) {
        adapt_terrain(state_,
            grid_,
            params_,
            mapped_inverse_dz_,
            halo_exchanger_,
            bounded_q2_stencils_.get(),
            *mapped_terrain_xi_con_,
            *mapped_terrain_eta_con_);
    }
    else {
        // With no blocked faces the terrain shadows are the diagnosed fields.
        // Keep them current for zero-height terrain output and downstream use.
        for (const auto* name : {"u", "v", "w", "xi", "eta"}) {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
                state_.get_field<3>(std::string(name) + "_topo").get_mutable_device_data(),
                state_.get_field<3>(name).get_device_data());
        }
    }
}

void
WindSolver::execute_generalized_wind_diagnostic(const bool initial,
    GeneralizedWindDiagnosticFields& fields,
    HorizontalDiagnosticWorkspace& workspace,
    const GeneralizedWindDiagnosticOptions& options) {

    const auto diagnose = [&]() {
        GeneralizedWindDiagnostic::diagnose(grid_,
            halo_exchanger_,
            *mapped_vertical_solver_,
            horizontal_elliptic_solver_,
            fields,
            workspace,
            options);
    };

#if defined(ENABLE_NCCL)

    if (initial) {
        diagnose();
        return;
    }

    // AB2 exchanges the backing zeta allocation.
    // Capture one graph for each allocation while all
    // solver-private storage remains fixed.
    const auto key = fields.zeta.get_device_data().data();

    auto found = mapped_graphs_.find(key);

    const auto stream = Kokkos::DefaultExecutionSpace().cuda_stream();

    if (found == mapped_graphs_.end()) {

        require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            "Begin generalized wind diagnostic capture");

        diagnose();

        cudaGraph_t graph = nullptr;

        require_cuda(cudaStreamEndCapture(stream, &graph),
            "End generalized wind diagnostic capture");

        cudaGraphExec_t executable = nullptr;

        const auto result = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);

        cudaGraphDestroy(graph);

        require_cuda(result, "Instantiate generalized wind diagnostic graph");

        found = mapped_graphs_.emplace(key, executable).first;
    }

    require_cuda(cudaGraphLaunch(found->second, stream),
        "Replay generalized wind diagnostic graph");

#else

    diagnose();

#endif
}

void
WindSolver::recover_generalized_horizontal_wind(const bool initial,
    const bool terrain,
    GeneralizedWindDiagnosticFields& fields,
    HorizontalDiagnosticWorkspace& workspace,
    const GeneralizedWindDiagnosticOptions& options) {
    // Produce recovered covariant horizontal wind.
    execute_generalized_wind_diagnostic(initial, fields, workspace, options);

    snapshot_generalized_top_vertical_vorticity();

    if (!horizontal_wind_constraint_) {
        throw std::logic_error("Generalized wind recovery requires an initialized "
                               "topology constraint.");
    }

    // Circulation constraints now operate directly on the recovered
    // covariant wind.
    horizontal_wind_constraint_->after_recovery(initial);

    // Cross the Generalized representation boundary only after the topology
    // correction has been applied.
    commit_generalized_recovered_wind(fields);
    horizontal_wind_constraint_->correct_canonical_recovered_wind();

    finalize_generalized_wind(state_.get_field<3>("u"), state_.get_field<3>("v"), terrain);
    if (config_.get_value<bool>("dynamics.forcings.areamn.enable", false)) {
        const Real fraction = initial ? real(1.0) : mapped_mean_nudging_fraction_;
        horizontal_wind_constraint_->nudge_physical_top_mean(state_.get_field<0>("areamn_utopmn0"),
            state_.get_field<0>("areamn_vtopmn0"),
            fraction);
        finalize_generalized_wind(state_.get_field<3>("u"), state_.get_field<3>("v"), terrain);
    }
}

} // namespace VVM::Dynamics
