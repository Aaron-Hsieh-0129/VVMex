#include "dynamics/solvers/WindSolver.hpp"
#include "dynamics/solvers/VerticalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindTopologyConstraint.hpp"
#include "dynamics/solvers/GeneralizedWindDiagnostic.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/RegularLatLonModelConfiguration.hpp"
#include "dynamics/operators/RegularLatLonTerrain.hpp"
#include "dynamics/operators/HorizontalVectorConversion.hpp"

#include <array>

namespace VVM::Dynamics {
namespace {
// Outside solver capture, like the existing Cartesian terrain adaptation.
// The elliptic matrices, sweep counts and graph replay are unchanged.
void
adapt_terrain(Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Real inverse_dz,
    Core::HaloExchanger& halo,
    Core::Boundary::HorizontalBoundaryStencils* boundary) {
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
    const auto xt = xf.get_mutable_device_data(), et = ef.get_mutable_device_data();
    const auto xi = state.get_field<3>("xi").get_device_data();
    const auto eta = state.get_field<3>("eta").get_device_data();
    // Copy vorticity in the same full-volume pass as the masked winds. The
    // following terrain correction overwrites only the blocked faces.
    Kokkos::parallel_for("RLLTerrainMaskedWinds",
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
        boundary->fill_regular_lat_lon_free_slip_physical_wind_halos(uf, vf);
        boundary->fill_centered_q2_neumann_halos(wf);
    }
    const auto flex = params.flex_height_coef_up.get_device_data();
    const Real rdz = inverse_dz;
    const auto operation = Operators::make_regular_lat_lon_terrain_device_view(grid.geometry());
    Kokkos::parallel_for("RLLTerrainAdaptedVorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h - 1, h, h}, {nz - h - 1, ny - h, nx - h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            if (mv(k, j, i) != real(1.)) {
                xt(k, j, i) = operation.xi(vt, wt, rdz * flex(k), k, j, i);
            }
            if (mu(k, j, i) != real(1.)) {
                et(k, j, i) = operation.eta(ut, wt, rdz * flex(k), k, j, i);
            }
        });
    halo.exchange_multiple_halos(std::vector<Core::Field<3>*>{&xf, &ef});
    if (boundary) {
        boundary->fill_positive_face_q2_homogeneous_dirichlet_halos(xf);
        boundary->fill_centered_q2_neumann_halos(ef);
    }
}

void
convert_regular_latlon_horizontal_vorticity_to_contravariant(const Core::Grid& grid,
    const Core::Field<3>& xi_physical,
    const Core::Field<3>& eta_physical,
    Core::Field<3>& xi_con,
    Core::Field<3>& eta_con) {

    using Core::Geometry::HorizontalLocation;
    using Operators::HorizontalVectorConversion;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    // xi = physical omega_1 and is V-staggered.
    const auto inverse_h1_at_v =
        grid.geometry().device_view(HorizontalLocation::V).physical_to_contravariant.a11;

    // eta = -physical omega_2 and is U-staggered.
    //
    // The historical VVM minus sign is already contained in eta, so no
    // additional sign change belongs to the representation conversion:
    //
    //     eta_con = eta / h2 = -omega^2.
    const auto inverse_h2_at_u =
        grid.geometry().device_view(HorizontalLocation::U).physical_to_contravariant.a22;

    const auto xi = xi_physical.get_device_data();
    const auto eta = eta_physical.get_device_data();

    auto xi_con_data = xi_con.get_mutable_device_data();
    auto eta_con_data = eta_con.get_mutable_device_data();

    // xi_topo / eta_topo already have their halo exchange and RLL wall
    // boundary conditions established by adapt_terrain(). Convert the full
    // allocation so the solver-private contravariant fields inherit those
    // valid halo values without another communication step.
    Kokkos::parallel_for("RLLTerrainVorticityToContravariant",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            xi_con_data(k, j, i) =
                HorizontalVectorConversion::physical_to_contravariant(xi(k, j, i),
                    inverse_h1_at_v(j, i));

            eta_con_data(k, j, i) =
                HorizontalVectorConversion::physical_to_contravariant(eta(k, j, i),
                    inverse_h2_at_u(j, i));
        });
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
WindSolver::initialize_regular_latlon_solver(const bool periodic, const int nz) {
    Core::validate_jung2019_rll(config_, Core::GridSpecification::from_config(config_));

    rll_inverse_dz_ = params_.get_value_host(params_.rdz);

    Kokkos::deep_copy(rll_psi_north_, state_.get_field<0>("rll_psi_north").get_device_data());

    rll_spacing_ = std::make_unique<Core::Field<1>>("RLL wind spacing", std::array<int, 1>{nz});

    const std::array<int, 3> volume_shape{nz,
        grid_.get_local_total_points_y(),
        grid_.get_local_total_points_x()};

    rll_covariant_q1_wind_ =
        std::make_unique<Core::Field<3>>("RLL covariant q1 wind scratch", volume_shape);

    rll_covariant_q2_wind_ =
        std::make_unique<Core::Field<3>>("RLL covariant q2 wind scratch", volume_shape);

    rll_terrain_xi_con_ =
        std::make_unique<Core::Field<3>>("RLL terrain xi contravariant scratch", volume_shape);

    rll_terrain_eta_con_ =
        std::make_unique<Core::Field<3>>("RLL terrain eta contravariant scratch", volume_shape);

    rll_prescribed_zonal_covariant_increment_ =
        std::make_unique<Core::Field<0>>("RLL prescribed zonal covariant increment",
            std::array<int, 0>{});

    horizontal_wind_constraint_ = make_regular_lat_lon_circulation_constraint(grid_,
        state_,
        *rll_covariant_q1_wind_,
        *rll_covariant_q2_wind_);

    Kokkos::deep_copy(rll_spacing_->get_mutable_device_data(), params_.get_value_host(params_.dz));

    rll_vertical_solver_ = std::make_unique<VerticalEllipticSolver>(grid_,
        halo_exchanger_,
        state_.get_field<1>("rhobar"),
        state_.get_field<1>("rhobar_up"),
        params_.flex_height_coef_mid,
        params_.flex_height_coef_up,
        rll_inverse_dz_,
        params_.get_value_host(params_.WRXMU));

    GeneralizedWindDiagnostic::prepare_execution();
}

void
WindSolver::solve_regular_latlon() {
    const bool initial = !rll_initialized_;
    const bool q2_periodic =
        grid_.horizontal_specification().topology.q2 == Core::HorizontalEdgeTopology::Periodic;
    const bool terrain = Core::is_rll_mountain(config_);
    const int nz = grid_.get_local_total_points_z();
    if (initial) {
        initialize_regular_latlon_solver(q2_periodic, nz);
    }

    if (!horizontal_wind_constraint_) {
        throw std::logic_error("RLL wind recovery requires an initialized "
                               "topology constraint.");
    }

    // Periodic topology captures its incoming harmonic cycles here.
    // Bounded-q2 is intentionally a no-op at this point.
    horizontal_wind_constraint_->before_recovery(initial);

    GeneralizedWindDiagnosticOptions options;

    if (q2_periodic) {
        options.boundary_policy = HorizontalDiagnosticBoundaryPolicy::PeriodicQ2;
    }

    auto fields = prepare_regular_latlon_wind_recovery(initial, terrain, options);

    HorizontalDiagnosticWorkspace workspace{rhs_psi_field_,
        rhs_chi_field_,
        psi_out_field_,
        chi_out_field_};

    recover_regular_latlon_horizontal_wind(initial, terrain, fields, workspace, options);

    rll_initialized_ = true;
}

GeneralizedWindDiagnosticFields
WindSolver::prepare_regular_latlon_wind_recovery(
    const bool initial, const bool terrain, GeneralizedWindDiagnosticOptions& options) {

    options.vertical_iterations = config_.get_value<int>("dynamics.solver.vertical_iterations");

    options.horizontal = horizontal_elliptic_options_;

    options.horizontal.psi_q2_plus = rll_psi_north_;

    options.horizontal.iterations =
        initial ? config_.get_value<int>("dynamics.solver.initial_iterations")
                : params_.solver_iteration;

    options.inverse_dz = rll_inverse_dz_;

    if (terrain) {
        adapt_terrain(state_,
            grid_,
            params_,
            rll_inverse_dz_,
            halo_exchanger_,
            bounded_q2_stencils_.get());

        convert_regular_latlon_horizontal_vorticity_to_contravariant(grid_,
            state_.get_field<3>("xi_topo"),
            state_.get_field<3>("eta_topo"),
            *rll_terrain_xi_con_,
            *rll_terrain_eta_con_);
    }

    const Core::Field<3>& active_xi_con =
        terrain ? *rll_terrain_xi_con_ : state_.get_field<3>("xi_con");

    const Core::Field<3>& active_eta_con =
        terrain ? *rll_terrain_eta_con_ : state_.get_field<3>("eta_con");

    return GeneralizedWindDiagnosticFields{state_.get_field<2>("psi"),
        state_.get_field<2>("psinm1"),
        state_.get_field<2>("chi"),
        state_.get_field<2>("chinm1"),
        state_.get_field<3>("zeta"),
        state_.get_field<3>("w"),
        state_.get_field<3>("W3DNM1"),
        active_xi_con,
        active_eta_con,
        *rll_covariant_q1_wind_,
        *rll_covariant_q2_wind_,
        state_.get_field<1>("rhobar"),
        state_.get_field<1>("rhobar_up"),
        params_.flex_height_coef_mid,
        *rll_spacing_,
        *rll_prescribed_zonal_covariant_increment_};
}

void
WindSolver::snapshot_regular_latlon_top_vertical_vorticity() {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const int top = nz - h - 1;

    const auto zeta = state_.get_field<3>("zeta").get_device_data();
    const auto snapshot = state_.get_field<2>("rll_zeta_top").get_mutable_device_data();

    Kokkos::parallel_for("SnapshotRLLTopVerticalVorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) { snapshot(j, i) = zeta(top, j, i); });
}

void
WindSolver::commit_regular_latlon_recovered_wind(const GeneralizedWindDiagnosticFields& fields,
    const GeneralizedWindDiagnosticOptions& options) {
    using Core::Geometry::HorizontalLocation;
    using Operators::HorizontalVectorConversion;

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const int bottom = h - 1;
    const int top = nz - h - 1;

    const auto inverse_h1_at_u =
        grid_.geometry().device_view(HorizontalLocation::U).physical_to_contravariant.a11;
    const auto inverse_h2_at_v =
        grid_.geometry().device_view(HorizontalLocation::V).physical_to_contravariant.a22;

    const auto q1_cov = fields.covariant_q1_wind.get_device_data();
    const auto q2_cov = fields.covariant_q2_wind.get_device_data();

    auto& u_field = state_.get_field<3>("u");
    auto& v_field = state_.get_field<3>("v");

    auto u = u_field.get_mutable_device_data();
    auto v = v_field.get_mutable_device_data();

    Kokkos::parallel_for("CommitRegularLatLonPhysicalWindFromCovariant",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u(k, j, i) = HorizontalVectorConversion::covariant_to_physical(q1_cov(k, j, i),
                inverse_h1_at_u(j, i));

            v(k, j, i) = HorizontalVectorConversion::covariant_to_physical(q2_cov(k, j, i),
                inverse_h2_at_v(j, i));
        });

    halo_exchanger_.exchange_multiple_halos(std::vector<Core::Field<3>*>{&u_field, &v_field});

    switch (options.boundary_policy) {
    case HorizontalDiagnosticBoundaryPolicy::FreeSlipBoundedQ2:
        if (!bounded_q2_stencils_) {
            throw std::logic_error("Bounded-q2 RLL wind commit requires "
                                   "horizontal boundary stencils.");
        }
        bounded_q2_stencils_->fill_regular_lat_lon_free_slip_physical_wind_halos(u_field, v_field);
        break;

    case HorizontalDiagnosticBoundaryPolicy::CvvmMode2Reference:
        if (!bounded_q2_stencils_) {
            throw std::logic_error("Reference RLL wind commit requires "
                                   "horizontal boundary stencils.");
        }
        bounded_q2_stencils_->fill_constant_q2_halos(u_field);
        bounded_q2_stencils_->fill_constant_q2_halos(v_field);
        break;

    case HorizontalDiagnosticBoundaryPolicy::PeriodicQ2:
        // MPI/halo exchange above supplies both horizontal directions.
        break;

    default:
        throw std::invalid_argument("Unknown horizontal diagnostic boundary policy.");
    }
}

void
WindSolver::finalize_regular_latlon_wind(
    Core::Field<3>& u_field, Core::Field<3>& v_field, const bool terrain) {

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int top = nz - h - 1;

    const auto u = u_field.get_mutable_device_data();
    const auto v = v_field.get_mutable_device_data();

    Kokkos::parallel_for("RLLWindVerticalGhosts",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            if (k < h - 1) {
                u(k, j, i) = u(h - 1, j, i);
                v(k, j, i) = v(h - 1, j, i);
            }

            if (k > top) {
                u(k, j, i) = u(top, j, i);
                v(k, j, i) = v(top, j, i);
            }
        });

    halo_exchanger_.exchange_multiple_halos(std::vector<Core::Field<3>*>{&u_field, &v_field});

    if (bounded_q2_stencils_) {
        bounded_q2_stencils_->fill_regular_lat_lon_free_slip_physical_wind_halos(u_field, v_field);
    }

    // Refresh the ordinary terrain scratch diagnostics for output and the next
    // adaptation. Prognostic xi/eta are never overwritten with solid-cell curl.
    if (terrain) {
        adapt_terrain(state_,
            grid_,
            params_,
            rll_inverse_dz_,
            halo_exchanger_,
            bounded_q2_stencils_.get());
    }

    // Keep persistent canonical wind synchronized with the final physical RLL
    // representation, including any PeriodicQ2 or bounded-q2 circulation
    // correction applied after generalized recovery.
    sync_regular_lat_lon_contravariant_wind_from_physical();
}

void
WindSolver::execute_generalized_wind_diagnostic(const bool initial,
    GeneralizedWindDiagnosticFields& fields,
    HorizontalDiagnosticWorkspace& workspace,
    const GeneralizedWindDiagnosticOptions& options) {

    const auto diagnose = [&]() {
        GeneralizedWindDiagnostic::diagnose(grid_,
            halo_exchanger_,
            *rll_vertical_solver_,
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

    auto found = rll_graphs_.find(key);

    const auto stream = Kokkos::DefaultExecutionSpace().cuda_stream();

    if (found == rll_graphs_.end()) {

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

        found = rll_graphs_.emplace(key, executable).first;
    }

    require_cuda(cudaGraphLaunch(found->second, stream),
        "Replay generalized wind diagnostic graph");

#else

    diagnose();

#endif
}

void
WindSolver::recover_regular_latlon_horizontal_wind(const bool initial,
    const bool terrain,
    GeneralizedWindDiagnosticFields& fields,
    HorizontalDiagnosticWorkspace& workspace,
    const GeneralizedWindDiagnosticOptions& options) {
    // Produce recovered covariant horizontal wind.
    execute_generalized_wind_diagnostic(initial, fields, workspace, options);

    snapshot_regular_latlon_top_vertical_vorticity();

    if (!horizontal_wind_constraint_) {
        throw std::logic_error("RLL wind recovery requires an initialized "
                               "topology constraint.");
    }

    // Circulation constraints now operate directly on the recovered
    // covariant wind.
    horizontal_wind_constraint_->after_recovery(initial);

    // Cross the RLL representation boundary only after the topology
    // correction has been applied.
    commit_regular_latlon_recovered_wind(fields, options);

    finalize_regular_latlon_wind(state_.get_field<3>("u"), state_.get_field<3>("v"), terrain);
}

void
WindSolver::sync_regular_lat_lon_contravariant_wind_from_physical() {
    using Core::Geometry::HorizontalLocation;
    using Operators::HorizontalVectorConversion;

    if (grid_.geometry().kind() != Core::Geometry::GeometryKind::RegularLatLon) {

        throw std::logic_error("RLL physical-to-contravariant wind synchronization "
                               "requires RegularLatLon geometry.");
    }

    const auto u = state_.get_field<3>("u").get_device_data();

    const auto v = state_.get_field<3>("v").get_device_data();

    auto u_con = state_.get_field<3>("u_con").get_mutable_device_data();

    auto v_con = state_.get_field<3>("v_con").get_mutable_device_data();

    const auto inverse_h1_at_u =
        grid_.geometry().device_view(HorizontalLocation::U).physical_to_contravariant.a11;

    const auto inverse_h2_at_v =
        grid_.geometry().device_view(HorizontalLocation::V).physical_to_contravariant.a22;

    const int nz = grid_.get_local_total_points_z();

    const int ny = grid_.get_local_total_points_y();

    const int nx = grid_.get_local_total_points_x();

    Kokkos::parallel_for("SyncRegularLatLonContravariantWindFromPhysical",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u_con(k, j, i) = HorizontalVectorConversion::physical_to_contravariant(u(k, j, i),
                inverse_h1_at_u(j, i));

            v_con(k, j, i) = HorizontalVectorConversion::physical_to_contravariant(v(k, j, i),
                inverse_h2_at_v(j, i));
        });
}

} // namespace VVM::Dynamics
