#include "dynamics/solvers/WindSolver.hpp"
#include "dynamics/solvers/VerticalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindStateAdapter.hpp"
#include "core/RegularLatLonModelConfiguration.hpp"
#include "dynamics/operators/RegularLatLonTerrain.hpp"

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

    rll_increment_ =
        std::make_unique<Core::Field<0>>("RLL harmonic increment", std::array<int, 0>{});

    rll_wall_contributions_ = std::make_unique<Core::Field<1>>("RLL circulation contributions",
        std::array<int, 1>{
            grid_.get_global_points_x() + (periodic ? 2 * grid_.get_global_points_y() : 0)});

    Kokkos::deep_copy(rll_spacing_->get_mutable_device_data(), params_.get_value_host(params_.dz));

    rll_vertical_solver_ = std::make_unique<VerticalEllipticSolver>(grid_,
        halo_exchanger_,
        state_.get_field<1>("rhobar"),
        state_.get_field<1>("rhobar_up"),
        params_.flex_height_coef_mid,
        params_.flex_height_coef_up,
        rll_inverse_dz_,
        params_.get_value_host(params_.WRXMU));

    prepare_regular_latlon_diagnostic_execution();
}

void
WindSolver::solve_regular_latlon() {
    const bool initial = !rll_initialized_;
    const bool q2_periodic =
        grid_.horizontal_specification().topology.q2 == Core::HorizontalEdgeTopology::Periodic;
    const bool terrain = Core::is_rll_mountain(config_);
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    if (initial) {
        initialize_regular_latlon_solver(q2_periodic, nz);
    }
    RegularLatLonDiagnosticOptions options;
    if (q2_periodic) {
        options.boundary_policy = HorizontalDiagnosticBoundaryPolicy::RegularLatLonPeriodic;
        if (initial) {
            preserve_regular_latlon_periodic_circulation(true);
        }
    }
    options.vertical_iterations = config_.get_value<int>("dynamics.solver.vertical_iterations");
    options.horizontal = horizontal_elliptic_options_;
    options.horizontal.channel_psi_north = rll_psi_north_;
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
    }
    RegularLatLonDiagnosticFields fields{state_.get_field<2>("psi"),
        state_.get_field<2>("psinm1"),
        state_.get_field<2>("chi"),
        state_.get_field<2>("chinm1"),
        state_.get_field<3>("zeta"),
        state_.get_field<3>("w"),
        state_.get_field<3>("W3DNM1"),
        state_.get_field<3>(terrain ? "xi_topo" : "xi"),
        state_.get_field<3>(terrain ? "eta_topo" : "eta"),
        state_.get_field<3>("u"),
        state_.get_field<3>("v"),
        state_.get_field<1>("rhobar"),
        state_.get_field<1>("rhobar_up"),
        params_.flex_height_coef_mid,
        *rll_spacing_,
        *rll_increment_};

    HorizontalDiagnosticWorkspace workspace{rhs_psi_field_,
        rhs_chi_field_,
        psi_out_field_,
        chi_out_field_};

    recover_regular_latlon_horizontal_wind(initial,
        q2_periodic,
        terrain,
        fields,
        workspace,
        options);

    rll_initialized_ = true;
}

void
WindSolver::preserve_regular_latlon_channel_circulation(const bool initialize) {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int top = nz - h - 1;

    // Kelvin circulation at the southern wall is a separate harmonic degree
    // of freedom. Preserve its initialized zonal integral of covariant u_1.
    // This correction has zero discrete curl and divergence on RLL.
    const auto h1 = grid_.geometry()
                        .device_view(Core::Geometry::HorizontalLocation::U)
                        .contravariant_to_physical.a11;

    const auto u = state_.get_field<3>("u").get_mutable_device_data();

    const auto zeta = state_.get_field<3>("zeta").get_device_data();

    const auto top_snapshot = state_.get_field<2>("rll_zeta_top").get_mutable_device_data();

    const auto row = rll_wall_contributions_->get_mutable_device_data();

    Kokkos::deep_copy(row, real(0.));

    const bool owns_south = grid_.get_local_physical_start_y() == 0;

    const int start_i = grid_.get_local_physical_start_x();

    Kokkos::parallel_for("RLLSouthWallCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(int j, int i) {
            if (owns_south && j == h) {
                row(start_i + i - h) = h1(j, i) * u(top, j, i);
            }

            // Optional compact history uses the shared State/output machinery;
            // the prognostic three-dimensional zeta field is unchanged.
            top_snapshot(j, i) = zeta(top, j, i);
        });

    // Each longitude has exactly one owner. Reduce disjoint contributions
    // (only additions to zero), then sum in the same global order on all ranks.
    // An ordinary parallel mean has rank-dependent rounding that seeds a
    // different harmonic wind correction every step. This synchronization
    // remains outside capture, as did the original horizontal mean.
    auto wall = rll_wall_contributions_->get_host_data();

    MPI_Allreduce(MPI_IN_PLACE,
        wall.data(),
        grid_.get_global_points_x(),
        VVM_MPI_REAL,
        MPI_SUM,
        grid_.get_comm());

    Real current = real(0.);

    for (int i = 0; i < grid_.get_global_points_x(); ++i) {
        current += wall(i);
    }

    current /= static_cast<Real>(grid_.get_global_points_x());

    if (initialize) {
        rll_south_circulation_ = current;
    }

    const Real correction = rll_south_circulation_ - current;

    Kokkos::parallel_for("PreserveRLLWallCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(int k, int j, int i) { u(k, j, i) += correction / h1(j, i); });
}

void
WindSolver::preserve_regular_latlon_periodic_circulation(bool initialize) {
    // The repeating patch has two harmonic modes, unlike a channel. Fix their
    // cycle integrals to the prescribed initial top wind. This is an explicit
    // idealized circulation constraint, not Cartesian physical-mean subtraction
    // or a prognostic large-scale momentum equation. No RHS projection is made.
    // With wrapped metrics, u=C/h1(U) and v=D/h1(V) each have zero discrete curl
    // and divergence, including at the seam. Reductions remain outside capture.
    const int h = grid_.get_halo_cells(), nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y(), nx = grid_.get_local_total_points_x();
    const int gx = grid_.get_global_points_x(), gy = grid_.get_global_points_y();
    const int si = grid_.get_local_physical_start_x(), sj = grid_.get_local_physical_start_y();
    const int top = nz - h - 1;
    const auto u = state_.get_field<3>("u").get_mutable_device_data();
    const auto v = state_.get_field<3>("v").get_mutable_device_data();
    const auto zeta = state_.get_field<3>("zeta").get_device_data();
    const auto snapshot = state_.get_field<2>("rll_zeta_top").get_mutable_device_data();
    const auto hu = grid_.geometry()
                        .device_view(Core::Geometry::HorizontalLocation::U)
                        .contravariant_to_physical.a11;
    const auto hv = grid_.geometry()
                        .device_view(Core::Geometry::HorizontalLocation::V)
                        .contravariant_to_physical.a11;
    const Real radius = grid_.horizontal_specification().geometry.regular_lat_lon.radius;
    const auto contributions = rll_wall_contributions_->get_mutable_device_data();
    Kokkos::deep_copy(contributions, real(0.));
    Kokkos::parallel_for("RLLPeriodicCycleIntegrals",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(int j, int i) {
            if (sj == 0 && j == h) {
                contributions(si + i - h) = hu(j, i) * u(top, j, i);
            }
            if (si == 0 && i == h) {
                contributions(gx + sj + j - h) = radius * v(top, j, i);
                contributions(gx + gy + sj + j - h) = radius / hv(j, i);
            }
            snapshot(j, i) = zeta(top, j, i);
        });
    auto values = rll_wall_contributions_->get_host_data();
    MPI_Allreduce(MPI_IN_PLACE,
        values.data(),
        gx + 2 * gy,
        VVM_MPI_REAL,
        MPI_SUM,
        grid_.get_comm());
    Real zonal = real(0.), meridional = real(0.), weight = real(0.);
    for (int i = 0; i < gx; ++i) {
        zonal += values(i);
    }
    for (int j = 0; j < gy; ++j) {
        meridional += values(gx + j);
        weight += values(gx + gy + j);
    }
    zonal /= static_cast<Real>(gx);
    if (initialize) {
        rll_south_circulation_ = zonal;
        rll_meridional_circulation_ = meridional;
        return;
    }
    const Real du = rll_south_circulation_ - zonal;
    const Real dv = (rll_meridional_circulation_ - meridional) / weight;
    Kokkos::parallel_for("PreserveRLLPeriodicCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            u(k, j, i) += du / hu(j, i);
            v(k, j, i) += dv / hv(j, i);
        });
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
}

void
WindSolver::execute_regular_latlon_diagnostic(const bool initial,
    RegularLatLonDiagnosticFields& fields,
    HorizontalDiagnosticWorkspace& workspace,
    const RegularLatLonDiagnosticOptions& options) {

    const auto diagnose = [&]() {
        diagnose_regular_latlon_wind(grid_,
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

    // AB2 exchanges its two prognostic allocations. Capture once for each
    // backing allocation; all private solver/history storage stays fixed.
    const auto key = fields.zeta.get_device_data().data();

    auto found = rll_graphs_.find(key);

    const auto stream = Kokkos::DefaultExecutionSpace().cuda_stream();

    if (found == rll_graphs_.end()) {
        require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            "Begin RLL diagnostic capture");

        diagnose();

        cudaGraph_t graph = nullptr;

        require_cuda(cudaStreamEndCapture(stream, &graph), "End RLL diagnostic capture");

        cudaGraphExec_t executable = nullptr;

        const auto result = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);

        cudaGraphDestroy(graph);

        require_cuda(result, "Instantiate RLL diagnostic graph");

        found = rll_graphs_.emplace(key, executable).first;
    }

    require_cuda(cudaGraphLaunch(found->second, stream), "Replay RLL diagnostic graph");

#else

    diagnose();

#endif
}

void
WindSolver::apply_regular_latlon_wind_closure(const bool initial, const bool periodic) {

    if (periodic) {
        preserve_regular_latlon_periodic_circulation(false);
    }
    else {
        preserve_regular_latlon_channel_circulation(initial);
    }
}

void
WindSolver::recover_regular_latlon_horizontal_wind(const bool initial,
    const bool periodic,
    const bool terrain,
    RegularLatLonDiagnosticFields& fields,
    HorizontalDiagnosticWorkspace& workspace,
    const RegularLatLonDiagnosticOptions& options) {

    execute_regular_latlon_diagnostic(initial, fields, workspace, options);

    apply_regular_latlon_wind_closure(initial, periodic);

    finalize_regular_latlon_wind(fields.u, fields.v, terrain);
}

} // namespace VVM::Dynamics
