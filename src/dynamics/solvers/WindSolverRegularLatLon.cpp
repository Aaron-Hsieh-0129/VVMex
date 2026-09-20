#include "dynamics/solvers/WindSolver.hpp"
#include "dynamics/solvers/VerticalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindStateAdapter.hpp"
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

#if defined(ENABLE_NCCL)
void
require_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void
require_nccl(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(result));
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

    rll_prescribed_zonal_covariant_increment_ =
        std::make_unique<Core::Field<0>>("RLL prescribed zonal covariant increment",
            std::array<int, 0>{});

    rll_constraint_contributions_ =
        std::make_unique<Core::Field<1>>("RLL harmonic constraint contributions",
            std::array<int, 1>{
                grid_.get_global_points_x() + (periodic ? 2 * grid_.get_global_points_y() : 0)});

#if defined(ENABLE_NCCL)
    rll_harmonic_targets_device_ =
        Kokkos::View<VVM::Real*, Kokkos::DefaultExecutionSpace::memory_space>(
            "RLL harmonic constraint targets",
            2);

    rll_harmonic_measurements_device_ =
        Kokkos::View<VVM::Real*, Kokkos::DefaultExecutionSpace::memory_space>(
            "RLL harmonic constraint measurements",
            3);

    Kokkos::deep_copy(rll_harmonic_targets_device_, VVM::real(0.0));

    Kokkos::deep_copy(rll_harmonic_measurements_device_, VVM::real(0.0));
#endif

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
    const int nz = grid_.get_local_total_points_z();
    if (initial) {
        initialize_regular_latlon_solver(q2_periodic, nz);
    }
    RegularLatLonDiagnosticOptions options;
    if (q2_periodic) {
        options.boundary_policy = HorizontalDiagnosticBoundaryPolicy::RegularLatLonPeriodic;
        // This target belongs to the incoming physical wind, before either
        // terrain adaptation or the initial diagnostic can modify it.
        if (initial) {
            maintain_horizontal_wind_constraint(
                HorizontalWindConstraintKind::PeriodicCycleCirculation,
                true);
        }
    }
    // Assemble this invocation's field references after the initial periodic
    // target has been captured and before diagnostic execution / graph capture.
    auto fields = prepare_regular_latlon_wind_recovery(initial, terrain, options);

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

WindSolver::RegularLatLonDiagnosticFields
WindSolver::prepare_regular_latlon_wind_recovery(
    const bool initial, const bool terrain, RegularLatLonDiagnosticOptions& options) {
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

    return RegularLatLonDiagnosticFields{state_.get_field<2>("psi"),
        state_.get_field<2>("psinm1"),
        state_.get_field<2>("chi"),
        state_.get_field<2>("chinm1"),
        state_.get_field<3>("zeta"),
        state_.get_field<3>("w"),
        state_.get_field<3>("W3DNM1"),
        // Current physical representation.
        state_.get_field<3>(terrain ? "xi_topo" : "xi"),
        state_.get_field<3>(terrain ? "eta_topo" : "eta"),
        // Canonical persistent representation.
        state_.get_field<3>("xi_con"),
        state_.get_field<3>("eta_con"),
        // Solver-private covariant scratch.
        *rll_covariant_q1_wind_,
        *rll_covariant_q2_wind_,
        state_.get_field<3>("u"),
        state_.get_field<3>("v"),
        state_.get_field<1>("rhobar"),
        state_.get_field<1>("rhobar_up"),
        params_.flex_height_coef_mid,
        *rll_spacing_,
        *rll_prescribed_zonal_covariant_increment_};
}

void
WindSolver::measure_regular_latlon_channel_circulation() {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int gx = grid_.get_global_points_x();
    const int top = nz - h - 1;

    const auto h1_at_u = grid_.geometry()
                             .device_view(Core::Geometry::HorizontalLocation::U)
                             .contravariant_to_physical.a11;

    const auto u = state_.get_field<3>("u").get_device_data();
    const auto zeta = state_.get_field<3>("zeta").get_device_data();
    const auto top_snapshot = state_.get_field<2>("rll_zeta_top").get_mutable_device_data();
    const auto contributions = rll_constraint_contributions_->get_mutable_device_data();

    Kokkos::deep_copy(contributions, VVM::real(0.0));
    const bool owns_south = grid_.get_local_physical_start_y() == 0;
    const int start_i = grid_.get_local_physical_start_x();

    Kokkos::parallel_for("RLLSouthWallCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (owns_south && j == h) {
                contributions(start_i + i - h) =
                    Operators::HorizontalVectorConversion::physical_to_covariant(u(top, j, i),
                        h1_at_u(j, i));
            }

            top_snapshot(j, i) = zeta(top, j, i);
        });

#if defined(ENABLE_NCCL)
    const cudaStream_t nccl_stream = state_.get_cuda_stream();
    const cudaStream_t kokkos_stream = Kokkos::DefaultExecutionSpace().cuda_stream();
    const bool separate_nccl_stream = nccl_stream != kokkos_stream;

    if (separate_nccl_stream) {
        Kokkos::DefaultExecutionSpace().fence("RLL channel circulation producer ready");
    }

    require_nccl(ncclAllReduce(contributions.data(),
                     contributions.data(),
                     static_cast<size_t>(gx),
                     VVM_NCCL_REAL,
                     ncclSum,
                     state_.get_nccl_comm(),
                     nccl_stream),
        "RLL channel circulation NCCL all-reduce");

    if (separate_nccl_stream) {
        require_cuda(cudaStreamSynchronize(nccl_stream),
            "RLL channel circulation NCCL synchronization");
    }

    const auto measurements = rll_harmonic_measurements_device_;

    Kokkos::parallel_for("RLLSouthWallCirculationFinalize",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int) {
            Real current = VVM::real(0.0);

            for (int i = 0; i < gx; ++i) {
                current += contributions(i);
            }

            measurements(0) = current / static_cast<Real>(gx);
        });

#else

    auto wall = rll_constraint_contributions_->get_host_data();
    const int mpi_result =
        MPI_Allreduce(MPI_IN_PLACE, wall.data(), gx, VVM_MPI_REAL, MPI_SUM, grid_.get_comm());
    if (mpi_result != MPI_SUCCESS) {
        throw std::runtime_error("RLL channel circulation MPI_Allreduce failed.");
    }
    Real current = VVM::real(0.0);

    for (int i = 0; i < gx; ++i) {
        current += wall(i);
    }

    current /= static_cast<Real>(gx);

    rll_harmonic_measurements_.q1 = current;

#endif
}

void
WindSolver::capture_regular_latlon_channel_circulation_target() {
#if defined(ENABLE_NCCL)

    const auto targets = rll_harmonic_targets_device_;

    const auto measurements = rll_harmonic_measurements_device_;

    Kokkos::parallel_for("CaptureRLLChannelCirculationTarget",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int) { targets(0) = measurements(0); });

#else

    rll_harmonic_targets_.q1 = rll_harmonic_measurements_.q1;

#endif
}

void
WindSolver::apply_regular_latlon_channel_circulation_correction() {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const auto h1 = grid_.geometry()
                        .device_view(Core::Geometry::HorizontalLocation::U)
                        .contravariant_to_physical.a11;

    const auto u = state_.get_field<3>("u").get_mutable_device_data();

#if defined(ENABLE_NCCL)

    const auto targets = rll_harmonic_targets_device_;

    const auto measurements = rll_harmonic_measurements_device_;

    Kokkos::parallel_for("PreserveRLLWallCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real correction = targets(0) - measurements(0);

            u(k, j, i) += correction / h1(j, i);
        });

#else

    const Real correction = rll_harmonic_targets_.q1 - rll_harmonic_measurements_.q1;

    Kokkos::parallel_for("PreserveRLLWallCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u(k, j, i) += correction / h1(j, i);
        });

#endif
}

void
WindSolver::preserve_regular_latlon_channel_circulation(const bool initialize) {

    measure_regular_latlon_channel_circulation();

    if (initialize) {
        capture_regular_latlon_channel_circulation_target();
    }

    // Preserve the existing behavior:
    // initialization still executes the zero-correction kernel.
    apply_regular_latlon_channel_circulation_correction();
}

void
WindSolver::measure_regular_latlon_periodic_circulation() {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const int gx = grid_.get_global_points_x();
    const int gy = grid_.get_global_points_y();

    const int si = grid_.get_local_physical_start_x();
    const int sj = grid_.get_local_physical_start_y();

    const int top = nz - h - 1;

    const auto u = state_.get_field<3>("u").get_device_data();
    const auto v = state_.get_field<3>("v").get_device_data();
    const auto zeta = state_.get_field<3>("zeta").get_device_data();
    const auto snapshot = state_.get_field<2>("rll_zeta_top").get_mutable_device_data();

    const auto h1_at_u = grid_.geometry()
                             .device_view(Core::Geometry::HorizontalLocation::U)
                             .contravariant_to_physical.a11;

    const auto h2_at_v = grid_.geometry()
                             .device_view(Core::Geometry::HorizontalLocation::V)
                             .contravariant_to_physical.a22;

    const auto h1_at_v = grid_.geometry()
                             .device_view(Core::Geometry::HorizontalLocation::V)
                             .contravariant_to_physical.a11;

    const Real radius = grid_.horizontal_specification().geometry.regular_lat_lon.radius;

    const auto contributions = rll_constraint_contributions_->get_mutable_device_data();

    Kokkos::deep_copy(contributions, VVM::real(0.0));

    Kokkos::parallel_for("RLLPeriodicCycleIntegrals",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (sj == 0 && j == h) {
                contributions(si + i - h) =
                    Operators::HorizontalVectorConversion::physical_to_covariant(u(top, j, i),
                        h1_at_u(j, i));
            }

            if (si == 0 && i == h) {
                contributions(gx + sj + j - h) =
                    Operators::HorizontalVectorConversion::physical_to_covariant(v(top, j, i),
                        h2_at_v(j, i));

                contributions(gx + gy + sj + j - h) = radius / h1_at_v(j, i);
            }

            snapshot(j, i) = zeta(top, j, i);
        });

    const int contribution_count = gx + 2 * gy;

#if defined(ENABLE_NCCL)
    const cudaStream_t nccl_stream = state_.get_cuda_stream();
    const cudaStream_t kokkos_stream = Kokkos::DefaultExecutionSpace().cuda_stream();
    const bool separate_nccl_stream = nccl_stream != kokkos_stream;

    if (separate_nccl_stream) {
        Kokkos::DefaultExecutionSpace().fence("RLL periodic circulation producer ready");
    }

    require_nccl(ncclAllReduce(contributions.data(),
                     contributions.data(),
                     static_cast<size_t>(contribution_count),
                     VVM_NCCL_REAL,
                     ncclSum,
                     state_.get_nccl_comm(),
                     nccl_stream),
        "RLL periodic circulation NCCL all-reduce");

    if (separate_nccl_stream) {
        require_cuda(cudaStreamSynchronize(nccl_stream),
            "RLL periodic circulation NCCL synchronization");
    }

    const auto measurements = rll_harmonic_measurements_device_;

    Kokkos::parallel_for("RLLPeriodicCycleIntegralsFinalize",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int) {
            Real zonal = VVM::real(0.0);

            Real meridional = VVM::real(0.0);

            Real weight = VVM::real(0.0);

            for (int i = 0; i < gx; ++i) {
                zonal += contributions(i);
            }

            for (int j = 0; j < gy; ++j) {
                meridional += contributions(gx + j);

                weight += contributions(gx + gy + j);
            }

            measurements(0) = zonal / static_cast<Real>(gx);
            measurements(1) = meridional;
            measurements(2) = weight;
        });

#else
    auto values = rll_constraint_contributions_->get_host_data();

    const int mpi_result = MPI_Allreduce(MPI_IN_PLACE,
        values.data(),
        contribution_count,
        VVM_MPI_REAL,
        MPI_SUM,
        grid_.get_comm());

    if (mpi_result != MPI_SUCCESS) {
        throw std::runtime_error("RLL periodic circulation MPI_Allreduce failed.");
    }

    Real zonal = VVM::real(0.0);
    Real meridional = VVM::real(0.0);
    Real weight = VVM::real(0.0);

    for (int i = 0; i < gx; ++i) {
        zonal += values(i);
    }

    for (int j = 0; j < gy; ++j) {
        meridional += values(gx + j);
        weight += values(gx + gy + j);
    }

    zonal /= static_cast<Real>(gx);

    rll_harmonic_measurements_.q1 = zonal;
    rll_harmonic_measurements_.q2 = meridional;
    rll_harmonic_measurements_.weight = weight;

#endif
}

void
WindSolver::capture_regular_latlon_periodic_circulation_target() {
#if defined(ENABLE_NCCL)
    const auto targets = rll_harmonic_targets_device_;
    const auto measurements = rll_harmonic_measurements_device_;

    Kokkos::parallel_for("CaptureRLLPeriodicCirculationTarget",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int) {
            targets(0) = measurements(0);
            targets(1) = measurements(1);
        });

#else
    rll_harmonic_targets_.q1 = rll_harmonic_measurements_.q1;
    rll_harmonic_targets_.q2 = rll_harmonic_measurements_.q2;
#endif
}

void
WindSolver::apply_regular_latlon_periodic_circulation_correction() {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const auto u = state_.get_field<3>("u").get_mutable_device_data();
    const auto v = state_.get_field<3>("v").get_mutable_device_data();
    const auto hu = grid_.geometry()
                        .device_view(Core::Geometry::HorizontalLocation::U)
                        .contravariant_to_physical.a11;
    const auto hv = grid_.geometry()
                        .device_view(Core::Geometry::HorizontalLocation::V)
                        .contravariant_to_physical.a11;

#if defined(ENABLE_NCCL)
    const auto targets = rll_harmonic_targets_device_;
    const auto measurements = rll_harmonic_measurements_device_;

    Kokkos::parallel_for("PreserveRLLPeriodicCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real du = targets(0) - measurements(0);
            const Real dv = (targets(1) - measurements(1)) / measurements(2);

            u(k, j, i) += du / hu(j, i);
            v(k, j, i) += dv / hv(j, i);
        });
#else
    const Real du = rll_harmonic_targets_.q1 - rll_harmonic_measurements_.q1;
    const Real dv = (rll_harmonic_targets_.q2 - rll_harmonic_measurements_.q2) /
                    rll_harmonic_measurements_.weight;

    Kokkos::parallel_for("PreserveRLLPeriodicCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u(k, j, i) += du / hu(j, i);
            v(k, j, i) += dv / hv(j, i);
        });

#endif
}

void
WindSolver::preserve_regular_latlon_periodic_circulation(const bool initialize) {
    measure_regular_latlon_periodic_circulation();

    if (initialize) {
        capture_regular_latlon_periodic_circulation_target();
        // Preserve current behavior:
        // initial periodic capture does not apply a correction.
        return;
    }
    apply_regular_latlon_periodic_circulation_correction();
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
    const bool terrain,
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
            options,
            terrain);
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
WindSolver::recover_regular_latlon_horizontal_wind(const bool initial,
    const bool periodic,
    const bool terrain,
    RegularLatLonDiagnosticFields& fields,
    HorizontalDiagnosticWorkspace& workspace,
    const RegularLatLonDiagnosticOptions& options) {

    execute_regular_latlon_diagnostic(initial, terrain, fields, workspace, options);

    maintain_horizontal_wind_constraint(
        periodic ? HorizontalWindConstraintKind::PeriodicCycleCirculation
                 : HorizontalWindConstraintKind::BoundedQ2WallCirculation,
        periodic ? false : initial);

    finalize_regular_latlon_wind(fields.u, fields.v, terrain);
}

} // namespace VVM::Dynamics
