#include "dynamics/solvers/WindSolver.hpp"
#include "dynamics/solvers/VerticalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindStateAdapter.hpp"
#include "core/RegularLatLonModelConfiguration.hpp"

namespace VVM::Dynamics {
namespace {
#if defined(ENABLE_NCCL)
void require_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#endif
}

void WindSolver::solve_regular_latlon() {
    const bool initial = !rll_initialized_;
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int top = nz - h - 1;
    if (initial) {
        Core::validate_jung2019_rll(config_, Core::GridSpecification::from_config(config_));
        rll_spacing_ = std::make_unique<Core::Field<1>>("RLL wind spacing", std::array<int, 1>{nz});
        rll_increment_ = std::make_unique<Core::Field<0>>("RLL harmonic increment", std::array<int, 0>{});
        // The admitted Section 4.2 configuration has uniform physical levels.
        Kokkos::deep_copy(rll_spacing_->get_mutable_device_data(), params_.get_value_host(params_.dz));
        rll_vertical_solver_ = std::make_unique<VerticalEllipticSolver>(grid_, halo_exchanger_,
            state_.get_field<1>("rhobar"), state_.get_field<1>("rhobar_up"),
            params_.flex_height_coef_mid, params_.flex_height_coef_up,
            params_.get_value_host(params_.rdz), params_.get_value_host(params_.WRXMU));
        prepare_regular_latlon_diagnostic_execution();
    }
    RegularLatLonDiagnosticOptions options;
    options.vertical_iterations = config_.get_value<int>("dynamics.solver.vertical_iterations");
    options.horizontal = horizontal_elliptic_options_;
    Kokkos::deep_copy(options.horizontal.channel_psi_north, state_.get_field<0>("rll_psi_north").get_device_data());
    options.horizontal.iterations = initial ? config_.get_value<int>("dynamics.solver.initial_iterations") : params_.solver_iteration;
    options.inverse_dz = params_.get_value_host(params_.rdz);
    RegularLatLonDiagnosticFields fields{
        state_.get_field<2>("psi"), state_.get_field<2>("psinm1"),
        state_.get_field<2>("chi"), state_.get_field<2>("chinm1"),
        state_.get_field<3>("zeta"), state_.get_field<3>("w"), state_.get_field<3>("W3DNM1"),
        state_.get_field<3>("xi"), state_.get_field<3>("eta"), state_.get_field<3>("u"), state_.get_field<3>("v"),
        state_.get_field<1>("rhobar"), state_.get_field<1>("rhobar_up"), params_.flex_height_coef_mid,
        *rll_spacing_, *rll_increment_};
    HorizontalDiagnosticWorkspace workspace{rhs_psi_field_, rhs_chi_field_, psi_out_field_, chi_out_field_};
    const auto diagnose = [&]() {
        diagnose_regular_latlon_wind(grid_, halo_exchanger_, *rll_vertical_solver_, horizontal_elliptic_solver_, fields, workspace, options);
    };
#if defined(ENABLE_NCCL)
    if (initial) {
        diagnose();
    } else {
        // AB2 exchanges its two prognostic allocations. Capture once for each
        // backing allocation; all private solver/history storage stays fixed.
        const auto key = fields.zeta.get_device_data().data();
        auto found = rll_graphs_.find(key);
        const auto stream = Kokkos::DefaultExecutionSpace().cuda_stream();
        if (found == rll_graphs_.end()) {
            Kokkos::fence();
            require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "Begin RLL diagnostic capture");
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
    }
#else
    diagnose();
#endif

    // Kelvin circulation at the southern wall is a separate harmonic degree
    // of freedom. Preserve its initialized zonal integral of covariant u_1.
    // This correction has zero discrete curl and divergence on RLL.
    const auto u = fields.u.get_mutable_device_data();
    const auto h1 = grid_.geometry().device_view(Core::Geometry::HorizontalLocation::U).contravariant_to_physical.a11;
    const auto row = psi_tmp_field_.get_mutable_device_data();
    const auto zeta = fields.zeta.get_device_data();
    const auto top_snapshot = state_.get_field<2>("rll_zeta_top").get_mutable_device_data();
    const bool owns_south = grid_.get_local_physical_start_y() == 0;
    const Real rows = static_cast<Real>(grid_.get_global_points_y());
    Kokkos::parallel_for("RLLSouthWallCirculation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(int j, int i) {
            row(j, i) = owns_south && j == h ? rows * h1(j, i) * u(top, j, i) : real(0.0);
            // Optional compact history uses the shared State/output machinery;
            // the prognostic three-dimensional zeta field is unchanged.
            top_snapshot(j, i) = zeta(top, j, i);
        });
    // This existing reduction synchronizes and runs outside solver capture.
    const auto mean = state_.get_field<0>("utop_mean_tmp").get_mutable_device_data();
    state_.calculate_horizontal_mean(psi_tmp_field_, mean);
    Real current;
    Kokkos::deep_copy(current, mean);
    if (initial) rll_south_circulation_ = current;
    const Real correction = rll_south_circulation_ - current;
    Kokkos::parallel_for("PreserveRLLWallCirculation", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(int k, int j, int i) { u(k, j, i) += correction / h1(j, i); });
    const auto v = fields.v.get_mutable_device_data();
    Kokkos::parallel_for("RLLWindVerticalGhosts", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            if (k < h - 1) { u(k, j, i) = u(h - 1, j, i); v(k, j, i) = v(h - 1, j, i); }
            if (k > top) { u(k, j, i) = u(top, j, i); v(k, j, i) = v(top, j, i); }
        });
    halo_exchanger_.exchange_multiple_halos(std::vector<Core::Field<3>*>{&fields.u, &fields.v});
    bounded_q2_stencils_->fill_regular_lat_lon_free_slip_physical_wind_halos(fields.u, fields.v);
    rll_initialized_ = true;
}
} // namespace VVM::Dynamics
