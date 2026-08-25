#include "AreaMeanNudging.hpp"
#include <netcdf.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace VVM {
namespace Dynamics {

AreaMeanNudging::AreaMeanNudging(const Utils::ConfigurationManager& config, 
                                 const Core::Grid& grid, 
                                 const Core::Parameters& params)
    : config_(config), grid_(grid), params_(params) 
{
    enable_ = config_.get_value<bool>("dynamics.forcings.areamn.enable", false);
    
    if (enable_) {
        uvtau_ = config_.get_value<VVM::Real>("dynamics.forcings.areamn.uvtau", 0.0);
        nudgelim_ = config_.get_value<VVM::Real>("dynamics.forcings.areamn.nudge_start_m", 0.0);

        const std::string target_source = config_.get_value<std::string>("dynamics.forcings.areamn.target_source", "initial");
        if (target_source == "netcdf") {
            use_netcdf_target_ = true;
            forcing_directory_ = config_.get_value<std::string>("dynamics.forcings.areamn.forcing_data.directory");
            forcing_file_prefix_ = config_.get_value<std::string>("dynamics.forcings.areamn.forcing_data.file_prefix", "ls_forcing_");
            forcing_interval_s_ = config_.get_value<VVM::Real>("dynamics.forcings.areamn.forcing_data.update_interval_s", 3600.0);
            if (forcing_interval_s_ <= real(0.0)) {
                throw std::runtime_error("AreaMeanNudging forcing update interval must be positive.");
            }
            if (!forcing_directory_.empty() && forcing_directory_.back() != '/') {
                forcing_directory_ += '/';
            }

            constant_upper_wind_ = config_.get_value<bool>("initial_conditions.constant_upper_wind.enable", false);
            constant_upper_wind_threshold_Pa_ = config_.get_value<VVM::Real>("initial_conditions.constant_upper_wind.pressure_threshold_Pa", 3000.0);
        }
        else if (target_source != "initial") {
            throw std::runtime_error(
                "Unsupported dynamics.forcings.areamn.target_source '" +
                target_source + "'. Expected 'initial' or 'netcdf'.");
        }
        
        VVM::Real total_pts = static_cast<VVM::Real>(grid_.get_global_points_x() * grid_.get_global_points_y());
        inv_total_xy_pts_ = 1.0 / total_pts;

        if (grid_.get_mpi_rank() == 0) {
            std::cout << "--- Initializing Area Mean Nudging (AREAMN) ---" << std::endl;
            std::cout << "  * UVTAU: " << uvtau_ << " s, Nudge Limit: " << nudgelim_ << " m" << std::endl;
            std::cout << "  * Target source: " << target_source << std::endl;
        }
    }
}

void AreaMeanNudging::check_nc_error(
    int status,
    const std::string& message) const
{
    if (status == NC_NOERR) return;

    if (grid_.get_mpi_rank() == 0) {
        std::cerr << "NetCDF error in AreaMeanNudging: "
                  << message << ": " << nc_strerror(status) << std::endl;
    }
    MPI_Abort(grid_.get_cart_comm(), 21);
    throw std::runtime_error(message + ": " + nc_strerror(status));
}

std::string AreaMeanNudging::forcing_filename(VVM::Real time) const {
    std::ostringstream filename;
    filename << forcing_directory_ << forcing_file_prefix_
             << std::setfill('0') << std::setw(6)
             << static_cast<long long>(std::llround(time)) << ".nc";
    return filename.str();
}

void AreaMeanNudging::load_wind_profiles(
    const std::string& filename,
    VVM::Real expected_time,
    Kokkos::View<VVM::Real*>& u_target,
    Kokkos::View<VVM::Real*>& v_target) const
{
    const int global_nz = grid_.get_global_points_z();
    std::vector<double> u_profile(global_nz);
    std::vector<double> v_profile(global_nz);

    if (grid_.get_mpi_rank() == 0) {
        int ncid = -1;
        check_nc_error(nc_open(filename.c_str(), NC_NOWRITE, &ncid), "Failed to open " + filename);

        int nz_dimid = -1;
        check_nc_error(nc_inq_dimid(ncid, "nz", &nz_dimid), "Missing nz dimension in " + filename);

        size_t file_nz = 0;
        check_nc_error(nc_inq_dimlen(ncid, nz_dimid, &file_nz), "Failed to read nz dimension in " + filename);
        if (file_nz != static_cast<size_t>(global_nz)) {
            nc_close(ncid);
            check_nc_error(NC_EINVAL, "Forcing nz in " + filename + " does not match the VVM grid");
        }

        long long file_time = 0;
        check_nc_error(nc_get_att_longlong(ncid, NC_GLOBAL, "time_seconds", &file_time), "Missing time_seconds in " + filename);
        const long long expected_seconds = static_cast<long long>(std::llround(expected_time));
        if (file_time != expected_seconds) {
            nc_close(ncid);
            check_nc_error(NC_EINVAL, "time_seconds in " + filename + " does not match its requested forcing time");
        }

        int u_varid = -1;
        int v_varid = -1;
        int pbar_varid = -1;
        check_nc_error(nc_inq_varid(ncid, "U", &u_varid), "Missing U in " + filename);
        check_nc_error(nc_inq_varid(ncid, "V", &v_varid), "Missing V in " + filename);
        check_nc_error(nc_inq_varid(ncid, "pbar", &pbar_varid), "Missing pbar in " + filename);

        std::vector<double> pbar_profile(global_nz);
        check_nc_error(nc_get_var_double(ncid, u_varid, u_profile.data()), "Failed to read U from " + filename);
        check_nc_error(nc_get_var_double(ncid, v_varid, v_profile.data()), "Failed to read V from " + filename);
        check_nc_error(nc_get_var_double(ncid, pbar_varid, pbar_profile.data()), "Failed to read pbar from " + filename);
        check_nc_error(nc_close(ncid), "Failed to close " + filename);

        if (constant_upper_wind_) {
            size_t first_upper_level = pbar_profile.size();
            for (size_t k = 0; k < pbar_profile.size(); ++k) {
                if (pbar_profile[k] <= constant_upper_wind_threshold_Pa_) {
                    first_upper_level = k;
                    break;
                }
            }
            if (first_upper_level < pbar_profile.size()) {
                for (size_t k = first_upper_level + 1; k < pbar_profile.size(); ++k) {
                    u_profile[k] = u_profile[first_upper_level];
                    v_profile[k] = v_profile[first_upper_level];
                }
            }
        }

        std::cout << "  - AREAMN loaded wind profiles: " << filename << std::endl;
    }

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    auto u_host = Kokkos::create_mirror_view(u_target);
    auto v_host = Kokkos::create_mirror_view(v_target);

    const auto fill_host_profiles = [&]() {
        for (int k = 0; k < global_nz; ++k) {
            u_host(k + h) = static_cast<VVM::Real>(u_profile[k]);
            v_host(k + h) = static_cast<VVM::Real>(v_profile[k]);
        }
        for (int k = 0; k < h; ++k) {
            u_host(k) = static_cast<VVM::Real>(u_profile.front());
            v_host(k) = static_cast<VVM::Real>(v_profile.front());
            u_host(nz - 1 - k) = static_cast<VVM::Real>(u_profile.back());
            v_host(nz - 1 - k) = static_cast<VVM::Real>(v_profile.back());
        }
    };

#if defined(ENABLE_NCCL)
    if (grid_.get_mpi_rank() == 0) {
        fill_host_profiles();
        Kokkos::deep_copy(u_target, u_host);
        Kokkos::deep_copy(v_target, v_host);
    }
    Kokkos::fence();

    ncclResult_t result = ncclBroadcast(u_target.data(), u_target.data(), nz, VVM_NCCL_REAL, 0, nccl_comm_, stream_);
    if (result != ncclSuccess) {
        throw std::runtime_error(
            "AreaMeanNudging NCCL U-profile broadcast failed: " +
            std::string(ncclGetErrorString(result)));
    }

    result = ncclBroadcast(v_target.data(), v_target.data(), nz, VVM_NCCL_REAL, 0, nccl_comm_, stream_);
    if (result != ncclSuccess) {
        throw std::runtime_error(
            "AreaMeanNudging NCCL V-profile broadcast failed: " +
            std::string(ncclGetErrorString(result)));
    }

    const cudaError_t sync_result = cudaStreamSynchronize(stream_);
    if (sync_result != cudaSuccess) {
        throw std::runtime_error(
            "AreaMeanNudging NCCL broadcast synchronization failed: " +
            std::string(cudaGetErrorString(sync_result)));
    }
#else
    MPI_Bcast(u_profile.data(), global_nz, MPI_DOUBLE, 0, grid_.get_comm());
    MPI_Bcast(v_profile.data(), global_nz, MPI_DOUBLE, 0, grid_.get_comm());
    fill_host_profiles();
    Kokkos::deep_copy(u_target, u_host);
    Kokkos::deep_copy(v_target, v_host);
#endif
}


void AreaMeanNudging::deterministic_global_sum(
    const Core::State& state, const VVM::Real* local, VVM::Real* global, int count) const {
    int comm_size = 1;
#if defined(ENABLE_NCCL)
    const ncclResult_t count_result = ncclCommCount(state.get_nccl_comm(), &comm_size);
    if (count_result != ncclSuccess) {
        throw std::runtime_error(
            "AreaMeanNudging could not query the NCCL communicator size: " +
            std::string(ncclGetErrorString(count_result)));
    }
#else
    MPI_Comm_size(grid_.get_comm(), &comm_size);
#endif

    const size_t needed = static_cast<size_t>(count) * static_cast<size_t>(comm_size);
    if (gather_buffer_.extent(0) < needed) {
        gather_buffer_ = Kokkos::View<VVM::Real*>("areamn_rank_sums", needed);
    }

#if defined(ENABLE_NCCL)
    const ncclResult_t result = ncclAllGather(
        local, gather_buffer_.data(), static_cast<size_t>(count), VVM_NCCL_REAL,
        state.get_nccl_comm(), state.get_cuda_stream());
    if (result != ncclSuccess) {
        throw std::runtime_error(
            "AreaMeanNudging NCCL all-gather failed: " +
            std::string(ncclGetErrorString(result)));
    }
    const cudaError_t sync_result = cudaStreamSynchronize(state.get_cuda_stream());
    if (sync_result != cudaSuccess) {
        throw std::runtime_error(
            "AreaMeanNudging all-gather synchronization failed: " +
            std::string(cudaGetErrorString(sync_result)));
    }
#else
    // Gathered over grid_.get_comm(), not the Cartesian communicator: the latter
    // is created with reorder=1, so its rank order need not match the order the
    // NCCL communicator was built with.
    std::vector<VVM::Real> host_local(static_cast<size_t>(count), VVM::real(0.0));
    std::vector<VVM::Real> host_gather(needed, VVM::real(0.0));
    Kokkos::deep_copy(
        Kokkos::View<VVM::Real*, Kokkos::HostSpace>(host_local.data(), count),
        Kokkos::View<const VVM::Real*>(local, count));
    const int mpi_result = MPI_Allgather(
        host_local.data(), count, VVM_MPI_REAL,
        host_gather.data(), count, VVM_MPI_REAL, grid_.get_comm());
    if (mpi_result != MPI_SUCCESS) {
        throw std::runtime_error("AreaMeanNudging MPI_Allgather failed.");
    }
    Kokkos::deep_copy(
        Kokkos::subview(gather_buffer_, std::make_pair(static_cast<size_t>(0), needed)),
        Kokkos::View<const VVM::Real*, Kokkos::HostSpace>(host_gather.data(), needed));
#endif

    auto gather = gather_buffer_;
    const int num_ranks = comm_size;
    Kokkos::View<VVM::Real*> out(global, count);
    Kokkos::parallel_for("areamn_ordered_global_sum",
        Kokkos::RangePolicy<>(0, count),
        KOKKOS_LAMBDA(const int c) {
            VVM::Real total = VVM::real(0.0);
            for (int r = 0; r < num_ranks; ++r) {
                total += gather(static_cast<size_t>(r) * static_cast<size_t>(count) + c);
            }
            out(c) = total;
        });
    Kokkos::fence();
}

void AreaMeanNudging::initialize(Core::State& state) {
    if (!enable_) return;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h  = grid_.get_halo_cells();
    int top_k = nz - h - 1;

    if (!state.has_field("areamn_xi0")) {
        state.add_field<1>(
            "areamn_xi0",
            {nz},
            Core::FieldMetadata{
                Core::GridStaggering::StaggeredYZ,
                "s-1",
                "area-mean nudging x-vorticity target"});
    }
    if (!state.has_field("areamn_eta0")) {
        state.add_field<1>(
            "areamn_eta0",
            {nz},
            Core::FieldMetadata{
                Core::GridStaggering::StaggeredXZ,
                "s-1",
                "area-mean nudging y-vorticity target"});
    }

    if (!state.has_field("areamn_zeta0_top")) state.add_field<0>("areamn_zeta0_top", {});
    if (!state.has_field("areamn_local_sum_xi")) state.add_field<1>("areamn_local_sum_xi", {nz});
    if (!state.has_field("areamn_global_sum_xi")) state.add_field<1>("areamn_global_sum_xi", {nz});
    if (!state.has_field("areamn_local_sum_eta")) state.add_field<1>("areamn_local_sum_eta", {nz});
    if (!state.has_field("areamn_global_sum_eta")) state.add_field<1>("areamn_global_sum_eta", {nz});
    if (!state.has_field("areamn_local_sum_zeta_top")) state.add_field<0>("areamn_local_sum_zeta_top", {});
    if (!state.has_field("areamn_global_sum_zeta_top")) state.add_field<0>("areamn_global_sum_zeta_top", {});
    if (!state.has_field("areamn_utopmn0")) state.add_field<0>("areamn_utopmn0", {});
    if (!state.has_field("areamn_vtopmn0")) state.add_field<0>("areamn_vtopmn0", {});
    if (!state.has_field("areamn_u_target")) {
        state.add_field<1>(
            "areamn_u_target",
            {nz},
            Core::FieldMetadata{
                Core::GridStaggering::StaggeredX,
                "m s-1",
                "area-mean nudging x-wind target"});
    }
    if (!state.has_field("areamn_v_target")) {
        state.add_field<1>(
            "areamn_v_target",
            {nz},
            Core::FieldMetadata{
                Core::GridStaggering::StaggeredY,
                "m s-1",
                "area-mean nudging y-wind target"});
    }

    const auto& xi   = xi_ref_.get(state, "xi").get_device_data();
    const auto& eta  = eta_ref_.get(state, "eta").get_device_data();
    const auto& zeta = zeta_ref_.get(state, "zeta").get_device_data();
    const auto& u    = u_ref_.get(state, "u").get_device_data();
    const auto& v    = v_ref_.get(state, "v").get_device_data();

    auto& l_sum_xi = areamn_local_sum_xi_ref_.get(state, "areamn_local_sum_xi").get_mutable_device_data();
    auto& l_sum_eta = areamn_local_sum_eta_ref_.get(state, "areamn_local_sum_eta").get_mutable_device_data();
    auto& g_sum_xi = areamn_global_sum_xi_ref_.get(state, "areamn_global_sum_xi").get_mutable_device_data();
    auto& g_sum_eta = areamn_global_sum_eta_ref_.get(state, "areamn_global_sum_eta").get_mutable_device_data();
    auto& l_sum_zeta = areamn_local_sum_zeta_top_ref_.get(state, "areamn_local_sum_zeta_top").get_mutable_device_data();
    auto& g_sum_zeta = areamn_global_sum_zeta_top_ref_.get(state, "areamn_global_sum_zeta_top").get_mutable_device_data();

    Kokkos::View<VVM::Real> l_sum_u("l_sum_u");
    Kokkos::View<VVM::Real> g_sum_u("g_sum_u");
    Kokkos::View<VVM::Real> l_sum_v("l_sum_v");
    Kokkos::View<VVM::Real> g_sum_v("g_sum_v");

    Kokkos::deep_copy(l_sum_xi, 0.0);
    Kokkos::deep_copy(l_sum_eta, 0.0);
    Kokkos::deep_copy(l_sum_zeta, 0.0);
    Kokkos::deep_copy(l_sum_u, 0.0);
    Kokkos::deep_copy(l_sum_v, 0.0);

    // Deterministic reductions. These sums were previously accumulated with
    // Kokkos::atomic_add: the order in which threads reach an atomic varies between
    // runs, and floating-point addition is not associative, so the area means -- and
    // therefore the whole solution -- were not reproducible run to run. A team
    // reduction sums in a fixed order for a fixed launch configuration, which is what
    // State::calculate_horizontal_mean already does.
    {
        using TeamPol = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using Member  = TeamPol::member_type;
        const int nj = ny - 2 * h, ni = nx - 2 * h;
        const long ncol = static_cast<long>(nj) * ni;
        // Pinned, not Kokkos::AUTO. AUTO derives the team size from the compiled
        // kernel's occupancy, so it can change when compiler flags change -- and a
        // different team size means a different reduction tree, which shifts the
        // last bits of the sum. Results stay reproducible run to run either way,
        // but only a fixed team size keeps them reproducible across builds.
        // Host backends cap a team at the thread-pool size, so clamp by concurrency
        // or Kokkos aborts. On CUDA concurrency far exceeds 256, leaving the GPU
        // team size -- and therefore the GPU reduction tree -- exactly as before.
        const int kTeamSize = std::min(256, std::max(1, Kokkos::DefaultExecutionSpace().concurrency()));

        Kokkos::parallel_for("AREAMN_Init_Local_Sum", TeamPol(nz - 2 * h, kTeamSize),
            KOKKOS_LAMBDA(const Member& team) {
                const int k = h + team.league_rank();
                VVM::Real sxi = 0.0, seta = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, ncol),
                    [&](const long c, VVM::Real& acc) {
                        acc += xi(k, h + static_cast<int>(c / ni), h + static_cast<int>(c % ni));
                    }, sxi);
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, ncol),
                    [&](const long c, VVM::Real& acc) {
                        acc += eta(k, h + static_cast<int>(c / ni), h + static_cast<int>(c % ni));
                    }, seta);
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    l_sum_xi(k)  = sxi;
                    l_sum_eta(k) = seta;
                });
            }
        );

        VVM::Real szeta = 0.0, su = 0.0, sv = 0.0;
        Kokkos::parallel_reduce("AREAMN_Init_Local_Sum_top",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i, VVM::Real& az, VVM::Real& au, VVM::Real& av) {
                az += zeta(top_k, j, i);
                au += u(top_k, j, i);
                av += v(top_k, j, i);
            }, szeta, su, sv);
        Kokkos::deep_copy(l_sum_zeta, szeta);
        Kokkos::deep_copy(l_sum_u, su);
        Kokkos::deep_copy(l_sum_v, sv);
    }

    deterministic_global_sum(state, l_sum_xi.data(), g_sum_xi.data(), nz);
    deterministic_global_sum(state, l_sum_eta.data(), g_sum_eta.data(), nz);
    deterministic_global_sum(state, l_sum_zeta.data(), g_sum_zeta.data(), 1);
    deterministic_global_sum(state, l_sum_u.data(), g_sum_u.data(), 1);
    deterministic_global_sum(state, l_sum_v.data(), g_sum_v.data(), 1);

    auto& xi0 = areamn_xi0_ref_.get(state, "areamn_xi0").get_mutable_device_data();
    auto& eta0 = areamn_eta0_ref_.get(state, "areamn_eta0").get_mutable_device_data();
    auto& zeta0_top = areamn_zeta0_top_ref_.get(state, "areamn_zeta0_top").get_mutable_device_data();
    auto& utopmn0 = areamn_utopmn0_ref_.get(state, "areamn_utopmn0").get_mutable_device_data();
    auto& vtopmn0 = areamn_vtopmn0_ref_.get(state, "areamn_vtopmn0").get_mutable_device_data();

    auto& utopmn = utopmn_ref_.get(state, "utopmn").get_mutable_device_data();
    auto& vtopmn = vtopmn_ref_.get(state, "vtopmn").get_mutable_device_data();

    VVM::Real inv_pts = inv_total_xy_pts_;

    Kokkos::parallel_for("AREAMN_Init_Save", nz, KOKKOS_LAMBDA(const int k) {
        xi0(k) = g_sum_xi(k) * inv_pts;
        eta0(k) = g_sum_eta(k) * inv_pts;

        if (k == top_k) {
            zeta0_top() = g_sum_zeta() * inv_pts;
            utopmn0() = g_sum_u() * inv_pts;
            vtopmn0() = g_sum_v() * inv_pts;
            
            utopmn() = utopmn0();
            vtopmn() = vtopmn0();
        }
    });

    if (use_netcdf_target_) {
#if defined(ENABLE_NCCL)
        nccl_comm_ = state.get_nccl_comm();
        stream_ = state.get_cuda_stream();
#endif

        u_T1_ = Kokkos::View<VVM::Real*>("areamn_u_T1", nz);
        u_T2_ = Kokkos::View<VVM::Real*>("areamn_u_T2", nz);
        v_T1_ = Kokkos::View<VVM::Real*>("areamn_v_T1", nz);
        v_T2_ = Kokkos::View<VVM::Real*>("areamn_v_T2", nz);

        const VVM::Real current_time = state.get_time();
        time_T1_ = std::floor(current_time / forcing_interval_s_) * forcing_interval_s_;
        time_T2_ = time_T1_ + forcing_interval_s_;

        load_wind_profiles(forcing_filename(time_T1_), time_T1_, u_T1_, v_T1_);
        load_wind_profiles(forcing_filename(time_T2_), time_T2_, u_T2_, v_T2_);
        update_forcing_target(state, current_time);
    }
}

void AreaMeanNudging::update_forcing_target(
    Core::State& state,
    VVM::Real current_time)
{
    if (!enable_ || !use_netcdf_target_) return;

    while (current_time >= time_T2_) {
        std::swap(u_T1_, u_T2_);
        std::swap(v_T1_, v_T2_);
        time_T1_ = time_T2_;
        time_T2_ += forcing_interval_s_;
        load_wind_profiles(forcing_filename(time_T2_), time_T2_, u_T2_, v_T2_);
    }

    VVM::Real weight =
        (current_time - time_T1_) / (time_T2_ - time_T1_);
    weight = std::max(real(0.0), std::min(real(1.0), weight));

    const int nz = grid_.get_local_total_points_z();
    const int h = grid_.get_halo_cells();
    const int top_k = nz - h - 1;

    auto& xi_target = areamn_xi0_ref_.get(state, "areamn_xi0").get_mutable_device_data();
    auto& eta_target = areamn_eta0_ref_.get(state, "areamn_eta0").get_mutable_device_data();
    auto& utop_target = areamn_utopmn0_ref_.get(state, "areamn_utopmn0").get_mutable_device_data();
    auto& vtop_target = areamn_vtopmn0_ref_.get(state, "areamn_vtopmn0").get_mutable_device_data();
    auto& u_target = areamn_u_target_ref_.get(state, "areamn_u_target").get_mutable_device_data();
    auto& v_target = areamn_v_target_ref_.get(state, "areamn_v_target").get_mutable_device_data();

    const auto u_T1 = u_T1_;
    const auto u_T2 = u_T2_;
    const auto v_T1 = v_T1_;
    const auto v_T2 = v_T2_;
    const auto rdz = params_.rdz;
    const auto flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    Kokkos::parallel_for(
        "AREAMN_Update_NetCDF_Target",
        Kokkos::RangePolicy<>(h, top_k),
        KOKKOS_LAMBDA(const int k) {
            const VVM::Real u_k = (real(1.0) - weight) * u_T1(k) + weight * u_T2(k);
            const VVM::Real u_kp1 = (real(1.0) - weight) * u_T1(k + 1) + weight * u_T2(k + 1);
            const VVM::Real v_k = (real(1.0) - weight) * v_T1(k) + weight * v_T2(k);
            const VVM::Real v_kp1 = (real(1.0) - weight) * v_T1(k + 1) + weight * v_T2(k + 1);

            u_target(k) = u_k;
            v_target(k) = v_k;
            eta_target(k) = -(u_kp1 - u_k) * rdz() * flex_height_coef_up(k);
            xi_target(k) = -(v_kp1 - v_k) * rdz() * flex_height_coef_up(k);
        });

    Kokkos::parallel_for(
        "AREAMN_Update_NetCDF_Top_Target",
        1,
        KOKKOS_LAMBDA(const int) {
            xi_target(top_k) = real(0.0);
            eta_target(top_k) = real(0.0);
            u_target(top_k) = (real(1.0) - weight) * u_T1(top_k) + weight * u_T2(top_k);
            v_target(top_k) = (real(1.0) - weight) * v_T1(top_k) + weight * v_T2(top_k);
            utop_target() = u_target(top_k);
            vtop_target() = v_target(top_k);
        });
}

void AreaMeanNudging::apply_vorticity(Core::State& state, VVM::Real dt) {
    if (!enable_) return;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h  = grid_.get_halo_cells();
    int top_k = nz - h - 1;

    auto& xi = xi_ref_.get(state, "xi").get_mutable_device_data();
    auto& eta = eta_ref_.get(state, "eta").get_mutable_device_data();
    auto& zeta = zeta_ref_.get(state, "zeta").get_mutable_device_data();

    const auto& itypeu = ITYPEU_ref_.get(state, "ITYPEU").get_device_data();
    const auto& itypev = ITYPEV_ref_.get(state, "ITYPEV").get_device_data();

    auto& l_sum_xi = areamn_local_sum_xi_ref_.get(state, "areamn_local_sum_xi").get_mutable_device_data();
    auto& l_sum_eta = areamn_local_sum_eta_ref_.get(state, "areamn_local_sum_eta").get_mutable_device_data();
    auto& g_sum_xi = areamn_global_sum_xi_ref_.get(state, "areamn_global_sum_xi").get_mutable_device_data();
    auto& g_sum_eta = areamn_global_sum_eta_ref_.get(state, "areamn_global_sum_eta").get_mutable_device_data();
    auto& l_sum_zeta = areamn_local_sum_zeta_top_ref_.get(state, "areamn_local_sum_zeta_top").get_mutable_device_data();
    auto& g_sum_zeta = areamn_global_sum_zeta_top_ref_.get(state, "areamn_global_sum_zeta_top").get_mutable_device_data();

    Kokkos::deep_copy(l_sum_xi, 0.0);
    Kokkos::deep_copy(l_sum_eta, 0.0);
    Kokkos::deep_copy(l_sum_zeta, 0.0);

    // NOTE: This might need to consider topography to do the average but now just follow Fortran VVM
    // Deterministic reductions -- see the note at AREAMN_Init_Local_Sum above.
    {
        using TeamPol = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using Member  = TeamPol::member_type;
        const int nj = ny - 2 * h, ni = nx - 2 * h;
        const long ncol = static_cast<long>(nj) * ni;
        // Pinned, not Kokkos::AUTO. AUTO derives the team size from the compiled
        // kernel's occupancy, so it can change when compiler flags change -- and a
        // different team size means a different reduction tree, which shifts the
        // last bits of the sum. Results stay reproducible run to run either way,
        // but only a fixed team size keeps them reproducible across builds.
        // Host backends cap a team at the thread-pool size, so clamp by concurrency
        // or Kokkos aborts. On CUDA concurrency far exceeds 256, leaving the GPU
        // team size -- and therefore the GPU reduction tree -- exactly as before.
        const int kTeamSize = std::min(256, std::max(1, Kokkos::DefaultExecutionSpace().concurrency()));

        Kokkos::parallel_for("AREAMN_Sum", TeamPol(nz - 2 * h, kTeamSize),
            KOKKOS_LAMBDA(const Member& team) {
                const int k = h + team.league_rank();
                VVM::Real sxi = 0.0, seta = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, ncol),
                    [&](const long c, VVM::Real& acc) {
                        acc += xi(k, h + static_cast<int>(c / ni), h + static_cast<int>(c % ni));
                    }, sxi);
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, ncol),
                    [&](const long c, VVM::Real& acc) {
                        acc += eta(k, h + static_cast<int>(c / ni), h + static_cast<int>(c % ni));
                    }, seta);
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    l_sum_xi(k)  = sxi;
                    l_sum_eta(k) = seta;
                });
            }
        );

        VVM::Real szeta = 0.0;
        Kokkos::parallel_reduce("AREAMN_Sum_top",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i, VVM::Real& az) {
                az += zeta(top_k, j, i);
            }, szeta);
        Kokkos::deep_copy(l_sum_zeta, szeta);
    }

    deterministic_global_sum(state, l_sum_xi.data(), g_sum_xi.data(), nz);
    deterministic_global_sum(state, l_sum_eta.data(), g_sum_eta.data(), nz);
    deterministic_global_sum(state, l_sum_zeta.data(), g_sum_zeta.data(), 1);

    const auto& xi0 = areamn_xi0_ref_.get(state, "areamn_xi0").get_device_data();
    const auto& eta0 = areamn_eta0_ref_.get(state, "areamn_eta0").get_device_data();
    const auto& z0_top = areamn_zeta0_top_ref_.get(state, "areamn_zeta0_top").get_device_data();
    const auto& z_coords = params_.z_up.get_device_data();

    VVM::Real uvtau = uvtau_;
    VVM::Real nlim = nudgelim_;
    VVM::Real inv_pts = inv_total_xy_pts_;

    Kokkos::parallel_for("AREAMN_Apply",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {

            if (z_coords(k) >= nlim) {
                VVM::Real sum_xi = g_sum_xi(k) * inv_pts;
                VVM::Real sum_eta = g_sum_eta(k) * inv_pts;
                
                VVM::Real sumxn = (uvtau == 0.0) ? xi0(k) : (1.0 - dt/uvtau)*sum_xi + xi0(k)*(dt/uvtau);
                VVM::Real sumyn = (uvtau == 0.0) ? eta0(k) : (1.0 - dt/uvtau)*sum_eta + eta0(k)*(dt/uvtau);

                if (itypev(k,j,i) == 1) {
                    xi(k,j,i) = xi(k,j,i) - sum_xi + sumxn;
                }
                if (itypeu(k,j,i) == 1) {
                    eta(k,j,i) = eta(k,j,i) - sum_eta + sumyn;
                }
            }

            if (k == top_k) {
                VVM::Real sum_zeta = g_sum_zeta() * inv_pts;
                zeta(top_k,j,i) = zeta(top_k,j,i) - sum_zeta + z0_top();
            }
        }
    );
}

void AreaMeanNudging::apply_uvtopmn(Core::State& state, VVM::Real dt) {
    if (!enable_) return;

    auto& utopmn = utopmn_ref_.get(state, "utopmn").get_mutable_device_data();
    auto& vtopmn = vtopmn_ref_.get(state, "vtopmn").get_mutable_device_data();

    const auto& utopmn0 = areamn_utopmn0_ref_.get(state, "areamn_utopmn0").get_device_data();
    const auto& vtopmn0 = areamn_vtopmn0_ref_.get(state, "areamn_vtopmn0").get_device_data();

    VVM::Real uvtau = uvtau_;

    Kokkos::parallel_for("AREAMN_Apply_UVTOPMN", 1, KOKKOS_LAMBDA(const int i) {
        if (uvtau == 0.0) {
            utopmn() = utopmn0();
            vtopmn() = vtopmn0();
        } 
        else {
            utopmn() = (1.0 - dt / uvtau) * utopmn() + utopmn0() * (dt / uvtau);
            vtopmn() = (1.0 - dt / uvtau) * vtopmn() + vtopmn0() * (dt / uvtau);
        }
    });
}

} // namespace Dynamics
} // namespace VVM
