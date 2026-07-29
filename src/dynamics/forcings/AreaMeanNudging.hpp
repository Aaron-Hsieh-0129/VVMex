#pragma once

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"
#include <Kokkos_Core.hpp>
#include <string>

#if defined(ENABLE_NCCL)
#include <nccl.h>
#else
#include <mpi.h>
#endif

namespace VVM {
namespace Dynamics {

class AreaMeanNudging {
public:
    AreaMeanNudging(const Utils::ConfigurationManager& config, 
                    const Core::Grid& grid, 
                    const Core::Parameters& params);
    ~AreaMeanNudging() = default;

    void initialize(Core::State& state);
    void update_forcing_target(Core::State& state, VVM::Real current_time);

    void apply_vorticity(Core::State& state, VVM::Real dt);
    void apply_uvtopmn(Core::State& state, VVM::Real dt);

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;

    bool enable_;
    VVM::Real uvtau_;
    VVM::Real nudgelim_;
    
    VVM::Real inv_total_xy_pts_;

    bool use_netcdf_target_ = false;
    bool constant_upper_wind_ = false;
    std::string forcing_directory_;
    std::string forcing_file_prefix_;
    VVM::Real forcing_interval_s_ = 3600.0;
    VVM::Real constant_upper_wind_threshold_Pa_ = 3000.0;
    VVM::Real time_T1_ = 0.0;
    VVM::Real time_T2_ = 3600.0;

    Kokkos::View<VVM::Real*> u_T1_;
    Kokkos::View<VVM::Real*> u_T2_;
    Kokkos::View<VVM::Real*> v_T1_;
    Kokkos::View<VVM::Real*> v_T2_;

    std::string forcing_filename(VVM::Real time) const;
    void load_wind_profiles(
        const std::string& filename,
        VVM::Real expected_time,
        Kokkos::View<VVM::Real*>& u_target,
        Kokkos::View<VVM::Real*>& v_target) const;
    void check_nc_error(int status, const std::string& message) const;

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm_; 
    cudaStream_t stream_;
#endif
};

} // namespace Dynamics
} // namespace VVM
