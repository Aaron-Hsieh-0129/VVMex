#include "RandomForcing.hpp"
#include "DeterministicPerturbation.hpp"

namespace VVM {
namespace Dynamics {

RandomForcing::RandomForcing(const Utils::ConfigurationManager& config, 
                             const Core::Grid& grid,
                             const Core::Parameters& params)
    : config_(config), grid_(grid), params_(params) {
    
    enabled_ = config.get_value<bool>("dynamics.forcings.random_perturbation.enable", false);
    end_time_ = config.get_value<VVM::Real>("dynamics.forcings.random_perturbation.time_s", 50.0);
    amplitude_ = config.get_value<VVM::Real>("dynamics.forcings.random_perturbation.amplitude", 1.0);
    seed_ = config.get_value<int>("dynamics.forcings.random_perturbation.random_seed", 12345); 
}

void RandomForcing::initialize(Core::State& state) {
    VVM::Real z_start_m = config_.get_value<VVM::Real>("dynamics.forcings.random_perturbation.z_start_m", 0); 
    VVM::Real z_end_m = config_.get_value<VVM::Real>("dynamics.forcings.random_perturbation.z_end_m", 0); 
    auto z_mid_host = params_.z_mid.get_host_data();
    int nz = grid_.get_local_total_points_z();
    int h = grid_.get_halo_cells();
    
    for (int k = h; k < nz; ++k) {
        VVM::Real z = z_mid_host(k);
        if (z >= z_start_m) {
            k_start_ = k;
            z_start_m = z;
            break;
        }
    }
    for (int k = nz-h; k > 0; k--) {
        VVM::Real z = z_mid_host(k);
        if (z <= z_end_m) {
            k_end_ = k;
            z_end_m = z;
            break;
        }
    }

    if (grid_.get_mpi_rank() == 0 && enabled_) {
        if (k_start_ > k_end_) {
            std::cout << "[RandomForcing] WARNING: No vertical levels found between " 
                      << z_start_m << "m and " << z_end_m << "m." << std::endl;
        } 
        else {
            std::cout << "[RandomForcing] Initialized. Range: " << z_start_m << "m to " << z_end_m << "m "
                      << "(Indices k: " << k_start_ << " to " << k_end_ << ")" << std::endl;
        }
    }

}

void RandomForcing::apply(Core::State& state) {
    if (!enabled_) return;
    if (state.get_time() >= end_time_) return;
    if (k_start_ > k_end_) return;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();

    auto& th = th_ref_.get(state, "th").get_mutable_device_data();
    VVM::Real amp = amplitude_;
    int k_start = k_start_;
    int k_end = k_end_ + 1;
    int seed = seed_;
    std::uint64_t step = static_cast<std::uint64_t>(state.get_step());
    int global_z_start = grid_.get_local_physical_start_z();
    int global_y_start = grid_.get_local_physical_start_y();
    int global_x_start = grid_.get_local_physical_start_x();

    if (k_end > grid_.get_local_total_points_z()) k_end = grid_.get_local_total_points_z();

    Kokkos::parallel_for("RandomForcing_Apply",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const std::uint64_t global_k = static_cast<std::uint64_t>(global_z_start + k - h);
            const std::uint64_t global_j = static_cast<std::uint64_t>(global_y_start + j - h);
            const std::uint64_t global_i = static_cast<std::uint64_t>(global_x_start + i - h);
            const VVM::Real noise = RandomForcingDetail::signed_unit_random(
                seed, step, global_k, global_j, global_i) * amp;
            th(k, j, i) += noise;
        }
    );
    return;
}

} // namespace Dynamics
} // namespace VVM
