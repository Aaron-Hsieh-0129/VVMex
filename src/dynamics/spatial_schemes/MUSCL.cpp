#include "MUSCL.hpp"

#include "core/vvm_types.hpp"

#include <cmath>
#include <mpi.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace VVM {
namespace Dynamics {

MUSCL::Options MUSCL::parse_options(
    const std::string& variable_name,
    const nlohmann::json& advection_config) {
    Options result;
    nlohmann::json options = nlohmann::json::object();
    if (advection_config.contains("scheme_options")) {
        options = advection_config.at("scheme_options");
        if (!options.is_object()) {
            throw std::runtime_error(
                "Advected variable '" + variable_name +
                "': MUSCL scheme_options must be an object.");
        }
    }

    if (options.contains("limiter") &&
        !options.at("limiter").is_string()) {
        throw std::runtime_error(
            "Advected variable '" + variable_name +
            "': MUSCL option 'limiter' must be a string.");
    }
    const std::string limiter = options.value("limiter", "vanLeer");
    if (limiter != "vanLeer") {
        throw std::runtime_error(
            "Advected variable '" + variable_name + "': unknown MUSCL limiter '" +
            limiter + "'; supported limiter: vanLeer.");
    }

    auto read_finite_number = [&](const char* key, VVM::Real default_value) {
        if (!options.contains(key)) return default_value;
        if (!options.at(key).is_number()) {
            throw std::runtime_error(
                "Advected variable '" + variable_name + "': MUSCL option '" +
                std::string(key) + "' must be numeric.");
        }
        const VVM::Real value = options.at(key).get<VVM::Real>();
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Advected variable '" + variable_name + "': MUSCL option '" +
                std::string(key) + "' must be finite.");
        }
        return value;
    };

    result.lower_bound =
        read_finite_number("lower_bound", VVM::real(0.0));
    result.max_cfl = read_finite_number("max_cfl", VVM::real(0.9));

    if (result.lower_bound < VVM::real(0.0)) {
        throw std::runtime_error(
            "Advected variable '" + variable_name +
            "': MUSCL lower_bound must be nonnegative.");
    }
    if (result.max_cfl <= VVM::real(0.0)) {
        throw std::runtime_error(
            "Advected variable '" + variable_name +
            "': MUSCL max_cfl must be greater than zero.");
    }
    return result;
}

MUSCL::Options MUSCL::validate_configuration(
    const std::string& variable_name,
    const std::string& spatial_scheme,
    const std::string& temporal_scheme,
    const nlohmann::json& advection_config,
    size_t enabled_tendency_count,
    int configured_halo_width) {
    if (spatial_scheme != "MUSCL" ||
        temporal_scheme != "SSPRK2") {
        throw std::runtime_error(
            "Advected variable '" + variable_name + "' requested spatial scheme '" +
            spatial_scheme + "' and temporal scheme '" +
            temporal_scheme + "'; supported pairing: spatial scheme "
            "'MUSCL' with temporal scheme 'SSPRK2'.");
    }
    if (enabled_tendency_count != 1) {
        throw std::runtime_error(
            "Advected variable '" + variable_name +
            "' uses MUSCL but has additional enabled tendency terms; "
            "MUSCL supports advection only.");
    }
    if (configured_halo_width < required_halo_width) {
        std::ostringstream message;
        message << "Advected variable '" << variable_name
                << "': MUSCL scalar halo width is insufficient; configured "
                << configured_halo_width << ", required "
                << required_halo_width << ".";
        throw std::runtime_error(message.str());
    }
    return parse_options(variable_name, advection_config);
}

MUSCL::MUSCL(
    std::string variable_name,
    const nlohmann::json& advection_config,
    const Core::Grid& grid)
    : variable_name_(std::move(variable_name)),
      options_(parse_options(variable_name_, advection_config)),
      grid_(grid),
      slope_x_("muscl_slope_x_" + variable_name_,
               {grid.get_local_total_points_z(),
                grid.get_local_total_points_y(),
                grid.get_local_total_points_x()}),
      slope_y_("muscl_slope_y_" + variable_name_,
               {grid.get_local_total_points_z(),
                grid.get_local_total_points_y(),
                grid.get_local_total_points_x()}),
      slope_z_("muscl_slope_z_" + variable_name_,
               {grid.get_local_total_points_z(),
                grid.get_local_total_points_y(),
                grid.get_local_total_points_x()}),
      donor_alpha_("muscl_donor_alpha_" + variable_name_,
                   {grid.get_local_total_points_z(),
                    grid.get_local_total_points_y(),
                    grid.get_local_total_points_x()}),
      flux_x_("muscl_flux_x_" + variable_name_,
              {grid.get_local_total_points_z(),
               grid.get_local_total_points_y(),
               grid.get_local_total_points_x()}),
      flux_y_("muscl_flux_y_" + variable_name_,
              {grid.get_local_total_points_z(),
               grid.get_local_total_points_y(),
               grid.get_local_total_points_x()}),
      flux_z_("muscl_flux_z_" + variable_name_,
              {grid.get_local_total_points_z(),
               grid.get_local_total_points_y(),
               grid.get_local_total_points_x()}),
      reduction_local_("muscl_reduction_local_" + variable_name_, {}),
      reduction_global_("muscl_reduction_global_" + variable_name_, {}) {
    if (grid.get_halo_cells() < required_halo_width) {
        std::ostringstream message;
        message << "Advected variable '" << variable_name_
                << "': MUSCL scalar halo width is insufficient; configured "
                << grid.get_halo_cells() << ", required "
                << required_halo_width << ".";
        throw std::runtime_error(message.str());
    }
}

void MUSCL::validate_initial_field(
    const Core::State& state,
    const Core::Field<3>& scalar) const {
    if (initial_field_validated_) return;

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const auto q = scalar.get_device_data();
    const auto fluid =
        state.get_field<3>("ITYPEW").get_device_data();

#if defined(ENABLE_NCCL)
    using DeviceMemorySpace =
        Core::Field<0>::ViewType::memory_space;
    auto local_reduction =
        reduction_local_.get_mutable_device_data();
    auto global_reduction =
        reduction_global_.get_mutable_device_data();

    Kokkos::parallel_reduce(
        "MUSCL_initial_min_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(
            const int k, const int j, const int i,
            VVM::Real& update) {
            if (fluid(k, j, i) == VVM::real(1.0) &&
                q(k, j, i) < update) {
                update = q(k, j, i);
            }
        },
        Kokkos::Min<VVM::Real, DeviceMemorySpace>(local_reduction));
    Kokkos::fence();

    const ncclResult_t min_result = ncclAllReduce(
        local_reduction.data(), global_reduction.data(), 1,
        VVM_NCCL_REAL, ncclMin, state.get_nccl_comm(),
        state.get_cuda_stream());
    if (min_result != ncclSuccess) {
        throw std::runtime_error(
            "MUSCL NCCL minimum reduction failed for '" +
            variable_name_ + "': " + ncclGetErrorString(min_result));
    }
    const cudaError_t min_sync =
        cudaStreamSynchronize(state.get_cuda_stream());
    if (min_sync != cudaSuccess) {
        throw std::runtime_error(
            "MUSCL NCCL minimum synchronization failed for '" +
            variable_name_ + "': " + cudaGetErrorString(min_sync));
    }

    VVM::Real global_min = VVM::real(0.0);
    Kokkos::deep_copy(global_min, global_reduction);
#else
    VVM::Real local_min = std::numeric_limits<VVM::Real>::max();
    Kokkos::parallel_reduce(
        "MUSCL_initial_min_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(
            const int k, const int j, const int i,
            VVM::Real& update) {
            if (fluid(k, j, i) == VVM::real(1.0) &&
                q(k, j, i) < update) {
                update = q(k, j, i);
            }
        },
        Kokkos::Min<VVM::Real>(local_min));

    VVM::Real global_min = local_min;
    MPI_Allreduce(
        &local_min, &global_min, 1, VVM_MPI_REAL, MPI_MIN,
        grid_.get_comm());
#endif

    const VVM::Real largest =
        std::numeric_limits<VVM::Real>::max();
#if defined(ENABLE_NCCL)
    Kokkos::parallel_reduce(
        "MUSCL_initial_finite_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(
            const int k, const int j, const int i,
            VVM::Real& update) {
            const VVM::Real flag =
                fluid(k, j, i) == VVM::real(1.0) &&
                (q(k, j, i) != q(k, j, i) ||
                 Kokkos::abs(q(k, j, i)) > largest)
                    ? VVM::real(1.0) : VVM::real(0.0);
            if (flag > update) update = flag;
        },
        Kokkos::Max<VVM::Real, DeviceMemorySpace>(
            local_reduction));
    Kokkos::fence();

    const ncclResult_t finite_result = ncclAllReduce(
        local_reduction.data(), global_reduction.data(), 1,
        VVM_NCCL_REAL, ncclMax, state.get_nccl_comm(),
        state.get_cuda_stream());
    if (finite_result != ncclSuccess) {
        throw std::runtime_error(
            "MUSCL NCCL finite-value reduction failed for '" +
            variable_name_ + "': " +
            ncclGetErrorString(finite_result));
    }
    const cudaError_t finite_sync =
        cudaStreamSynchronize(state.get_cuda_stream());
    if (finite_sync != cudaSuccess) {
        throw std::runtime_error(
            "MUSCL NCCL finite-value synchronization failed for '" +
            variable_name_ + "': " +
            cudaGetErrorString(finite_sync));
    }

    VVM::Real global_nonfinite = VVM::real(0.0);
    Kokkos::deep_copy(global_nonfinite, global_reduction);
#else
    int local_nonfinite = 0;
    Kokkos::parallel_reduce(
        "MUSCL_initial_finite_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(
            const int k, const int j, const int i,
            int& update) {
            const int flag =
                fluid(k, j, i) == VVM::real(1.0) &&
                (q(k, j, i) != q(k, j, i) ||
                 Kokkos::abs(q(k, j, i)) > largest)
                    ? 1 : 0;
            if (flag > update) update = flag;
        },
        Kokkos::Max<int>(local_nonfinite));
    int global_nonfinite = local_nonfinite;
    MPI_Allreduce(
        &local_nonfinite, &global_nonfinite, 1, MPI_INT, MPI_MAX,
        grid_.get_comm());
#endif
    if (global_nonfinite != 0) {
        throw std::runtime_error(
            "Advected variable '" + variable_name_ +
            "' contains a non-finite value in an active fluid cell; the "
            "field in the initial or restart NetCDF file must contain "
            "finite values satisfying the configured lower bound.");
    }

    if (violates_lower_bound(global_min, options_.lower_bound)) {
        std::ostringstream message;
        message << "Advected variable '" << variable_name_
                << "' has initial/restart minimum " << global_min
                << " below configured MUSCL lower_bound "
                << options_.lower_bound
                << ". The field in the initial or restart NetCDF "
                   "file must satisfy the bound.";
        throw std::runtime_error(message.str());
    }
    initial_field_validated_ = true;
}

void MUSCL::validate_cfl(
    const Core::State& state,
    const Core::Field<3>& mass_flux_x,
    const Core::Field<3>& mass_flux_y,
    const Core::Field<3>& mass_flux_z,
    const Core::Parameters& params,
    VVM::Real stage_dt) const {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const auto mx = mass_flux_x.get_device_data();
    const auto my = mass_flux_y.get_device_data();
    const auto mz = mass_flux_z.get_device_data();
    const auto rho = state.get_field<1>("rhobar").get_device_data();
    const auto fluid = state.get_field<3>("ITYPEW").get_device_data();
    const auto face_x = state.get_field<3>("ITYPEU").get_device_data();
    const auto face_y = state.get_field<3>("ITYPEV").get_device_data();
    const auto flex = params.flex_height_coef_mid.get_device_data();
    const auto rdx = params.rdx;
    const auto rdy = params.rdy;
    const auto rdz = params.rdz;

#if defined(ENABLE_NCCL)
    using DeviceMemorySpace =
        Core::Field<0>::ViewType::memory_space;
    auto local_reduction =
        reduction_local_.get_mutable_device_data();
    auto global_reduction =
        reduction_global_.get_mutable_device_data();

    Kokkos::parallel_reduce("MUSCL_outgoing_cfl_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(
            const int k, const int j, const int i,
            VVM::Real& update) {
            if (update < VVM::real(0.0)) {
                update = VVM::real(0.0);
            }
            if (fluid(k, j, i) != VVM::real(1.0)) return;

            VVM::Real outgoing = VVM::real(0.0);
            if (face_x(k, j, i) == VVM::real(1.0) &&
                fluid(k, j, i + 1) == VVM::real(1.0) &&
                mx(k, j, i) > VVM::real(0.0)) {
                outgoing += mx(k, j, i) * rdx();
            }
            if (face_x(k, j, i - 1) == VVM::real(1.0) &&
                fluid(k, j, i - 1) == VVM::real(1.0) &&
                mx(k, j, i - 1) < VVM::real(0.0)) {
                outgoing -= mx(k, j, i - 1) * rdx();
            }
            if (face_y(k, j, i) == VVM::real(1.0) &&
                fluid(k, j + 1, i) == VVM::real(1.0) &&
                my(k, j, i) > VVM::real(0.0)) {
                outgoing += my(k, j, i) * rdy();
            }
            if (face_y(k, j - 1, i) == VVM::real(1.0) &&
                fluid(k, j - 1, i) == VVM::real(1.0) &&
                my(k, j - 1, i) < VVM::real(0.0)) {
                outgoing -= my(k, j - 1, i) * rdy();
            }
            if (k < nz - h - 1 &&
                fluid(k + 1, j, i) == VVM::real(1.0) &&
                mz(k, j, i) > VVM::real(0.0)) {
                outgoing += mz(k, j, i) * rdz() * flex(k);
            }
            if (k > h &&
                fluid(k - 1, j, i) == VVM::real(1.0) &&
                mz(k - 1, j, i) < VVM::real(0.0)) {
                outgoing -= mz(k - 1, j, i) * rdz() * flex(k);
            }
            const VVM::Real cfl = stage_dt * outgoing / rho(k);
            if (cfl > update) update = cfl;
        },
        Kokkos::Max<VVM::Real, DeviceMemorySpace>(local_reduction));
    Kokkos::fence();

    const ncclResult_t cfl_result = ncclAllReduce(
        local_reduction.data(), global_reduction.data(), 1,
        VVM_NCCL_REAL, ncclMax, state.get_nccl_comm(),
        state.get_cuda_stream());
    if (cfl_result != ncclSuccess) {
        throw std::runtime_error(
            "MUSCL NCCL CFL reduction failed for '" +
            variable_name_ + "': " + ncclGetErrorString(cfl_result));
    }
    const cudaError_t cfl_sync = cudaStreamSynchronize(state.get_cuda_stream());
    if (cfl_sync != cudaSuccess) {
        throw std::runtime_error(
            "MUSCL NCCL CFL synchronization failed for '" +
            variable_name_ + "': " + cudaGetErrorString(cfl_sync));
    }

    VVM::Real global_max = VVM::real(0.0);
    Kokkos::deep_copy(global_max, global_reduction);
#else
    VVM::Real local_max = VVM::real(0.0);
    Kokkos::parallel_reduce("MUSCL_outgoing_cfl_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(
            const int k, const int j, const int i,
            VVM::Real& update) {
            if (update < VVM::real(0.0)) {
                update = VVM::real(0.0);
            }
            if (fluid(k, j, i) != VVM::real(1.0)) return;

            VVM::Real outgoing = VVM::real(0.0);
            if (face_x(k, j, i) == VVM::real(1.0) &&
                fluid(k, j, i + 1) == VVM::real(1.0) &&
                mx(k, j, i) > VVM::real(0.0)) {
                outgoing += mx(k, j, i) * rdx();
            }
            if (face_x(k, j, i - 1) == VVM::real(1.0) &&
                fluid(k, j, i - 1) == VVM::real(1.0) &&
                mx(k, j, i - 1) < VVM::real(0.0)) {
                outgoing -= mx(k, j, i - 1) * rdx();
            }
            if (face_y(k, j, i) == VVM::real(1.0) &&
                fluid(k, j + 1, i) == VVM::real(1.0) &&
                my(k, j, i) > VVM::real(0.0)) {
                outgoing += my(k, j, i) * rdy();
            }
            if (face_y(k, j - 1, i) == VVM::real(1.0) &&
                fluid(k, j - 1, i) == VVM::real(1.0) &&
                my(k, j - 1, i) < VVM::real(0.0)) {
                outgoing -= my(k, j - 1, i) * rdy();
            }
            if (k < nz - h - 1 &&
                fluid(k + 1, j, i) == VVM::real(1.0) &&
                mz(k, j, i) > VVM::real(0.0)) {
                outgoing += mz(k, j, i) * rdz() * flex(k);
            }
            if (k > h &&
                fluid(k - 1, j, i) == VVM::real(1.0) &&
                mz(k - 1, j, i) < VVM::real(0.0)) {
                outgoing -= mz(k - 1, j, i) * rdz() * flex(k);
            }
            const VVM::Real cfl = stage_dt * outgoing / rho(k);
            if (cfl > update) update = cfl;
        },
        Kokkos::Max<VVM::Real>(local_max));

    VVM::Real global_max = local_max;
    MPI_Allreduce(
        &local_max, &global_max, 1, VVM_MPI_REAL, MPI_MAX,
        grid_.get_comm());
#endif

    if (global_max > options_.max_cfl) {
        std::ostringstream message;
        message << "Advected variable '" << variable_name_
                << "' MUSCL outgoing CFL " << global_max
                << " exceeds configured max_cfl " << options_.max_cfl
                << " for model timestep " << stage_dt << ".";
        throw std::runtime_error(message.str());
    }
}

void MUSCL::calculate_advection_tendency(
    const Core::State& state,
    const Core::Field<3>& scalar,
    const Core::Field<3>& mass_flux_x,
    const Core::Field<3>& mass_flux_y,
    const Core::Field<3>& mass_flux_z,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency,
    const std::string& var_name,
    VVM::Real stage_dt) const {
    if (var_name != variable_name_) {
        throw std::runtime_error(
            "MUSCL instance for advected variable '" + variable_name_ +
            "' cannot advect field '" + var_name + "'.");
    }
    if (!(stage_dt > VVM::real(0.0)) || !std::isfinite(stage_dt)) {
        throw std::runtime_error(
            "Advected variable '" + variable_name_ +
            "': MUSCL requires a finite positive Forward-Euler stage timestep.");
    }

    // validate_initial_field(state, scalar);
    // validate_cfl(state, mass_flux_x, mass_flux_y, mass_flux_z, params, stage_dt);

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const VVM::Real lower_bound = options_.lower_bound;
    const auto q = scalar.get_device_data();
    const auto mx = mass_flux_x.get_device_data();
    const auto my = mass_flux_y.get_device_data();
    const auto mz = mass_flux_z.get_device_data();
    const auto rho = state.get_field<1>("rhobar").get_device_data();
    const auto fluid = state.get_field<3>("ITYPEW").get_device_data();
    const auto face_x = state.get_field<3>("ITYPEU").get_device_data();
    const auto face_y = state.get_field<3>("ITYPEV").get_device_data();
    const auto flex = params.flex_height_coef_mid.get_device_data();
    const auto z_mid = params.z_mid.get_device_data();
    const auto z_up = params.z_up.get_device_data();
    const auto rdx = params.rdx;
    const auto rdy = params.rdy;
    const auto rdz = params.rdz;

    auto sx = slope_x_.get_mutable_device_data();
    auto sy = slope_y_.get_mutable_device_data();
    auto sz = slope_z_.get_mutable_device_data();
    auto alpha = donor_alpha_.get_mutable_device_data();
    auto fx = flux_x_.get_mutable_device_data();
    auto fy = flux_y_.get_mutable_device_data();
    auto fz = flux_z_.get_mutable_device_data();
    auto tendency = out_tendency.get_mutable_device_data();

    slope_x_.set_to_zero();
    slope_y_.set_to_zero();
    slope_z_.set_to_zero();
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), alpha, VVM::real(1.0));
    flux_x_.set_to_zero();
    flux_y_.set_to_zero();
    flux_z_.set_to_zero();

    // The expanded horizontal range reconstructs the first halo donor. With
    // two halos, that donor has the same full three-point stencil as its
    // owning rank, so no directional communication or one-sided fallback is
    // required at MPI boundaries.
    Kokkos::parallel_for("MUSCL_reconstruct_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h - 1, h - 1}, {nz - h, ny - h + 1, nx - h + 1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (fluid(k, j, i) != VVM::real(1.0)) return;

            VVM::Real slope_x = VVM::real(0.0);
            VVM::Real slope_y = VVM::real(0.0);
            VVM::Real slope_z = VVM::real(0.0);
            if (fluid(k, j, i - 1) == VVM::real(1.0) &&
                fluid(k, j, i + 1) == VVM::real(1.0)) {
                slope_x = van_leer_slope(
                    q(k, j, i) - q(k, j, i - 1),
                    q(k, j, i + 1) - q(k, j, i));
            }
            if (fluid(k, j - 1, i) == VVM::real(1.0) &&
                fluid(k, j + 1, i) == VVM::real(1.0)) {
                slope_y = van_leer_slope(
                    q(k, j, i) - q(k, j - 1, i),
                    q(k, j + 1, i) - q(k, j, i));
            }
            if (fluid(k - 1, j, i) == VVM::real(1.0) &&
                fluid(k + 1, j, i) == VVM::real(1.0)) {
                const VVM::Real backward_spacing =
                    z_mid(k) - z_mid(k - 1);
                const VVM::Real forward_spacing =
                    z_mid(k + 1) - z_mid(k);
                if (backward_spacing > VVM::real(0.0) &&
                    forward_spacing > VVM::real(0.0)) {
                    slope_z = van_leer_slope(
                        (q(k, j, i) - q(k - 1, j, i)) /
                            backward_spacing,
                        (q(k + 1, j, i) - q(k, j, i)) /
                            forward_spacing);
                }
            }

            const VVM::Real lower_z_deviation =
                slope_z * (z_up(k - 1) - z_mid(k));
            const VVM::Real upper_z_deviation =
                slope_z * (z_up(k) - z_mid(k));
            const VVM::Real half = VVM::real(0.5);
            const VVM::Real theta =
                reconstructed_state_factor_from_deviations(
                    q(k, j, i), lower_bound,
                    -half * slope_x, half * slope_x,
                    -half * slope_y, half * slope_y,
                    lower_z_deviation, upper_z_deviation);
            sx(k, j, i) = theta * slope_x;
            sy(k, j, i) = theta * slope_y;
            sz(k, j, i) = theta * slope_z;
        });

    // One donor-cell factor accounts for simultaneous depletion through all
    // six faces. Fluxes are expressed for the nonnegative excess q-q_min;
    // the lower-bound background is added back to each final mass flux.
    Kokkos::parallel_for(
        "MUSCL_donor_factors_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h - 1, h - 1}, {nz - h, ny - h + 1, nx - h + 1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (fluid(k, j, i) != VVM::real(1.0)) {
                alpha(k, j, i) = VVM::real(1.0);
                return;
            }

            const VVM::Real excess = q(k, j, i) - lower_bound;
            const VVM::Real half = VVM::real(0.5);
            const VVM::Real rxm = excess - half * sx(k, j, i);
            const VVM::Real rxp = excess + half * sx(k, j, i);
            const VVM::Real rym = excess - half * sy(k, j, i);
            const VVM::Real ryp = excess + half * sy(k, j, i);
            const VVM::Real rzm = excess + sz(k, j, i) * (z_up(k - 1) - z_mid(k));
            const VVM::Real rzp = excess + sz(k, j, i) * (z_up(k) - z_mid(k));

            VVM::Real outgoing_rate = VVM::real(0.0);
            if (face_x(k, j, i) == VVM::real(1.0) &&
                fluid(k, j, i + 1) == VVM::real(1.0) &&
                mx(k, j, i) > VVM::real(0.0)) {
                outgoing_rate += mx(k, j, i) * rxp * rdx();
            }
            if (face_x(k, j, i - 1) == VVM::real(1.0) &&
                fluid(k, j, i - 1) == VVM::real(1.0) &&
                mx(k, j, i - 1) < VVM::real(0.0)) {
                outgoing_rate -= mx(k, j, i - 1) * rxm * rdx();
            }
            if (face_y(k, j, i) == VVM::real(1.0) &&
                fluid(k, j + 1, i) == VVM::real(1.0) &&
                my(k, j, i) > VVM::real(0.0)) {
                outgoing_rate += my(k, j, i) * ryp * rdy();
            }
            if (face_y(k, j - 1, i) == VVM::real(1.0) &&
                fluid(k, j - 1, i) == VVM::real(1.0) &&
                my(k, j - 1, i) < VVM::real(0.0)) {
                outgoing_rate -= my(k, j - 1, i) * rym * rdy();
            }
            if (k < nz - h - 1 &&
                fluid(k + 1, j, i) == VVM::real(1.0) &&
                mz(k, j, i) > VVM::real(0.0)) {
                outgoing_rate +=
                    mz(k, j, i) * rzp * rdz() * flex(k);
            }
            if (k > h &&
                fluid(k - 1, j, i) == VVM::real(1.0) &&
                mz(k - 1, j, i) < VVM::real(0.0)) {
                outgoing_rate -=
                    mz(k - 1, j, i) * rzm * rdz() * flex(k);
            }

            alpha(k, j, i) = donor_outflow_factor(
                excess, stage_dt * outgoing_rate / rho(k));
        });

    // Each local shared face is calculated once. Both neighboring cell
    // divergences read this same stored value; a face receives only its
    // upwind donor's alpha.
    Kokkos::parallel_for("MUSCL_final_flux_x_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h - 1}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const bool open =
                face_x(k, j, i) == VVM::real(1.0) &&
                fluid(k, j, i) == VVM::real(1.0) &&
                fluid(k, j, i + 1) == VVM::real(1.0);
            if (!open) {
                fx(k, j, i) = VVM::real(0.0);
                return;
            }
            const VVM::Real mass = mx(k, j, i);
            const int donor_i = mass >= VVM::real(0.0) ? i : i + 1;
            const VVM::Real sign =
                mass >= VVM::real(0.0) ? VVM::real(0.5) : VVM::real(-0.5);
            const VVM::Real reconstructed_excess =
                q(k, j, donor_i) - lower_bound +
                sign * sx(k, j, donor_i);
            fx(k, j, i) = mass *
                (lower_bound +
                 alpha(k, j, donor_i) * reconstructed_excess);
        });

    Kokkos::parallel_for("MUSCL_final_flux_y_" + variable_name_, 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h - 1, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const bool open =
                face_y(k, j, i) == VVM::real(1.0) &&
                fluid(k, j, i) == VVM::real(1.0) &&
                fluid(k, j + 1, i) == VVM::real(1.0);
            if (!open) {
                fy(k, j, i) = VVM::real(0.0);
                return;
            }
            const VVM::Real mass = my(k, j, i);
            const int donor_j = mass >= VVM::real(0.0) ? j : j + 1;
            const VVM::Real sign =
                mass >= VVM::real(0.0) ? VVM::real(0.5) : VVM::real(-0.5);
            const VVM::Real reconstructed_excess =
                q(k, donor_j, i) - lower_bound +
                sign * sy(k, donor_j, i);
            fy(k, j, i) = mass *
                (lower_bound +
                 alpha(k, donor_j, i) * reconstructed_excess);
        });

    // Vertical faces h-1 and nz-h-1 remain zero: VVM's scalar vertical
    // boundaries are closed. Internal faces use rhobar_up*w supplied by
    // AdvectionTerm and the same staggered k indexing as Takacs.
    Kokkos::parallel_for("MUSCL_final_flux_z_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h - 1, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const bool open =
                fluid(k, j, i) == VVM::real(1.0) &&
                fluid(k + 1, j, i) == VVM::real(1.0);
            if (!open) {
                fz(k, j, i) = VVM::real(0.0);
                return;
            }
            const VVM::Real mass = mz(k, j, i);
            const int donor_k = mass >= VVM::real(0.0) ? k : k + 1;
            const VVM::Real reconstructed_excess =
                q(donor_k, j, i) - lower_bound +
                sz(donor_k, j, i) *
                    (z_up(k) - z_mid(donor_k));
            fz(k, j, i) = mass *
                (lower_bound +
                 alpha(donor_k, j, i) * reconstructed_excess);
        });

    Kokkos::parallel_for("MUSCL_flux_divergence_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (fluid(k, j, i) != VVM::real(1.0)) return;
            tendency(k, j, i) +=
                -(fx(k, j, i) - fx(k, j, i - 1)) * rdx()
                -(fy(k, j, i) - fy(k, j - 1, i)) * rdy()
                -(fz(k, j, i) - fz(k - 1, j, i)) *
                    rdz() * flex(k);
        });
}

} // namespace Dynamics
} // namespace VVM
