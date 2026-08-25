#include "WENO5.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace VVM {
namespace Dynamics {

WENO5::Options WENO5::parse_options(
    const std::string& variable_name,
    const nlohmann::json& advection_config) {
    Options result;
    if (!advection_config.contains("scheme_options")) return result;

    const auto& options = advection_config.at("scheme_options");
    if (!options.is_object()) {
        throw std::runtime_error(
            "Tracer '" + variable_name +
            "': weno5 scheme_options must be an object.");
    }
    for (const auto& item : options.items()) {
        if (item.key() != "epsilon") {
            throw std::runtime_error(
                "Tracer '" + variable_name + "': unknown weno5 option '" +
                item.key() + "'; supported option: epsilon.");
        }
    }
    if (!options.contains("epsilon")) return result;
    if (!options.at("epsilon").is_number()) {
        throw std::runtime_error(
            "Tracer '" + variable_name +
            "': weno5 option 'epsilon' must be numeric.");
    }
    result.epsilon = options.at("epsilon").get<VVM::Real>();
    if (!std::isfinite(result.epsilon) ||
        result.epsilon <= VVM::real(0.0)) {
        throw std::runtime_error(
            "Tracer '" + variable_name +
            "': weno5 epsilon must be finite and greater than zero.");
    }
    return result;
}

WENO5::Options WENO5::validate_configuration(
    const std::string& variable_name,
    const std::string& spatial_scheme,
    const std::string& temporal_scheme,
    const nlohmann::json& advection_config,
    size_t enabled_tendency_count,
    int configured_halo_width) {
    if (spatial_scheme != "weno5" ||
        temporal_scheme != "SSPRK2") {
        throw std::runtime_error(
            "Tracer '" + variable_name + "' requested spatial scheme '" +
            spatial_scheme + "' and temporal scheme '" + temporal_scheme +
            "'; supported WENO5 pairing: spatial scheme 'weno5' with "
            "temporal scheme 'SSPRK2'.");
    }
    if (enabled_tendency_count != 1) {
        throw std::runtime_error(
            "Tracer '" + variable_name +
            "' uses weno5 but has additional enabled tendency terms; "
            "passive WENO5 tracers support advection only.");
    }
    if (configured_halo_width < required_halo_width) {
        std::ostringstream message;
        message << "Tracer '" << variable_name
                << "': weno5 halo width is insufficient; configured "
                << configured_halo_width << ", required "
                << required_halo_width << ".";
        throw std::runtime_error(message.str());
    }
    return parse_options(variable_name, advection_config);
}

WENO5::WENO5(
    std::string variable_name,
    const nlohmann::json& advection_config,
    const Utils::ConfigurationManager& config,
    const Core::Grid& grid,
    Core::HaloExchanger& halo_exchanger,
    const Core::BoundaryConditionManager& bc_manager)
    : variable_name_(std::move(variable_name)),
      options_(parse_options(variable_name_, advection_config)),
      grid_(grid),
      vertical_scheme_(config, grid, halo_exchanger, bc_manager) {
    if (grid.get_halo_cells() < required_halo_width) {
        std::ostringstream message;
        message << "Tracer '" << variable_name_
                << "': weno5 halo width is insufficient; configured "
                << grid.get_halo_cells() << ", required "
                << required_halo_width << ".";
        throw std::runtime_error(message.str());
    }
    if (grid.get_global_points_x() > 1 &&
        grid.get_local_physical_points_x() < required_halo_width) {
        throw std::runtime_error(
            "Tracer '" + variable_name_ +
            "': weno5 requires at least three physical x cells per MPI rank.");
    }
    if (grid.get_global_points_y() > 1 &&
        grid.get_local_physical_points_y() < required_halo_width) {
        throw std::runtime_error(
            "Tracer '" + variable_name_ +
            "': weno5 requires at least three physical y cells per MPI rank.");
    }
}

void WENO5::calculate_advection_tendency(
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
    (void)stage_dt;
    if (var_name != variable_name_) {
        throw std::runtime_error(
            "weno5 instance for tracer '" + variable_name_ +
            "' cannot advect field '" + var_name + "'.");
    }

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();
    const VVM::Real epsilon = options_.epsilon;

    const auto q = scalar.get_device_data();
    const auto mx = mass_flux_x.get_device_data();
    const auto my = mass_flux_y.get_device_data();
    const auto fluid = ITYPEW_ref_.get(state, "ITYPEW").get_device_data();
    const auto face_x = ITYPEU_ref_.get(state, "ITYPEU").get_device_data();
    const auto face_y = ITYPEV_ref_.get(state, "ITYPEV").get_device_data();
    const auto rdx = params.rdx;
    const auto rdy = params.rdy;
    auto tendency = out_tendency.get_mutable_device_data();

    // u(k,j,i) is located between centered tracer cells i and i+1.
    Kokkos::parallel_for(
        "WENO5_flux_divergence_x_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (fluid(k, j, i) != VVM::real(1.0)) return;

            auto face_flux = [&](const int f) {
                const bool open =
                    face_x(k, j, f) == VVM::real(1.0) &&
                    fluid(k, j, f) == VVM::real(1.0) &&
                    fluid(k, j, f + 1) == VVM::real(1.0);
                if (!open) return VVM::real(0.0);

                const VVM::Real mass = mx(k, j, f);
                if (mass >= VVM::real(0.0)) {
                    const bool complete =
                        fluid(k, j, f - 2) == VVM::real(1.0) &&
                        fluid(k, j, f - 1) == VVM::real(1.0) &&
                        fluid(k, j, f + 2) == VVM::real(1.0);
                    if (!complete) return mass * q(k, j, f);

                    const VVM::Real q_im2 = q(k, j, f - 2);
                    const VVM::Real q_im1 = q(k, j, f - 1);
                    const VVM::Real q_i = q(k, j, f);
                    const VVM::Real q_ip1 = q(k, j, f + 1);
                    const VVM::Real q_ip2 = q(k, j, f + 2);
                    return mass * WENO5::reconstruct_left(
                        q_im2, q_im1, q_i, q_ip1, q_ip2, epsilon);
                }

                const bool complete =
                    fluid(k, j, f - 1) == VVM::real(1.0) &&
                    fluid(k, j, f + 2) == VVM::real(1.0) &&
                    fluid(k, j, f + 3) == VVM::real(1.0);
                if (!complete) return mass * q(k, j, f + 1);

                const VVM::Real q_im1 = q(k, j, f - 1);
                const VVM::Real q_i = q(k, j, f);
                const VVM::Real q_ip1 = q(k, j, f + 1);
                const VVM::Real q_ip2 = q(k, j, f + 2);
                const VVM::Real q_ip3 = q(k, j, f + 3);
                return mass * WENO5::reconstruct_right(
                    q_im1, q_i, q_ip1, q_ip2, q_ip3, epsilon);
            };

            const VVM::Real flux_right = face_flux(i);
            const VVM::Real flux_left = face_flux(i - 1);
            tendency(k, j, i) +=
                -(flux_right - flux_left) * rdx();
        });

    // v(k,j,i) is located between centered tracer cells j and j+1.
    Kokkos::parallel_for(
        "WENO5_flux_divergence_y_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (fluid(k, j, i) != VVM::real(1.0)) return;

            auto face_flux = [&](const int f) {
                const bool open =
                    face_y(k, f, i) == VVM::real(1.0) &&
                    fluid(k, f, i) == VVM::real(1.0) &&
                    fluid(k, f + 1, i) == VVM::real(1.0);
                if (!open) return VVM::real(0.0);

                const VVM::Real mass = my(k, f, i);
                if (mass >= VVM::real(0.0)) {
                    const bool complete =
                        fluid(k, f - 2, i) == VVM::real(1.0) &&
                        fluid(k, f - 1, i) == VVM::real(1.0) &&
                        fluid(k, f + 2, i) == VVM::real(1.0);
                    if (!complete) return mass * q(k, f, i);

                    const VVM::Real q_jm2 = q(k, f - 2, i);
                    const VVM::Real q_jm1 = q(k, f - 1, i);
                    const VVM::Real q_j = q(k, f, i);
                    const VVM::Real q_jp1 = q(k, f + 1, i);
                    const VVM::Real q_jp2 = q(k, f + 2, i);
                    return mass * WENO5::reconstruct_left(
                        q_jm2, q_jm1, q_j, q_jp1, q_jp2, epsilon);
                }

                const bool complete =
                    fluid(k, f - 1, i) == VVM::real(1.0) &&
                    fluid(k, f + 2, i) == VVM::real(1.0) &&
                    fluid(k, f + 3, i) == VVM::real(1.0);
                if (!complete) return mass * q(k, f + 1, i);

                const VVM::Real q_jm1 = q(k, f - 1, i);
                const VVM::Real q_j = q(k, f, i);
                const VVM::Real q_jp1 = q(k, f + 1, i);
                const VVM::Real q_jp2 = q(k, f + 2, i);
                const VVM::Real q_jp3 = q(k, f + 3, i);
                return mass * WENO5::reconstruct_right(
                    q_jm1, q_j, q_jp1, q_jp2, q_jp3, epsilon);
            };

            const VVM::Real flux_top = face_flux(j);
            const VVM::Real flux_bottom = face_flux(j - 1);
            tendency(k, j, i) +=
                -(flux_top - flux_bottom) * rdy();
        });

    // Deliberately preserve the established stretched-grid vertical scheme,
    // including its density, metric, staggering, and boundary conventions.
    vertical_scheme_.calculate_flux_convergence_z(
        scalar, mass_flux_z, grid, params, out_tendency, var_name);
}

} // namespace Dynamics
} // namespace VVM
