#include "core/Initializer.hpp"
#include "core/RegularLatLonModelConfiguration.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include <cmath>
#include <vector>

namespace VVM::Core {
namespace {
Real
jet(Real latitude, int experiment, Real amplitude) {
    const Real pi = std::acos(real(-1.0));
    const Real south = experiment == 1 ? -pi / real(8.0) : -pi / real(16.0);
    const Real north = pi / real(8.0);
    if (latitude <= south || latitude >= north) {
        return real(0.0);
    }
    return amplitude * real(80.0) *
           std::exp(real(1.0) / ((latitude - south) * (latitude - north)) +
                    real(4.0) / ((north - south) * (north - south)));
}
}

void
Initializer::initialize_jung2019() const {
    validate_jung2019_rll(config_, GridSpecification::from_config(config_));
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int experiment = config_.get_value<int>("initial_conditions.jung2019.case");
    const Real jet_scale =
        config_.get_value<Real>("initial_conditions.jung2019.jet_scale", real(1.0));
    const Real perturbation_scale =
        config_.get_value<Real>("initial_conditions.jung2019.perturbation_scale", real(1.0));
    const Real pi = std::acos(real(-1.0));
    const auto& geometry = grid_.horizontal_specification().geometry;
    const Real radius = geometry.regular_lat_lon.radius;
    const Real dphi = geometry.dq2;
    const Real dlambda = geometry.dq1;
    const Real south = geometry.regular_lat_lon.latitude_south_edge;
    const Real west = geometry.regular_lat_lon.longitude_west_edge;
    const bool zonal_mountain =
        is_rll_mountain(config_) &&
        config_.get_value<bool>("initial_conditions.rll_mountain.zonal_flow", false);
    const Real mountain_u0 =
        config_.get_value<Real>("initial_conditions.rll_mountain.u0_m_s", real(20.0));
    if (zonal_mountain && !std::isfinite(mountain_u0)) {
        throw std::runtime_error("RLL mountain u0_m_s must be finite.");
    }
    Real jet_shift = real(0.);
    const bool rll_mountain = is_rll_mountain(config_);
    if (!zonal_mountain && rll_mountain &&
        config_.has_key("initial_conditions.rll_mountain.jet_center_latitude_deg")) {
        const Real original_center = experiment == 1 ? real(0.) : pi / real(32.);
        const Real center =
            config_.get_value<Real>("initial_conditions.rll_mountain.jet_center_latitude_deg",
                original_center * real(180.) / pi) *
            pi / real(180.);
        jet_shift = center - original_center;
        const Real jet_south = (experiment == 1 ? -pi / real(8.) : -pi / real(16.)) + jet_shift;
        const Real jet_north = pi / real(8.) + jet_shift;
        if (!std::isfinite(center) || jet_south < south ||
            jet_north > south + grid_.get_global_points_y() * dphi) {
            throw std::runtime_error(
                "RLL mountain jet must have a finite center and fit inside the latitude walls.");
        }
    }
    // Shift only the wind profile. Spherical metric factors retain physical
    // latitude; derive both its discrete curl and channel streamfunction anew.
    const auto prescribed_jet = [&](Real latitude) {
        // Requested solid-body zonal flow: alpha=0, compatible with channel walls.
        // The same profile supplies wind, discrete curl and wall circulation.
        if (zonal_mountain) {
            return mountain_u0 * std::cos(latitude);
        }
        return jet(jet_shift == real(0.) ? latitude : latitude - jet_shift, experiment, jet_scale);
    };
    std::vector<Real> psi_prefix(grid_.get_global_points_y() + 1, real(0.0));
    for (int j = 0; j < grid_.get_global_points_y(); ++j) {
        psi_prefix[j + 1] =
            psi_prefix[j] - radius * dphi * prescribed_jet(south + (real(j) + real(.5)) * dphi);
    }
    state_.add_field<0>("rll_psi_north",
        {},
        {GridStaggering::StaggeredXY, "m2 s-1", "prescribed northern streamfunction wall value"});
    Kokkos::deep_copy(state_.get_field<0>("rll_psi_north").get_mutable_device_data(),
        psi_prefix.back());
    auto psi = state_.get_field<2>("psi").get_host_data();

    for (const char* name : {"ITYPEU", "ITYPEV", "ITYPEW"}) {
        Kokkos::deep_copy(state_.get_field<3>(name).get_mutable_device_data(), real(1.0));
    }
    parameters_.max_topo_idx = h;
    if (!rll_mountain) {
        initialize_topo();
    }
    initialize_zeta_factor_for_twisting();

    state_.add_field<2>("rll_background_u",
        {ny, nx},
        {GridStaggering::StaggeredX, "m s-1", "prescribed Jung background eastward wind"});
    state_.add_field<2>("rll_background_zeta",
        {ny, nx},
        {GridStaggering::StaggeredXY, "s-1", "discrete curl of prescribed Jung background wind"});
    state_.add_field<2>("rll_zeta_top",
        {ny, nx},
        {GridStaggering::StaggeredXY, "s-1", "relative vertical vorticity at native top Z points"});
    auto bg_u = state_.get_field<2>("rll_background_u").get_host_data();
    auto bg_z = state_.get_field<2>("rll_background_zeta").get_host_data();
    auto u = state_.get_field<3>("u").get_host_data();
    auto zeta = state_.get_field<3>("zeta").get_host_data();
    auto lon = state_.get_field<2>("lon").get_host_data();
    auto lat = state_.get_field<2>("lat").get_host_data();
    for (int j = 0; j < ny; ++j) {
        const int gj = grid_.get_local_physical_start_y() + j - h;
        const Real phi_u = south + (real(gj) + real(0.5)) * dphi;
        const Real phi_z = south + (real(gj) + real(1.0)) * dphi;
        const Real initial_u = prescribed_jet(phi_u);
        const Real curl =
            -(std::cos(phi_u + dphi) * prescribed_jet(phi_u + dphi) - std::cos(phi_u) * initial_u) /
            (radius * std::cos(phi_z) * dphi);
        for (int i = 0; i < nx; ++i) {
            const int gi = grid_.get_local_physical_start_x() + i - h;
            const Real lambda =
                std::remainder(west + (real(gi) + real(1.0)) * dlambda, real(2.0) * pi);
            const Real center = experiment == 1 ? pi / real(24.0) : pi / real(16.0);
            const Real bump = perturbation_scale * real(1.e-6) * std::cos(phi_z) *
                              std::exp(-real(9.0) * lambda * lambda -
                                       real(900.0) * (center - phi_z) * (center - phi_z));
            bg_u(j, i) = initial_u;
            bg_z(j, i) = curl;
            psi(j, i) = psi_prefix[std::max(0, std::min(gj + 1, grid_.get_global_points_y()))];
            lon(j, i) = (west + (real(gi) + real(0.5)) * dlambda) * real(180.0) / pi;
            lat(j, i) = phi_u * real(180.0) / pi;
            for (int k = 0; k < nz; ++k) {
                u(k, j, i) = initial_u;
                zeta(k, j, i) = curl + bump;
            }
        }
    }
    Kokkos::deep_copy(state_.get_field<2>("rll_background_u").get_mutable_device_data(), bg_u);
    Kokkos::deep_copy(state_.get_field<2>("psi").get_mutable_device_data(), psi);
    Kokkos::deep_copy(state_.get_field<2>("psinm1").get_mutable_device_data(), psi);
    Kokkos::deep_copy(state_.get_field<2>("rll_background_zeta").get_mutable_device_data(), bg_z);
    Kokkos::deep_copy(state_.get_field<2>("lon").get_mutable_device_data(), lon);
    Kokkos::deep_copy(state_.get_field<2>("lat").get_mutable_device_data(), lat);
    Kokkos::deep_copy(state_.get_field<3>("u").get_mutable_device_data(), u);
    Kokkos::deep_copy(state_.get_field<3>("zeta").get_mutable_device_data(), zeta);
    Boundary::HorizontalBoundaryStencils boundary(grid_);
    halo_exchanger_.exchange_halos(state_.get_field<3>("zeta"));
    boundary.fill_positive_face_q2_homogeneous_dirichlet_halos(state_.get_field<3>("zeta"));
    boundary.fill_regular_lat_lon_free_slip_physical_wind_halos(state_.get_field<3>("u"),
        state_.get_field<3>("v"));
}
} // namespace VVM::Core
