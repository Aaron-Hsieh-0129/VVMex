#include "core/Initializer.hpp"
#include "core/RegularLatLonModelConfiguration.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include <cmath>
#include <vector>

namespace VVM::Core {
namespace {
Real jet(Real latitude, int experiment, Real amplitude) {
    const Real pi = std::acos(real(-1.0));
    const Real south = experiment == 1 ? -pi / real(8.0) : -pi / real(16.0);
    const Real north = pi / real(8.0);
    if (latitude <= south || latitude >= north) return real(0.0);
    return amplitude * real(80.0) * std::exp(real(1.0) / ((latitude - south) * (latitude - north))
        + real(4.0) / ((north - south) * (north - south)));
}
}

void Initializer::initialize_jung2019() const {
    validate_jung2019_rll(config_, GridSpecification::from_config(config_));
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int experiment = config_.get_value<int>("initial_conditions.jung2019.case");
    const Real jet_scale = config_.get_value<Real>("initial_conditions.jung2019.jet_scale", real(1.0));
    const Real perturbation_scale = config_.get_value<Real>("initial_conditions.jung2019.perturbation_scale", real(1.0));
    const Real pi = std::acos(real(-1.0));
    const auto& geometry = grid_.horizontal_specification().geometry;
    const Real radius = geometry.regular_lat_lon.radius;
    const Real dphi = geometry.dq2;
    const Real dlambda = geometry.dq1;
    const Real south = geometry.regular_lat_lon.latitude_south_edge;
    const Real west = geometry.regular_lat_lon.longitude_west_edge;
    std::vector<Real> psi_prefix(grid_.get_global_points_y() + 1, real(0.0));
    for (int j = 0; j < grid_.get_global_points_y(); ++j)
        psi_prefix[j+1] = psi_prefix[j] - radius * dphi * jet(south + (real(j)+real(.5))*dphi, experiment, jet_scale);
    state_.add_field<0>("rll_psi_north", {}, {GridStaggering::StaggeredXY, "m2 s-1", "prescribed northern streamfunction wall value"});
    Kokkos::deep_copy(state_.get_field<0>("rll_psi_north").get_mutable_device_data(), psi_prefix.back());
    auto psi = state_.get_field<2>("psi").get_host_data();

    // Section 4.2 is independent of reference density and temperature:
    // winds are vertically homogeneous and thermodynamics is inactive.
    for (const char* name : {"rhobar", "rhobar_up", "pibar", "pibar_up"})
        Kokkos::deep_copy(state_.get_field<1>(name).get_mutable_device_data(), real(1.0));
    for (const char* name : {"thbar", "Tbar", "Tvbar"})
        Kokkos::deep_copy(state_.get_field<1>(name).get_mutable_device_data(), real(300.0));
    Kokkos::deep_copy(state_.get_field<3>("th").get_mutable_device_data(), real(300.0));
    for (const char* name : {"ITYPEU", "ITYPEV", "ITYPEW"})
        Kokkos::deep_copy(state_.get_field<3>(name).get_mutable_device_data(), real(1.0));
    parameters_.max_topo_idx = h;
    initialize_zeta_factor_for_twisting();

    state_.add_field<2>("rll_background_u", {ny, nx}, {GridStaggering::StaggeredX, "m s-1", "prescribed Jung background eastward wind"});
    state_.add_field<2>("rll_background_zeta", {ny, nx}, {GridStaggering::StaggeredXY, "s-1", "discrete curl of prescribed Jung background wind"});
    state_.add_field<2>("rll_zeta_top", {ny, nx}, {GridStaggering::StaggeredXY, "s-1", "relative vertical vorticity at native top Z points"});
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
        const Real initial_u = jet(phi_u, experiment, jet_scale);
        const Real curl = -(std::cos(phi_u + dphi) * jet(phi_u + dphi, experiment, jet_scale)
            - std::cos(phi_u) * initial_u) / (radius * std::cos(phi_z) * dphi);
        for (int i = 0; i < nx; ++i) {
            const int gi = grid_.get_local_physical_start_x() + i - h;
            const Real lambda = std::remainder(west + (real(gi) + real(1.0)) * dlambda, real(2.0) * pi);
            const Real center = experiment == 1 ? pi / real(24.0) : pi / real(16.0);
            const Real bump = perturbation_scale * real(1.e-6) * std::cos(phi_z)
                * std::exp(-real(9.0) * lambda * lambda - real(900.0) * (center - phi_z) * (center - phi_z));
            bg_u(j, i) = initial_u;
            bg_z(j, i) = curl;
            psi(j, i) = psi_prefix[std::max(0, std::min(gj+1, grid_.get_global_points_y()))];
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
    boundary.fill_regular_lat_lon_free_slip_physical_wind_halos(state_.get_field<3>("u"), state_.get_field<3>("v"));

    if (is_rll_mountain(config_)) {
        // A distinct terrain experiment reuses the jet, not the Section 4.2
        // reproduction claim. Longitude defaults to the domain midpoint.
        // Planetary vorticity is physical f at native Z latitude, not T latitude.
        const Real omega = config_.get_value<Real>("constants.OMEGA", real(0.));
        auto f = state_.get_field<2>("f_2d").get_host_data();
        for (int j = 0; j < ny; ++j) {
            const Real phi_z = south + (grid_.get_local_physical_start_y()+j-h+real(1.))*dphi;
            for (int i = 0; i < nx; ++i) f(j,i) = real(2.)*omega*std::sin(phi_z);
        }
        Kokkos::deep_copy(state_.get_field<2>("f_2d").get_mutable_device_data(), f);
        const Real peak = config_.get_value<Real>("initial_conditions.rll_mountain.height_m");
        const Real width = config_.get_value<Real>("initial_conditions.rll_mountain.half_width_m");
        const Real center_lambda = west + real(.5)*grid_.get_global_points_x()*dlambda;
        const Real center_phi = config_.get_value<Real>("initial_conditions.rll_mountain.center_latitude_deg",
            (south + real(.5)*grid_.get_global_points_y()*dphi)*real(180.)/pi)*pi/real(180.);
        state_.add_field<2>("rll_terrain_height", {ny, nx}, {GridStaggering::Centered, "m", "discretized centered spherical mountain height"});
        auto elevation = state_.get_field<2>("rll_terrain_height").get_host_data();
        auto terrain = state_.get_field<2>("topo").get_host_data();
        const auto z = parameters_.z_up.get_host_data();
        for (int j = h; j < ny-h; ++j) {
            const Real phi = south + (grid_.get_local_physical_start_y()+j-h+real(.5))*dphi;
            for (int i = h; i < nx-h; ++i) {
                const Real lambda = west + (grid_.get_local_physical_start_x()+i-h+real(.5))*dlambda;
                const Real cosine = std::sin(phi)*std::sin(center_phi)
                    + std::cos(phi)*std::cos(center_phi)*std::cos(lambda-center_lambda);
                const Real distance = radius*std::acos(std::max(real(-1.), std::min(real(1.), cosine)));
                const Real height = distance < real(3.)*width ? peak*std::exp(-distance*distance/(width*width)) : real(0.);
                int level = h-1;
                for (int k = h; k < nz-2*h-2; ++k)
                    if (std::abs(z(k)-height) < std::abs(z(level)-height)) level = k;
                terrain(j,i) = level == h-1 ? real(0.) : static_cast<Real>(level);
                elevation(j,i) = z(level);
            }
        }
        Kokkos::deep_copy(state_.get_field<2>("topo").get_mutable_device_data(), terrain);
        Kokkos::deep_copy(state_.get_field<2>("rll_terrain_height").get_mutable_device_data(), elevation);
        halo_exchanger_.exchange_halos(state_.get_field<2>("topo"));
        boundary.fill_centered_q2_neumann_halos(state_.get_field<2>("topo"));
        initialize_topo();
        // Gather neighboring W masks at each owned positive face. The legacy
        // scatter into i-1/j-1 can target a halo on a decomposition boundary;
        // exchanging that halo does not transfer the write to its owner.
        auto& mask_w_field = state_.get_field<3>("ITYPEW");
        halo_exchanger_.exchange_halos(mask_w_field);
        boundary.fill_centered_q2_neumann_halos(mask_w_field);
        const auto mask_w = mask_w_field.get_device_data();
        const auto mask_u = state_.get_field<3>("ITYPEU").get_mutable_device_data();
        const auto mask_v = state_.get_field<3>("ITYPEV").get_mutable_device_data();
        Kokkos::parallel_for("RLLTerrainOwnedFaceMasks",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,h,h}, {nz,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                mask_u(k,j,i) = mask_w(k,j,i)*mask_w(k,j,i+1);
                mask_v(k,j,i) = mask_w(k,j,i)*mask_w(k,j+1,i);
            });
        halo_exchanger_.exchange_halos(state_.get_field<2>("topo"));
        boundary.fill_centered_q2_neumann_halos(state_.get_field<2>("topo"));
        for (const char* name : {"ITYPEU", "ITYPEV", "ITYPEW"}) {
            halo_exchanger_.exchange_halos(state_.get_field<3>(name));
            boundary.fill_centered_q2_neumann_halos(state_.get_field<3>(name));
        }
    }
}
} // namespace VVM::Core
