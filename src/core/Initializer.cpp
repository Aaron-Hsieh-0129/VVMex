#include "Initializer.hpp"
#include "BoundaryConditionManager.hpp"
#include "io/TxtReader.hpp"
#include "io/PnetcdfReader.hpp"
#include "io/Hdf5RestartReader.hpp"
#include "io/bp5/Bp5RestartReader.hpp"
#include "vvm_types.hpp"
#include "core/RegularLatLonModelConfiguration.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cctype>

namespace VVM {
namespace Core {

namespace {
bool
ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool
is_standard_test(const Utils::ConfigurationManager& config) {
    return config.get_value<std::string>("simulation.idealized_test", "none") != "none";
}
} // namespace

Initializer::Initializer(const Utils::ConfigurationManager& config,
    const Grid& grid,
    Parameters& parameters,
    State& state,
    HaloExchanger& halo_exchanger)
    : config_(config), grid_(grid), parameters_(parameters), state_(state),
      halo_exchanger_(halo_exchanger) {
    initialize_grid();

    const bool standard_test = is_standard_test(config);
    if (!standard_test) {
        if (!config.has_key("initial_conditions.format") ||
            !config.has_key("initial_conditions.source_file")) {
            throw std::runtime_error("[Initializer] Nonstandard cases require a text profile at "
                                     "'initial_conditions.source_file'.");
        }
        if (config.get_value<std::string>("initial_conditions.format") != "txt") {
            throw std::runtime_error("[Initializer] Nonstandard cases require "
                                     "'initial_conditions.format' to be 'txt' so vertical profiles "
                                     "remain separate from the spatial NetCDF file.");
        }
        if (!config.has_key("netcdf_reader.source_file")) {
            throw std::runtime_error("[Initializer] Nonstandard cases require a spatial "
                                     "initial-condition NetCDF at 'netcdf_reader.source_file'.");
        }
    }

    if (config.has_key("initial_conditions.format")) {
        const std::string format = config.get_value<std::string>("initial_conditions.format");
        if (format == "txt") {
            if (!config.has_key("initial_conditions.source_file")) {
                throw std::runtime_error("[Initializer] Missing profile path at "
                                         "'initial_conditions.source_file'.");
            }
            const std::string source_file =
                config.get_value<std::string>("initial_conditions.source_file");
            reader_ = std::make_unique<VVM::IO::TxtReader>(source_file, grid, parameters_, config_);
        }
        else if (format != "netcdf") {
            throw std::runtime_error("[Initializer] Unsupported initial_conditions.format '" +
                                     format + "'. Expected 'txt' or 'netcdf'.");
        }
    }

    if (config.has_key("netcdf_reader.source_file")) {
        const std::string source_file = config.get_value<std::string>("netcdf_reader.source_file");
        pnetcdf_reader_ = std::make_unique<VVM::IO::PnetcdfReader>(source_file,
            grid,
            parameters_,
            config_,
            halo_exchanger_);
    }
    if (config.get_value<bool>("restart.enable", false)) {
        std::string restart_source_file = config.get_value<std::string>("restart.source_file");
        // A trailing slash is easy to pick up when the source is a .bp
        // directory rather than a file, and would otherwise look like an
        // unsupported extension.
        while (restart_source_file.size() > 1 && restart_source_file.back() == '/') {
            restart_source_file.pop_back();
        }
        if (ends_with(restart_source_file, ".h5")) {
            restart_reader_ = std::make_unique<VVM::IO::Hdf5RestartReader>(restart_source_file,
                grid,
                parameters_,
                config_,
                halo_exchanger_);
        }
        else if (ends_with(restart_source_file, ".bp")) {
            restart_reader_ = std::make_unique<VVM::IO::BP5::Bp5RestartReader>(restart_source_file,
                grid,
                parameters_,
                config_,
                halo_exchanger_);
        }
        else if (ends_with(restart_source_file, ".nc")) {
            restart_reader_ = std::make_unique<VVM::IO::PnetcdfReader>(restart_source_file,
                grid,
                parameters_,
                config_,
                halo_exchanger_,
                "restart");
        }
        else {
            throw std::runtime_error(
                "[Initializer] Unsupported restart file extension: " + restart_source_file);
        }
    }
    // TODO: Initialize vorticity after loading velocity field
}

void
Initializer::initialize_state() const {
    initialize_background_state();

    // Construct case-specific topography first. For flat cases this is a no-op.
    initialize_case_terrain();

    // Common topology/mask construction for Cartesian and RLL geometries.
    initialize_topo();

    // RLL mountain face masks need a decomposition-independent correction after
    // the common ITYPE fields have been constructed.
    finalize_rll_terrain_masks();

    // Cartesian cases use assign_vars(); Jung/RLL cases use the analytic state.
    initialize_prognostic_state();

    // Tg requires both topography and the thermodynamic state to exist.
    initialize_surface_temperature();

    // Preserve the optional spatial-IC reapplication semantics. The initial
    // NetCDF read happens in initialize_background_state(); this optional second
    // read reapplies spatial fields after the generic prognostic initialization.
    if (pnetcdf_reader_ &&
        config_.get_value<bool>("initial_conditions.reapply_spatial_initial_conditions", false)) {
        pnetcdf_reader_->read_and_initialize(state_);
    }

    if (restart_reader_) {
        load_restart();
    }
    else if (!is_rll_idealized(config_)) {
        // Jung already contains its analytic perturbation.
        initialize_perturbation();
    }

    apply_tracer_boundary_conditions();

    // Poisson factors depend on the initialized reference density/state.
    initialize_poisson();
    initialize_zeta_factor_for_twisting();
}

void
Initializer::initialize_background_state() const {
    if (!is_rll_idealized(config_)) {
        // Preserve the ordinary VVMex reader order exactly: first the vertical
        // profile, then the spatial NetCDF state when one is configured.
        if (reader_) {
            reader_->read_and_initialize(state_);
        }

        if (pnetcdf_reader_) {
            pnetcdf_reader_->read_and_initialize(state_);
        }

        return;
    }

    // RLL uses a deliberately simple constant reference atmosphere:
    // rho = 1 kg m-3, Pi = 1, theta = T = Tv = 300 K, qv = 0.
    auto& rhobar = state_.get_field<1>("rhobar").get_mutable_device_data();
    auto& rhobar_up = state_.get_field<1>("rhobar_up").get_mutable_device_data();

    auto& pibar = state_.get_field<1>("pibar").get_mutable_device_data();
    auto& pibar_up = state_.get_field<1>("pibar_up").get_mutable_device_data();

    auto& pbar = state_.get_field<1>("pbar").get_mutable_device_data();
    auto& pbar_up = state_.get_field<1>("pbar_up").get_mutable_device_data();

    auto& thbar = state_.get_field<1>("thbar").get_mutable_device_data();
    auto& Tbar = state_.get_field<1>("Tbar").get_mutable_device_data();
    auto& Tvbar = state_.get_field<1>("Tvbar").get_mutable_device_data();
    auto& qvbar = state_.get_field<1>("qvbar").get_mutable_device_data();

    auto& th = state_.get_field<3>("th").get_mutable_device_data();
    auto& qv = state_.get_field<3>("qv").get_mutable_device_data();

    auto& v = state_.get_field<3>("v").get_mutable_device_data();
    auto& w = state_.get_field<3>("w").get_mutable_device_data();
    auto& xi = state_.get_field<3>("xi").get_mutable_device_data();
    auto& eta = state_.get_field<3>("eta").get_mutable_device_data();

    Kokkos::deep_copy(rhobar, real(1.0));
    Kokkos::deep_copy(rhobar_up, real(1.0));

    Kokkos::deep_copy(pibar, real(1.0));
    Kokkos::deep_copy(pibar_up, real(1.0));

    // Pi = 1 corresponds to the standard 1000-hPa reference pressure.
    constexpr double reference_pressure_pa = 100000.0;
    Kokkos::deep_copy(pbar, real(reference_pressure_pa));
    Kokkos::deep_copy(pbar_up, real(reference_pressure_pa));

    Kokkos::deep_copy(thbar, real(300.0));
    Kokkos::deep_copy(Tbar, real(300.0));
    Kokkos::deep_copy(Tvbar, real(300.0));

    Kokkos::deep_copy(qvbar, real(0.0));

    Kokkos::deep_copy(th, real(300.0));
    Kokkos::deep_copy(qv, real(0.0));

    // Jung supplies u and zeta analytically later. These fields have generic
    // zero initial values until initialize_jung2019() overwrites what it owns.
    Kokkos::deep_copy(v, real(0.0));
    Kokkos::deep_copy(w, real(0.0));
    Kokkos::deep_copy(xi, real(0.0));
    Kokkos::deep_copy(eta, real(0.0));
}

void
Initializer::initialize_case_terrain() const {
    // RLL Jung terrain is generated analytically here. Flat Jung cases leave
    // topo untouched and are handled by the common initialize_topo() call.
    if (is_rll_idealized(config_)) {
        if (!is_rll_mountain(config_)) {
            return;
        }

        const int h = grid_.get_halo_cells();
        const int nz = grid_.get_local_total_points_z();
        const int ny = grid_.get_local_total_points_y();
        const int nx = grid_.get_local_total_points_x();

        const auto& geometry = grid_.horizontal_specification().geometry;
        const Real radius = geometry.regular_lat_lon.radius;
        const Real dlambda = geometry.dq1;
        const Real dphi = geometry.dq2;
        const Real pi = std::acos(real(-1.0));
        const Real south = geometry.regular_lat_lon.latitude_south_edge;
        const Real west = geometry.regular_lat_lon.longitude_west_edge;

        Boundary::HorizontalBoundaryStencils boundary(grid_);

        // Preserve the existing RLL-mountain Coriolis initialization.
        const Real omega = config_.get_value<Real>("constants.OMEGA", real(0.));
        auto f = state_.get_field<2>("f_2d").get_host_data();
        for (int j = 0; j < ny; ++j) {
            const Real phi_z =
                south + (grid_.get_local_physical_start_y() + j - h + real(1.)) * dphi;
            for (int i = 0; i < nx; ++i) {
                f(j, i) = real(2.) * omega * std::sin(phi_z);
            }
        }
        Kokkos::deep_copy(state_.get_field<2>("f_2d").get_mutable_device_data(), f);

        const Real peak = config_.get_value<Real>("initial_conditions.rll_mountain.height_m");
        const Real width = config_.get_value<Real>("initial_conditions.rll_mountain.half_width_m");
        const Real center_lambda =
            config_.has_key("initial_conditions.rll_mountain.center_longitude_deg")
                ? config_.get_value<Real>("initial_conditions.rll_mountain.center_longitude_deg") *
                      pi / real(180.)
                : west + real(.5) * grid_.get_global_points_x() * dlambda;
        if (!std::isfinite(center_lambda)) {
            throw std::runtime_error("RLL mountain longitude must be finite.");
        }

        const Real center_phi =
            config_.get_value<Real>("initial_conditions.rll_mountain.center_latitude_deg",
                (south + real(.5) * grid_.get_global_points_y() * dphi) * real(180.) / pi) *
            pi / real(180.);

        state_.add_field<2>("rll_terrain_height",
            {ny, nx},
            {GridStaggering::Centered, "m", "discretized centered spherical mountain height"});
        auto elevation = state_.get_field<2>("rll_terrain_height").get_host_data();
        auto terrain = state_.get_field<2>("topo").get_host_data();
        const auto z = parameters_.z_up.get_host_data();

        for (int j = h; j < ny - h; ++j) {
            const Real phi = south + (grid_.get_local_physical_start_y() + j - h + real(.5)) * dphi;
            for (int i = h; i < nx - h; ++i) {
                const Real lambda =
                    west + (grid_.get_local_physical_start_x() + i - h + real(.5)) * dlambda;
                const Real cosine =
                    std::sin(phi) * std::sin(center_phi) +
                    std::cos(phi) * std::cos(center_phi) * std::cos(lambda - center_lambda);
                const Real distance =
                    radius * std::acos(std::max(real(-1.), std::min(real(1.), cosine)));
                const Real height = distance < real(3.) * width
                                        ? peak * std::exp(-distance * distance / (width * width))
                                        : real(0.);

                int level = h - 1;
                for (int k = h; k < nz - 2 * h - 2; ++k) {
                    if (std::abs(z(k) - height) < std::abs(z(level) - height)) {
                        level = k;
                    }
                }

                terrain(j, i) = level == h - 1 ? real(0.) : static_cast<Real>(level);
                elevation(j, i) = z(level);
            }
        }

        Kokkos::deep_copy(state_.get_field<2>("topo").get_mutable_device_data(), terrain);
        Kokkos::deep_copy(state_.get_field<2>("rll_terrain_height").get_mutable_device_data(),
            elevation);
        halo_exchanger_.exchange_halos(state_.get_field<2>("topo"));
        boundary.fill_centered_q2_neumann_halos(state_.get_field<2>("topo"));
        return;
    }

    // The only Cartesian terrain generated analytically by Initializer is the
    // existing Gaussian-mountain test. Other Cartesian cases use topo already
    // supplied by their readers.
    const std::string test_mode =
        config_.get_value<std::string>("simulation.idealized_test", "none");
    if (test_mode != "gaussian_mountain") {
        return;
    }

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const Real dx = parameters_.get_value_host(parameters_.dx);
    const Real length = grid_.get_global_points_x() * dx;
    const Real height =
        config_.get_value<Real>("initial_conditions.gaussian_mountain.height_m", real(800.));
    const Real width =
        config_.get_value<Real>("initial_conditions.gaussian_mountain.half_width_m", real(2000.));
    const Real center = config_.get_value<Real>("initial_conditions.gaussian_mountain.center_x_m",
        length / real(2.));
    const auto z = parameters_.z_up.get_host_data();

    if (grid_.geometry().kind() != Geometry::GeometryKind::Cartesian || !reader_ ||
        pnetcdf_reader_ || restart_reader_ || nz - 2 * h < 4 || !std::isfinite(height) ||
        height <= real(0.) || height > z(nz - h - 3) || !std::isfinite(width) ||
        width <= real(0.) || !std::isfinite(center)) {
        throw std::runtime_error(
            "gaussian_mountain requires Cartesian geometry, a text profile, no spatial "
            "input/restart, positive finite height/width, and at least two atmospheric levels "
            "above terrain.");
    }

    state_.add_field<2>("terrain_height", {ny, nx});
    auto& terrain = state_.get_field<2>("topo");
    auto topo = terrain.get_host_data();
    auto elevation = state_.get_field<2>("terrain_height").get_host_data();

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            const int global_i = grid_.get_local_physical_start_x() + i - h;
            const Real distance = std::remainder((global_i + real(.5)) * dx - center, length);
            const Real requested = height * std::exp(-distance * distance / (width * width));
            int level = h - 1;
            for (int k = h; k <= nz - h - 3; ++k) {
                if (std::abs(z(k) - requested) < std::abs(z(level) - requested)) {
                    level = k;
                }
            }
            // Input convention: zero denotes flat ground, otherwise an
            // absolute local vertical index (including the lower halo).
            topo(j, i) = level == h - 1 ? real(0.) : static_cast<Real>(level);
            elevation(j, i) = z(level);
        }
    }

    Kokkos::deep_copy(terrain.get_mutable_device_data(), topo);
    Kokkos::deep_copy(state_.get_field<2>("terrain_height").get_mutable_device_data(), elevation);
    halo_exchanger_.exchange_halos(terrain);
    halo_exchanger_.exchange_halos(state_.get_field<2>("terrain_height"));
}

void
Initializer::initialize_prognostic_state() const {
    if (is_rll_idealized(config_)) {
        initialize_jung2019();
    }
    else {
        assign_vars();
    }
}

void
Initializer::initialize_surface_temperature() const {
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const auto& topo = state_.get_field<2>("topo").get_device_data();
    const auto& pibar = state_.get_field<1>("pibar").get_device_data();
    const auto& th = state_.get_field<3>("th").get_device_data();

    auto& Tg = state_.get_field<2>("Tg").get_mutable_device_data();

    const std::string Tg_source =
        config_.get_value<std::string>("netcdf_reader.Tg_source", "atmosphere");

    if (Tg_source == "atmosphere") {
        Kokkos::parallel_for("Init_Tg",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
            KOKKOS_LAMBDA(const int j, const int i) {
                // NOTE: Fortran VVM uses hx rather than hxp here.
                const int hx = static_cast<int>(topo(j, i));

                Tg(j, i) = th(hx, j, i) * pibar(hx);
            });

        // Keep the current behavior during the refactor.
        halo_exchanger_.exchange_halos(state_.get_field<2>("Tg"));

        const int rank = grid_.get_mpi_rank();
        if (rank == 0) {
            std::cout << " [Initializer] Initialized Tg from the "
                         "lowest atmosphere level."
                      << std::endl;
        }
    }
    else if (Tg_source == "netcdf") {
        const int rank = grid_.get_mpi_rank();

        if (rank == 0) {
            std::cout << " [Initializer] Skipped Tg initialization "
                         "(using values read from NetCDF)."
                      << std::endl;
        }
    }
    else {
        const int rank = grid_.get_mpi_rank();

        if (rank == 0) {
            std::cerr << " [Initializer] Warning: Unknown Tg_source '" << Tg_source
                      << "'. Keeping existing Tg values." << std::endl;
        }
    }

    halo_exchanger_.exchange_halos(state_.get_field<2>("Tg"));
}

void
Initializer::finalize_rll_terrain_masks() const {
    if (!is_rll_idealized(config_) || !is_rll_mountain(config_)) {
        return;
    }

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    Boundary::HorizontalBoundaryStencils boundary(grid_);

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
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            mask_u(k, j, i) = mask_w(k, j, i) * mask_w(k, j, i + 1);
            mask_v(k, j, i) = mask_w(k, j, i) * mask_w(k, j + 1, i);
        });

    halo_exchanger_.exchange_halos(state_.get_field<2>("topo"));
    boundary.fill_centered_q2_neumann_halos(state_.get_field<2>("topo"));

    for (const char* name : {"ITYPEU", "ITYPEV", "ITYPEW"}) {
        halo_exchanger_.exchange_halos(state_.get_field<3>(name));
        boundary.fill_centered_q2_neumann_halos(state_.get_field<3>(name));
    }
}

void
Initializer::apply_tracer_boundary_conditions() const {
    if (state_.get_tracer_names().empty()) {
        return;
    }

    BoundaryConditionManager bc_manager(grid_);

    for (const auto& tracer_name : state_.get_tracer_names()) {
        auto& tracer = state_.get_field<3>(tracer_name);
        bc_manager.apply_horizontal_bcs(tracer);
        bc_manager.apply_zero_gradient(tracer);
    }
}

void
Initializer::load_restart() const {
    int rank = grid_.get_mpi_rank();
    const std::string restart_source_file = config_.get_value<std::string>("restart.source_file");
    if (rank == 0) {
        std::cout << "  [Initializer] Restart enabled. Loading restart variables from: "
                  << restart_source_file << std::endl;
    }

    restart_reader_->read_and_initialize(state_);

    // The clock comes from the file's own metadata, never from its name. Every
    // rank reads the same bytes, but rank 0's answer is broadcast anyway so the
    // recovered time and step are bit-identical on every rank by construction.
    VVM::Utils::RestartFileMetadata metadata = restart_reader_->read_restart_metadata();
    broadcast_restart_metadata(metadata);

    const auto resolution =
        Utils::resolve_restart_time(metadata, restart_time_policy(), restart_source_file);

    if (!resolution.ok) {
        throw std::runtime_error("[Initializer] " + resolution.error);
    }

    if (rank == 0) {
        for (const auto& warning : resolution.warnings) {
            std::cout << "  " << warning << std::endl;
        }
    }

    state_.set_time(static_cast<VVM::Real>(resolution.time_s));
    state_.set_step(static_cast<size_t>(resolution.step));

    if (rank == 0) {
        std::cout << "  [Initializer] Restart metadata: time=" << resolution.time_s
                  << " s, step=" << resolution.step
                  << " (source: " << Utils::to_string(resolution.source) << ")" << std::endl;
    }
}

Utils::RestartTimePolicy
Initializer::restart_time_policy() const {
    Utils::RestartTimePolicy policy;
    policy.dt_s = static_cast<double>(config_.get_value<VVM::Real>("simulation.dt_s"));
    policy.has_legacy_time = config_.has_key("restart.legacy_time_s");
    if (policy.has_legacy_time) {
        policy.legacy_time_s = config_.get_value<double>("restart.legacy_time_s");
    }
    // Filename parsing is what this whole path replaced: off unless a run asks
    // for it by name, and loud when it is on.
    policy.allow_filename_fallback =
        config_.get_value<bool>("restart.allow_filename_time_fallback", false);
    policy.filename_interval_s = config_.get_value<double>("restart.file_interval_s", 3600.0);
    policy.ignore_stored_step = config_.get_value<bool>("restart.ignore_stored_step", false);
    return policy;
}

void
Initializer::broadcast_restart_metadata(Utils::RestartFileMetadata& metadata) const {
    int flags[2] = {metadata.has_time ? 1 : 0, metadata.has_step ? 1 : 0};
    double time_s = metadata.time_s;
    long long step = metadata.step;

    MPI_Bcast(flags, 2, MPI_INT, 0, grid_.get_comm());
    MPI_Bcast(&time_s, 1, MPI_DOUBLE, 0, grid_.get_comm());
    MPI_Bcast(&step, 1, MPI_LONG_LONG, 0, grid_.get_comm());

    metadata.has_time = (flags[0] != 0);
    metadata.has_step = (flags[1] != 0);
    metadata.time_s = time_s;
    metadata.step = step;
}

void
Initializer::initialize_grid() const {
    const auto& vertical = grid_.vertical_specification();

    VVM::Real DOMAIN = real(15000.);
    VVM::Real dz = vertical.dz;
    VVM::Real dz1 = vertical.dz1;
    VVM::Real CZ2 = (dz - dz1) / (dz * (DOMAIN - dz));
    VVM::Real CZ1 = real(1.) - CZ2 * DOMAIN;

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();

    auto& z_mid_mutable = parameters_.z_mid.get_mutable_device_data();
    auto& z_up_mutable = parameters_.z_up.get_mutable_device_data();
    auto& flex_height_coef_mid_mutable = parameters_.flex_height_coef_mid.get_mutable_device_data();
    auto& flex_height_coef_up_mutable = parameters_.flex_height_coef_up.get_mutable_device_data();

    auto z_mid_mutable_h = parameters_.z_mid.get_host_data();
    auto z_up_mutable_h = parameters_.z_up.get_host_data();
    auto flex_height_coef_mid_h = parameters_.flex_height_coef_mid.get_host_data();
    auto flex_height_coef_up_h = parameters_.flex_height_coef_up.get_host_data();

    VVM::Real ZB = real(0.);
    z_up_mutable_h(h - 1) = ZB;

    const std::string v_coord_type = vertical_coordinate_type_to_string(vertical.type);

    if (v_coord_type == "taiwanvvm") {
        // TaiwanVVM Vertical Coordinate Logic
        for (int k = h; k < nz; k++) {
            z_up_mutable_h(k) = z_up_mutable_h(k - 1) + dz;
        }

        z_mid_mutable_h(h - 1) = z_up_mutable_h(h - 1);
        z_mid_mutable_h(h) = z_up_mutable_h(h - 1) + real(0.5) * dz;
        for (int k = h + 1; k < nz; k++) {
            z_mid_mutable_h(k) = z_mid_mutable_h(k - 1) + dz;
        }

        // Fortran: DO 40 (ZZ & FNZ transformation)
        for (int k = h - 1; k < nz; k++) {
            flex_height_coef_up_h(k) = real(1.) / (CZ1 + real(2.) * CZ2 * z_up_mutable_h(k));
            z_up_mutable_h(k) = z_up_mutable_h(k) * (CZ1 + CZ2 * z_up_mutable_h(k));
        }

        int KT = static_cast<int>((real(1.) - CZ1) / CZ2 / real(2.) / dz);
        int kt_idx = (KT + h - 2 >= nz) ? nz - 1 : KT + h - 2;
        if (kt_idx < 0) {
            kt_idx = 0;
        }
        int KT1 = static_cast<int>(z_up_mutable_h(kt_idx) / dz1 + real(0.999));

        z_up_mutable_h(h - 1) = ZB;
        flex_height_coef_up_h(h - 1) = dz / dz1;

        for (int K = 2; K <= KT1 - 1; K++) {
            int kc = K + h - 2;
            for (int KK = nz - 1; KK >= kc + 1; KK--) {
                flex_height_coef_up_h(KK) = flex_height_coef_up_h(KK - 1);
                z_up_mutable_h(KK) = z_up_mutable_h(KK - 1) + dz1;
            }
            if (kc < nz) {
                flex_height_coef_up_h(kc) = dz / dz1;
                z_up_mutable_h(kc) = z_up_mutable_h(kc - 1) + dz1;
            }
        }

        for (int k = h - 1; k < nz; k++) {
            flex_height_coef_mid_h(k) = real(1.) / (CZ1 + real(2.) * CZ2 * z_mid_mutable_h(k));
            z_mid_mutable_h(k) = z_mid_mutable_h(k) * (CZ1 + CZ2 * z_mid_mutable_h(k));
        }

        z_mid_mutable_h(h - 1) = ZB;
        flex_height_coef_mid_h(h - 1) = dz / dz1;
        z_mid_mutable_h(h) = z_up_mutable_h(h - 1) + dz1 / real(2.);
        flex_height_coef_mid_h(h) = dz / dz1;

        for (int K = 3; K <= KT1; K++) {
            int kc = K + h - 2;
            for (int KK = nz - 1; KK >= kc + 1; KK--) {
                flex_height_coef_mid_h(KK) = flex_height_coef_mid_h(KK - 1);
                z_mid_mutable_h(KK) = z_mid_mutable_h(KK - 1) + dz1;
            }
            if (kc < nz) {
                flex_height_coef_mid_h(kc) = dz / dz1;
                z_mid_mutable_h(kc) = z_mid_mutable_h(kc - 1) + dz1;
            }
        }

        int kt1_idx = KT1 + 1 + h - 2;
        if (kt1_idx < nz && (KT1 + 2 + h - 2) < nz) {
            flex_height_coef_up_h(kt1_idx) =
                dz / (z_mid_mutable_h(KT1 + 2 + h - 2) - z_mid_mutable_h(KT1 + 1 + h - 2));
        }

        Kokkos::deep_copy(z_up_mutable, z_up_mutable_h);
        Kokkos::deep_copy(z_mid_mutable, z_mid_mutable_h);
        Kokkos::deep_copy(flex_height_coef_up_mutable, flex_height_coef_up_h);
        Kokkos::deep_copy(flex_height_coef_mid_mutable, flex_height_coef_mid_h);
    }
    else if (v_coord_type == "rcemip") {
        const std::string& source_file = vertical.rcemip_grid_data_path;
        std::ifstream infile(source_file);

        if (!infile.is_open()) {
            throw std::runtime_error(
                "[Initializer] Cannot open rcemip source file: " + source_file);
        }

        std::string line;
        std::getline(infile, line);

        z_up_mutable_h(h - 1) = ZB;
        z_mid_mutable_h(h - 1) = ZB;

        for (int k = h; k < nz - h; k++) {
            if (!std::getline(infile, line)) {
                throw std::runtime_error(
                    "[Initializer] Unexpected end of file in rcemip source file.");
            }
            std::istringstream iss(line);
            VVM::Real zz_val, zt_val;

            if (!(iss >> zz_val >> zt_val)) {
                throw std::runtime_error(
                    "[Initializer] Error reading data from rcemip source file.");
            }

            z_up_mutable_h(k) = zz_val;
            z_mid_mutable_h(k) = zt_val;
        }
        infile.close();

        for (int k = nz - h; k < nz; k++) {
            z_up_mutable_h(k) = real(2.) * z_up_mutable_h(k - 1) - z_up_mutable_h(k - 2);
            z_mid_mutable_h(k) = real(2.) * z_mid_mutable_h(k - 1) - z_mid_mutable_h(k - 2);
        }

        for (int k = h - 2; k >= 0; k--) {
            z_up_mutable_h(k) =
                z_up_mutable_h(k + 1) - (z_up_mutable_h(k + 2) - z_up_mutable_h(k + 1));
            z_mid_mutable_h(k) =
                z_mid_mutable_h(k + 1) - (z_mid_mutable_h(k + 2) - z_mid_mutable_h(k + 1));
        }

        for (int k = 0; k < nz - 1; k++) {
            flex_height_coef_up_h(k) = dz / (z_mid_mutable_h(k + 1) - z_mid_mutable_h(k));
        }
        flex_height_coef_up_h(nz - 1) = flex_height_coef_up_h(nz - 2);

        for (int k = 1; k < nz; k++) {
            flex_height_coef_mid_h(k) = dz / (z_up_mutable_h(k) - z_up_mutable_h(k - 1));
        }
        flex_height_coef_mid_h(0) = flex_height_coef_mid_h(1);

        Kokkos::deep_copy(z_up_mutable, z_up_mutable_h);
        Kokkos::deep_copy(z_mid_mutable, z_mid_mutable_h);
        Kokkos::deep_copy(flex_height_coef_up_mutable, flex_height_coef_up_h);
        Kokkos::deep_copy(flex_height_coef_mid_mutable, flex_height_coef_mid_h);

        int rank = grid_.get_mpi_rank();
        if (rank == 0) {
            std::cout << "  [Initializer] Initialized RCEMIP vertical coordinate from "
                      << source_file << std::endl;
        }
    }
    else {
        for (int k = h; k < nz; k++) {
            z_up_mutable_h(k) = z_up_mutable_h(k - 1) + dz;
        }
        z_mid_mutable_h(h - 1) = z_up_mutable_h(h - 1);
        z_mid_mutable_h(h) = z_up_mutable_h(h - 1) + real(0.5) * dz;
        for (int k = h + 1; k < nz; k++) {
            z_mid_mutable_h(k) = z_mid_mutable_h(k - 1) + dz;
        }

        Kokkos::deep_copy(z_up_mutable, z_up_mutable_h);
        Kokkos::deep_copy(z_mid_mutable, z_mid_mutable_h);

        Kokkos::parallel_for("Init_Z_flexZCoef",
            Kokkos::RangePolicy<>(h - 1, nz),
            KOKKOS_LAMBDA(const int k) {
                flex_height_coef_mid_mutable(k) =
                    real(1.) / (CZ1 + real(2.) * CZ2 * z_mid_mutable(k));
                flex_height_coef_up_mutable(k) =
                    real(1.) / (CZ1 + real(2.) * CZ2 * z_up_mutable(k));
                z_mid_mutable(k) = z_mid_mutable(k) * (CZ1 + CZ2 * z_mid_mutable(k));
                z_up_mutable(k) = z_up_mutable(k) * (CZ1 + CZ2 * z_up_mutable(k));
            });
    }

    auto& dz_mid_mutable = parameters_.dz_mid.get_mutable_device_data();
    auto& dz_up_mutable = parameters_.dz_up.get_mutable_device_data();
    Kokkos::parallel_for("Init_dz", Kokkos::RangePolicy<>(h, nz - h), KOKKOS_LAMBDA(const int k) {
        dz_mid_mutable(k) = z_up_mutable(k) - z_up_mutable(k - 1);
        dz_up_mutable(k) = z_mid_mutable(k + 1) - z_mid_mutable(k);
    });

    auto& fact1_xi_eta_mutable = parameters_.fact1_xi_eta.get_mutable_device_data();
    auto& fact2_xi_eta_mutable = parameters_.fact2_xi_eta.get_mutable_device_data();
    Kokkos::parallel_for("Init_zflex_fact",
        Kokkos::RangePolicy<>(h, nz - h - 1),
        KOKKOS_LAMBDA(const int k) {
            fact1_xi_eta_mutable(k) =
                flex_height_coef_up_mutable(k) / flex_height_coef_mid_mutable(k + 1);
            fact2_xi_eta_mutable(k) =
                flex_height_coef_up_mutable(k) / flex_height_coef_mid_mutable(k);
        });
    return;
}

void
Initializer::initialize_topo() const {
    auto dims =
        std::array<int, 2>{grid_.get_local_total_points_y(), grid_.get_local_total_points_x()};
    if (!state_.has_field("topou")) {
        state_.add_field<2>("topou", dims);
    }
    if (!state_.has_field("topov")) {
        state_.add_field<2>("topov", dims);
    }

    const auto& topo = state_.get_field<2>("topo").get_device_data();
    auto& ITYPEU = state_.get_field<3>("ITYPEU").get_mutable_device_data();
    auto& ITYPEV = state_.get_field<3>("ITYPEV").get_mutable_device_data();
    auto& ITYPEW = state_.get_field<3>("ITYPEW").get_mutable_device_data();
    auto topo_h = state_.get_field<2>("topo").get_host_data();

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    VVM::Real local_maxtopo_h, maxtopo_h;
    Kokkos::parallel_reduce("FindMax",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i, VVM::Real& local_max) {
            if (topo(j, i) > local_max) {
                local_max = topo(j, i);
            }
        },
        Kokkos::Max<VVM::Real>(local_maxtopo_h));

    MPI_Allreduce(&local_maxtopo_h, &maxtopo_h, 1, VVM_MPI_REAL, MPI_MAX, grid_.get_comm());
    maxtopo_h += h;
    parameters_.max_topo_idx = maxtopo_h;

    // Assign ITYPE
    Kokkos::deep_copy(ITYPEU, real(1.));
    Kokkos::deep_copy(ITYPEV, real(1.));
    Kokkos::deep_copy(ITYPEW, real(1.));
    Kokkos::parallel_for("assign_ITYPE",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (topo(j, i) != 0) {
                for (int k = 0; k <= topo(j, i); k++) {
                    ITYPEU(k, j, i) = real(0.);
                    ITYPEV(k, j, i) = real(0);
                    ITYPEW(k, j, i) = real(0);
                }
            }
        });
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEW"));

    Kokkos::parallel_for("assign_ITYPE",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEW(k, j, i) == 0) {
                ITYPEU(k, j, i - 1) = real(0);
                ITYPEV(k, j - 1, i) = real(0);
            }
        });
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEU"));
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEV"));

    // Assign topou, topov
    auto& topou = state_.get_field<2>("topou").get_mutable_device_data();
    auto& topov = state_.get_field<2>("topov").get_mutable_device_data();
    Kokkos::parallel_for("assign_topou_topov",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (topo(j, i + 1) - topo(j, i) > 0) {
                topou(j, i) = topo(j, i + 1);
            }
            if (topo(j + 1, i) - topo(j, i) > 0) {
                topov(j, i) = topo(j + 1, i);
            }
        });

    Kokkos::parallel_for("modifyTopo",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (topo(j, i) == 0) {
                topo(j, i) = h - 1;
            }
        });
    return;
}

void
Initializer::initialize_poisson() const {
    const auto& rdx2 = parameters_.rdx2;
    const auto& rdy2 = parameters_.rdy2;
    const auto& rdz2 = parameters_.rdz2;
    const auto& WRXMU = parameters_.WRXMU;
    const auto& flex_height_coef_mid = parameters_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = parameters_.flex_height_coef_up.get_device_data();
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state_.get_field<1>("rhobar_up").get_device_data();

    // Poisson iteration coefficient
    auto& AGAU = parameters_.AGAU.get_mutable_device_data();
    auto& BGAU = parameters_.BGAU.get_mutable_device_data();
    auto& CGAU = parameters_.CGAU.get_mutable_device_data();
    Kokkos::parallel_for("Init_Poisson_Coef",
        Kokkos::RangePolicy<>(0, nz),
        KOKKOS_LAMBDA(const int k) {
            if (k >= h && k <= nz - h - 2) {
                AGAU(k) = -flex_height_coef_up(k) * flex_height_coef_mid(k) * rdz2() / rhobar(k);
                BGAU(k) = (WRXMU() + real(2.) * rdx2() + real(2.) * rdy2()) / rhobar_up(k) +
                          flex_height_coef_up(k) *
                              (flex_height_coef_mid(k + 1) / rhobar(k + 1) +
                                  flex_height_coef_mid(k) / rhobar(k)) *
                              rdz2();
                CGAU(k) =
                    -flex_height_coef_up(k) * flex_height_coef_mid(k + 1) * rdz2() / rhobar(k + 1);
            }
            else {
                AGAU(k) = BGAU(k) = CGAU(k) = real(-9e16);
            }
        });

    auto& bn_new = parameters_.bn_new.get_mutable_device_data();
    auto& cn_new = parameters_.cn_new.get_mutable_device_data();

    auto h_AGAU = parameters_.AGAU.get_host_data();
    auto h_BGAU = parameters_.BGAU.get_host_data();
    auto h_CGAU = parameters_.CGAU.get_host_data();
    auto h_bn_new = parameters_.bn_new.get_host_data();
    auto h_cn_new = parameters_.cn_new.get_host_data();

    for (int k = 0; k <= nz; k++) {
        if (k == h) {
            h_cn_new(h) = h_CGAU(h) / h_BGAU(h);
        }
        else if (k >= h + 1 && k <= nz - h - 2) {
            h_bn_new(k) = h_BGAU(k) - h_AGAU(k) * h_cn_new(k - 1);
            h_cn_new(k) = h_CGAU(k) / h_bn_new(k);
        }
        else {
            h_bn_new(k) = h_cn_new(k) = real(-9e16);
        }
    }
    Kokkos::deep_copy(bn_new, h_bn_new);
    Kokkos::deep_copy(cn_new, h_cn_new);
    return;
}

void
Initializer::initialize_zeta_factor_for_twisting() const {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();

    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state_.get_field<1>("rhobar_up").get_device_data();
    const auto& flex_height_coef_mid = parameters_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = parameters_.flex_height_coef_up.get_device_data();

    auto& fact1_zeta_mutable = parameters_.fact1_zeta.get_mutable_device_data();
    auto& fact2_zeta_mutable = parameters_.fact2_zeta.get_mutable_device_data();
    Kokkos::parallel_for("AssignFactor", 1, KOKKOS_LAMBDA(const int) {
        fact1_zeta_mutable() = flex_height_coef_mid(nz - h - 1) * rhobar_up(nz - h - 1) /
                               flex_height_coef_up(nz - h - 1);
        fact2_zeta_mutable() = flex_height_coef_mid(nz - h - 1) * rhobar_up(nz - h - 2) /
                               flex_height_coef_up(nz - h - 2);
    });
}

void
Initializer::assign_vars() const {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    // Vertical B.C. process
    // WARNING: This causes errors in P3
    // VVM::Core::BoundaryConditionManager bc_manager(grid_);
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("thbar")); bc_manager.apply_z_bcs_to_field(state_.get_field<1>("qvbar")); bc_manager.apply_z_bcs_to_field(state_.get_field<1>("Tbar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("Tvbar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("rhobar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("rhobar_up"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("pbar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("pibar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("U"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("V"));

    int rank = grid_.get_mpi_rank();
    if (rank == 0) {
        state_.get_field<1>("qvbar").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("rhobar_up").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("rhobar").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("thbar").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("Tbar").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("Tvbar").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("pibar").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("pibar_up").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("pbar").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        parameters_.z_mid.print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        parameters_.z_up.print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        parameters_.flex_height_coef_mid.print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        parameters_.flex_height_coef_up.print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("U").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0) {
        state_.get_field<1>("V").print_profile(grid_, 0, 0, 0);
    }
    if (rank == 0 && state_.has_field("Q1")) {
        state_.get_field<1>("Q1").print_profile(grid_, 0, 0, 0);
    }

    const auto& rdx = parameters_.rdx;
    const auto& rdy = parameters_.rdy;
    const auto& rdz = parameters_.rdz;
    const auto& flex_height_coef_mid = parameters_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = parameters_.flex_height_coef_up.get_device_data();

    auto& u = state_.get_field<3>("u").get_mutable_device_data();
    auto& v = state_.get_field<3>("v").get_mutable_device_data();
    const auto& w = state_.get_field<3>("w").get_device_data();
    const auto& U = state_.get_field<1>("U").get_device_data();
    const auto& V = state_.get_field<1>("V").get_device_data();
    Kokkos::parallel_for("assign_initial_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            u(k, j, i) = U(k);
            v(k, j, i) = V(k);
        });
    Kokkos::deep_copy(w, real(0.));
    // utop predict
    VVM::Core::ScalarView utopmn("utopmn");
    VVM::Core::ScalarView vtopmn("vtopmn");
    state_.calculate_horizontal_mean(state_.get_field<3>("u"), utopmn);
    state_.calculate_horizontal_mean(state_.get_field<3>("v"), vtopmn);
    auto utopmn_view = state_.get_field<0>("utopmn").get_mutable_device_data();
    auto vtopmn_view = state_.get_field<0>("vtopmn").get_mutable_device_data();
    Kokkos::parallel_for("assign_uvtopmn", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int k) {
        utopmn_view() = utopmn();
        vtopmn_view() = vtopmn();
    });

    auto& eta = state_.get_field<3>("eta").get_mutable_device_data();
    auto& xi = state_.get_field<3>("xi").get_mutable_device_data();
    Kokkos::parallel_for("assign_vorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h - 1, ny - h, nx - h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            eta(k, j, i) = (w(k, j, i + 1) - w(k, j, i)) * rdx() -
                           (u(k + 1, j, i) - u(k, j, i)) * rdz() * flex_height_coef_up(k);
            xi(k, j, i) = (w(k, j + 1, i) - w(k, j, i)) * rdy() -
                          (v(k + 1, j, i) - v(k, j, i)) * rdz() * flex_height_coef_up(k);
        });

    // Assign pbar_up
    const auto& pbar = state_.get_field<1>("pbar").get_device_data();
    auto& pbar_up = state_.get_field<1>("pbar_up").get_mutable_device_data();
    auto& dpbar_mid = state_.get_field<1>("dpbar_mid").get_mutable_device_data();
    Kokkos::parallel_for("assign_pbar_up",
        Kokkos::RangePolicy<>(1, nz),
        KOKKOS_LAMBDA(const int k) {
            if (k == 1) {
                pbar_up(k) = pbar(k);
            }
            else {
                pbar_up(k) = real(0.5) * (pbar(k) + pbar(k + 1));
            }
        });
    Kokkos::parallel_for("assign_pbar_up",
        Kokkos::RangePolicy<>(2, nz),
        KOKKOS_LAMBDA(const int k) {
            dpbar_mid(k) = -(pbar_up(k) - pbar_up(k - 1)); // make it positive
        });
    if (rank == 0) {
        state_.get_field<1>("pbar_up").print_profile(grid_, 0, 0, 0);
    }

    // Assign qv
    const auto& qvbar = state_.get_field<1>("qvbar").get_device_data();
    auto& qv = state_.get_field<3>("qv").get_mutable_device_data();
    Kokkos::parallel_for("assign_qv",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(int k, int j, int i) { qv(k, j, i) = qvbar(k); });

    // Assign th
    auto& thbar = state_.get_field<1>("thbar").get_mutable_device_data();
    auto& th = state_.get_field<3>("th").get_mutable_device_data();
    auto& rhobar = state_.get_field<1>("rhobar").get_mutable_device_data();
    auto& rhobar_up = state_.get_field<1>("rhobar_up").get_mutable_device_data();
    Kokkos::parallel_for("assign_th",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(int k, int j, int i) { th(k, j, i) = thbar(k); });
    std::string test_mode = config_.get_value<std::string>("simulation.idealized_test", "none");
    if (test_mode == "2dbubble") {
        Kokkos::parallel_for("assign_th",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                thbar(k) = 300.;
                th(k, j, i) = thbar(k);
                rhobar(k) = rhobar_up(k) = 1;
            });
    }

    initialize_geographic_coordinates();

    // A configured reference value selects constant f by default. An optional
    // beta adds a linear north-south gradient while preserving the same
    // Coriolis tendency implementation. Preserve the legacy initialization
    // when the reference value is absent so existing configurations remain
    // unchanged.
    VVM::Real OMEGA = config_.get_value<VVM::Real>("constants.OMEGA", real(7.292e-5));
    const bool use_constant_coriolis = config_.has_key("constants.coriolis_parameter");
    const VVM::Real constant_coriolis =
        config_.get_value<VVM::Real>("constants.coriolis_parameter", real(0.0));
    const VVM::Real coriolis_beta =
        config_.get_value<VVM::Real>("constants.coriolis_beta", real(0.0));
    const VVM::Real coriolis_reference_y =
        config_.get_value<VVM::Real>("constants.coriolis_reference_y_m",
            real(0.5) * grid_.get_global_points_y() * grid_.get_dy());
    auto& f = state_.get_field<1>("f").get_mutable_device_data();
    auto& f_2d = state_.get_field<2>("f_2d").get_mutable_device_data();
    const int global_start_j = grid_.get_local_physical_start_y();
    const VVM::Real grid_dy = grid_.get_dy();
    const auto& dy = parameters_.dy;
    Kokkos::parallel_for("Init_Coriolis",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const int global_j = global_start_j + j - h;
            const VVM::Real y = (static_cast<VVM::Real>(global_j) + real(0.5)) * grid_dy;
            const VVM::Real coriolis =
                use_constant_coriolis
                    ? constant_coriolis + coriolis_beta * (y - coriolis_reference_y)
                    : real(2.) * OMEGA *
                          Kokkos::sin((global_j - real(1540.) / real(2.) - real(0.5)) * dy() /
                                      real(6.37e6));
            // VVM::Real coriolis = 5e-5;
            f(j) = coriolis;
            f_2d(j, i) = coriolis;
        });
    halo_exchanger_.exchange_halos(state_.get_field<2>("f_2d"));

    return;
}

void
Initializer::initialize_perturbation() const {
    std::string perturbation =
        config_.get_value<std::string>("initial_conditions.perturbation", "none");
    const int global_start_j = grid_.get_local_physical_start_y();
    const int global_start_i = grid_.get_local_physical_start_x();
    const auto& dx = parameters_.dx;
    const auto& dy = parameters_.dy;
    const auto& z_mid = parameters_.z_mid.get_device_data();

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    // Domain centre for the bubble perturbations. nx/ny above are this RANK's local
    // totals, so deriving the centre from them moved the bubble whenever the
    // decomposition changed. Take it from the global extent instead, in the same
    // halo-shifted frame as global_i/global_j (on one rank this is identical to the
    // old (int)(nx/2), so existing baselines are unchanged).
    const int global_center_i = static_cast<int>(grid_.get_global_points_x() / 2) + h;
    const int global_center_j = static_cast<int>(grid_.get_global_points_y() / 2) + h;
    VVM::Real PI = config_.get_value<VVM::Real>("constants.PI");

    auto& th = state_.get_field<3>("th").get_mutable_device_data();
    auto& xi = state_.get_field<3>("xi").get_mutable_device_data();
    auto& eta = state_.get_field<3>("eta").get_mutable_device_data();
    auto& zeta = state_.get_field<3>("zeta").get_mutable_device_data();
    auto& u = state_.get_field<3>("u").get_mutable_device_data();
    auto& v = state_.get_field<3>("v").get_mutable_device_data();
    auto& w = state_.get_field<3>("w").get_mutable_device_data();

    std::string test_mode = config_.get_value<std::string>("simulation.idealized_test", "none");
    if (test_mode == "advection_u" || test_mode == "advection_v" || test_mode == "advection_w" ||
        test_mode == "stretching" || test_mode == "twisting") {
        auto& rhobar = state_.get_field<1>("rhobar").get_mutable_device_data();
        auto& rhobar_up = state_.get_field<1>("rhobar_up").get_mutable_device_data();

        if (test_mode == "advection_w") {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), th, real(300.));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), rhobar, real(1.));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), rhobar_up, real(1.));
        }

        Kokkos::parallel_for("test_init",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Global indices: with local j/i every rank stamped its own blob, so a
                // 2-D decomposition produced one blob per rank instead of one per domain.
                const int global_j = global_start_j + j;
                const int global_i = global_start_i + i;

                if (k == h + 16 && (h + 3 <= global_j && h + 11 >= global_j) &&
                    (h + 3 <= global_i && h + 11 >= global_i)) {
                    th(k, j, i) += real(50.);
                    xi(k, j, i) += real(50.);
                    eta(k, j, i) += real(50.);
                    zeta(nz - h - 1, j, i) += real(50.);
                }
            });
        if (test_mode == "advection_u") {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), u, real(10.));
        }
        else if (test_mode == "advection_v") {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), v, real(10.));
        }
        else if (test_mode == "advection_w") {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), w, real(10.));
        }
        else if (test_mode == "stretching") {
            Kokkos::parallel_for("test_wind_init",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    const int global_j = global_start_j + j;
                    const int global_i = global_start_i + i;

                    u(k, j, i) = real(32.) / real(2.) - global_i - real(1.);
                    v(k, j, i) = real(32.) / real(2.) - global_j - real(1.);
                    w(k, j, i) = -(real(32.) / real(2.) - k - real(1.));
                });
        }
        else if (test_mode == "twisting") {
            Kokkos::parallel_for("test_wind_init_cross_derivatives",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    const int global_j = global_start_j + j;
                    const int global_i = global_start_i + i;

                    if (k == nz - h - 1 && (h + 3 <= global_j && h + 11 >= global_j) &&
                        (h + 3 <= global_i && h + 11 >= global_i)) {
                        xi(k, j, i) += real(50.);
                        eta(k, j, i) += real(50.);
                    }

                    u(k, j, i) = -real(0.2) * (real(32.) / real(2.) - global_j - real(1.));
                    v(k, j, i) = -real(0.2) * (real(32.) / real(2.) - global_i - real(1.));
                    w(k, j, i) = real(0.2) * (real(32.) / real(2.) - global_i - real(1.));
                });
        }
    }
    else if (test_mode == "2dbubble") {
        Kokkos::parallel_for("init_perturbation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int global_i = global_start_i + i;

                VVM::Real radius_norm = Kokkos::sqrt(
                    Kokkos::pow(((global_i + 1) - global_center_i) * dx() / real(1000.), 2) +
                    Kokkos::pow((z_mid(k) - real(5000.)) / real(1000.), 2));
                if (radius_norm <= real(1.)) {
                    th(k, j, i) += real(5.) * (Kokkos::cos(PI * real(0.5) * radius_norm));
                }
            });
    }
    else if (test_mode == "3dbubble") {
        Kokkos::parallel_for("init_perturbation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int global_i = global_start_i + i;
                const int global_j = global_start_j + j;

                VVM::Real radius_norm = Kokkos::sqrt(
                    Kokkos::pow(((global_i + 1) - global_center_i) * dx() / real(1000.), 2) +
                    Kokkos::pow(((global_j + 1) - global_center_j) * dy() / real(1000.), 2) +
                    Kokkos::pow((z_mid(k) - real(5000.)) / real(1000.), 2));
                if (radius_norm <= real(1.)) {
                    th(k, j, i) += real(5.) * (Kokkos::cos(PI * real(0.5) * radius_norm));
                }
            });
    }
    else if (test_mode == "topo") {
        auto& utopmn = state_.get_field<0>("utopmn").get_mutable_device_data();
        Kokkos::deep_copy(u, real(10.));
        Kokkos::deep_copy(utopmn, real(10.));
    }

    if (perturbation == "none") {
        return;
    }
    else if (perturbation == "2dbubble") {
        Kokkos::parallel_for("init_perturbation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int global_i = global_start_i + i;

                VVM::Real radius_norm = Kokkos::sqrt(
                    Kokkos::pow(((global_i + 1) - global_center_i) * dx() / real(5000.), 2) +
                    Kokkos::pow((z_mid(k) - real(1000.)) / real(2000.), 2));
                if (radius_norm <= real(1.)) {
                    th(k, j, i) += real(5.) * (Kokkos::cos(PI * real(0.5) * radius_norm));
                }
            });
    }
    else if (perturbation == "3dbubble") {
        Kokkos::parallel_for("init_perturbation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int global_j = global_start_j + j;
                const int global_i = global_start_i + i;

                VVM::Real radius_norm = Kokkos::sqrt(
                    Kokkos::pow(((global_i + 1) - global_center_i) * dx() / real(2000.), 2) +
                    Kokkos::pow(((global_j + 1) - global_center_j) * dy() / real(2000.), 2) +
                    Kokkos::pow((z_mid(k) - real(3000.)) / real(2000.), 2));
                if (radius_norm <= real(1.)) {
                    th(k, j, i) += real(5.) * (Kokkos::cos(PI * real(0.5) * radius_norm));
                }
            });
    }

    halo_exchanger_.exchange_multiple_halos({"th", "xi", "eta", "zeta"}, state_);
    if (test_mode == "none") {
        halo_exchanger_.exchange_multiple_halos({"u", "v", "w"}, state_);
    }
    return;
}

void
Initializer::initialize_geographic_coordinates() const {
    if (grid_.geometry().kind() == Geometry::GeometryKind::RegularLatLon) {
        // Geometry is authoritative on RLL, including local halos. Spatial
        // input and the Cartesian fixed-location override must not replace it.
        const auto geometry = grid_.geometry().device_view(GridStaggering::Centered);
        const auto longitude = geometry.longitude;
        const auto latitude = geometry.latitude;
        auto lon = state_.get_field<2>("lon").get_mutable_device_data();
        auto lat = state_.get_field<2>("lat").get_mutable_device_data();
        const Real degrees = real(180.) / std::acos(real(-1.));
        Kokkos::parallel_for("InitializeRLLGeographicCoordinates",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                {grid_.get_local_total_points_y(), grid_.get_local_total_points_x()}),
            KOKKOS_LAMBDA(int j, int i) {
                lon(j, i) = longitude(j, i) * degrees;
                lat(j, i) = latitude(j, i) * degrees;
            });
        return;
    }
    // Without the fixed-location override, preserve coordinates supplied by
    // the existing spatial initial-condition reader.
    if (!grid_.horizontal_specification().fix_lonlat) {
        return;
    }

    auto& lon = state_.get_field<2>("lon").get_mutable_device_data();
    auto& lat = state_.get_field<2>("lat").get_mutable_device_data();

    Kokkos::deep_copy(lon, real(120.95));
    Kokkos::deep_copy(lat, real(23.458));
}

} // namespace Core
} // namespace VVM
