#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "physics/turbulence/TurbulenceProcess.hpp"
#include "utils/ConfigurationManager.hpp"
#include "../../externals/json/json.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

#if defined(ENABLE_NCCL)
#include <nccl.h>
#endif

namespace {

using VVM::Real;
using VVM::real;

using VVM::Core::Grid;
using VVM::Core::Parameters;
using VVM::Core::State;

using VVM::Core::Geometry::HorizontalLocation;
using VVM::Physics::TurbulenceProcess;
using VVM::Utils::ConfigurationManager;

using Json = nlohmann::json;

// ============================================================================
// Test configuration
// ============================================================================

constexpr int NX = 24;
constexpr int NY = 16;
constexpr int HALO = 2;

constexpr Real EARTH_RADIUS = real(6371220.0);
constexpr Real K0 = real(100.0);

constexpr Real DEG_TO_RAD = real(3.141592653589793238462643383279502884) / real(180.0);

constexpr Real LON_WEST = real(-30.0) * DEG_TO_RAD;

constexpr Real LON_EAST = real(30.0) * DEG_TO_RAD;

constexpr Real LAT_SOUTH = real(-20.0) * DEG_TO_RAD;

constexpr Real LAT_NORTH = real(20.0) * DEG_TO_RAD;

constexpr Real DLON = (LON_EAST - LON_WEST) / real(NX);

constexpr Real DLAT = (LAT_NORTH - LAT_SOUTH) / real(NY);

constexpr Real DZ = real(250.0);

int failures = 0;

// ============================================================================
// Small test utilities
// ============================================================================

void
check(const bool condition, const std::string& message) {

    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

void
report(const std::string& name, const bool passed) {

    std::printf("%-48s %s\n", name.c_str(), passed ? "PASS" : "FAIL");
}

Real
q1_offset(const HorizontalLocation location) {

    switch (location) {
    case HorizontalLocation::T:
    case HorizontalLocation::V:
        return real(0.5);

    case HorizontalLocation::U:
    case HorizontalLocation::Z:
        return real(1.0);
    }

    throw std::runtime_error("invalid horizontal location");
}

Real
q2_offset(const HorizontalLocation location) {

    switch (location) {
    case HorizontalLocation::T:
    case HorizontalLocation::U:
        return real(0.5);

    case HorizontalLocation::V:
    case HorizontalLocation::Z:
        return real(1.0);
    }

    throw std::runtime_error("invalid horizontal location");
}

Real
longitude_at(const HorizontalLocation location, const int i, const int h) {

    return LON_WEST + (real(i - h) + q1_offset(location)) * DLON;
}

Real
latitude_at(const HorizontalLocation location, const int j, const int h) {

    return LAT_SOUTH + (real(j - h) + q2_offset(location)) * DLAT;
}

// ============================================================================
// Field utilities
// ============================================================================

void
fill_3d(State& state, const std::string& name, const Real value) {

    Kokkos::deep_copy(state.get_field<3>(name).get_mutable_device_data(), value);
}

void
fill_2d(State& state, const std::string& name, const Real value) {

    Kokkos::deep_copy(state.get_field<2>(name).get_mutable_device_data(), value);
}

void
fill_1d(State& state, const std::string& name, const Real value) {

    Kokkos::deep_copy(state.get_field<1>(name).get_mutable_device_data(), value);
}

void
reset_tendency_3d(State& state, const std::string& variable) {

    Kokkos::deep_copy(state.get_field<3>("fe_tendency_" + variable).get_mutable_device_data(),
        real(0.0));
}

void
reset_tendency_2d(State& state, const std::string& variable) {

    Kokkos::deep_copy(state.get_field<2>("fe_tendency_" + variable).get_mutable_device_data(),
        real(0.0));
}

// ============================================================================
// Initialize a clean flat-domain turbulence fixture
// ============================================================================

void
initialize_base_state(State& state, Parameters& params, const Grid& grid) {

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const auto dims2 = std::array<int, 2>{ny, nx};

    // ------------------------------------------------------------
    // Flat terrain and completely active fluid cells.
    // ------------------------------------------------------------
    if (!state.has_field("topou")) {
        state.add_field<2>("topou", dims2);
    }

    if (!state.has_field("topov")) {
        state.add_field<2>("topov", dims2);
    }

    fill_2d(state, "topo", real(0.0));
    fill_2d(state, "topou", real(0.0));
    fill_2d(state, "topov", real(0.0));

    fill_3d(state, "ITYPEU", real(1.0));
    fill_3d(state, "ITYPEV", real(1.0));
    fill_3d(state, "ITYPEW", real(1.0));

    // ------------------------------------------------------------
    // Basic dynamical/thermodynamic state.
    // ------------------------------------------------------------

    fill_3d(state, "u", real(0.0));
    fill_3d(state, "v", real(0.0));
    fill_3d(state, "w", real(0.0));

    fill_3d(state, "xi", real(0.0));
    fill_3d(state, "eta", real(0.0));
    fill_3d(state, "zeta", real(0.0));

    fill_3d(state, "th", real(300.0));
    fill_3d(state, "qv", real(0.0));

    fill_3d(state, "R_xi", real(0.0));
    fill_3d(state, "R_eta", real(0.0));
    fill_3d(state, "R_zeta", real(0.0));

    fill_1d(state, "rhobar", real(1.0));
    fill_1d(state, "rhobar_up", real(1.0));

    // ------------------------------------------------------------
    // Make the vertical grid deliberately simple.
    //
    // The horizontal tests make fields constant in z, so vertical
    // turbulent diffusion must vanish exactly.
    // ------------------------------------------------------------

    Kokkos::deep_copy(params.flex_height_coef_mid.get_mutable_device_data(), real(1.0));

    Kokkos::deep_copy(params.flex_height_coef_up.get_mutable_device_data(), real(1.0));

    Kokkos::deep_copy(params.dz_mid.get_mutable_device_data(), DZ);

    Kokkos::deep_copy(params.dz_up.get_mutable_device_data(), DZ);

    auto z_mid = params.z_mid.get_mutable_device_data();

    Kokkos::parallel_for("InitializeTurbulenceTestZ",
        Kokkos::RangePolicy<>(0, nz),
        KOKKOS_LAMBDA(const int k) { z_mid(k) = (real(k - HALO) + real(0.5)) * DZ; });

    // Needed by TurbulenceProcess::init_boundary_masks().
    params.max_topo_idx = HALO + 1;

    Kokkos::fence();
}

// ============================================================================
// Prescribe constant Km / Kh
//
// This isolates the diffusion operator from the Shutts-Gray closure.
// ============================================================================

void
set_constant_diffusivity(State& state) {

    fill_3d(state, "RKM", K0);
    fill_3d(state, "RKH", K0);

    Kokkos::fence();
}

// ============================================================================
// Manufactured field setters
// ============================================================================

enum class ManufacturedPattern { SinLatitude, SinLongitude };

void
set_manufactured_field(State& state,
    const Grid& grid,
    const std::string& field_name,
    const HorizontalLocation location,
    const ManufacturedPattern pattern,
    const int zonal_mode = 2) {

    auto& field = state.get_field<3>(field_name).get_mutable_device_data();

    const auto geometry = grid.geometry().device_view(location);

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    Kokkos::parallel_for("SetTurbulenceManufacturedField_" + field_name,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{0, 0, 0}}, {{nz, ny, nx}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (pattern == ManufacturedPattern::SinLatitude) {

                field(k, j, i) = Kokkos::sin(geometry.latitude(j, i));
            }
            else {
                field(k, j, i) = Kokkos::sin(real(zonal_mode) * geometry.longitude(j, i));
            }
        });

    Kokkos::fence();
}

// ============================================================================
// Analytic Laplace-Beltrami eigenfunctions
// ============================================================================
//
// A = sin(phi)
//
//     nabla_h^2 A = -2/R^2 sin(phi)
//
// A = sin(m lambda)
//
//     nabla_h^2 A
//       = -m^2 / (R^2 cos^2(phi)) sin(m lambda)
//
// ============================================================================

Real
manufactured_expected(const ManufacturedPattern pattern,
    const HorizontalLocation location,
    const int j,
    const int i,
    const int h,
    const int zonal_mode = 2) {

    const Real phi = latitude_at(location, j, h);

    if (pattern == ManufacturedPattern::SinLatitude) {

        return -real(2.0) * K0 / (EARTH_RADIUS * EARTH_RADIUS) * std::sin(phi);
    }

    const Real lambda = longitude_at(location, i, h);

    const Real m = real(zonal_mode);

    const Real cos_phi = std::cos(phi);

    return -K0 * m * m * std::sin(m * lambda) / (EARTH_RADIUS * EARTH_RADIUS * cos_phi * cos_phi);
}

// ============================================================================
// Test 1: constant fields must have zero turbulent diffusion
// ============================================================================

void
test_constant_fields(TurbulenceProcess& turbulence, State& state, const Grid& grid) {

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int h = grid.get_halo_cells();

    set_constant_diffusivity(state);

    bool passed = true;

    // ------------------------------------------------------------
    // Dim == 3 fields
    // ------------------------------------------------------------

    for (const char* variable : {"th", "xi", "eta"}) {

        fill_3d(state, variable, real(7.0));

        reset_tendency_3d(state, variable);

        auto& tendency = state.get_field<3>("fe_tendency_" + std::string(variable));

        turbulence.calculate_tendencies<3>(state, variable, tendency);

        const auto host = tendency.get_host_data();

        const int k_end = std::string(variable) == "th" ? nz - h : nz - h - 1;

        for (int k = h; k < k_end; ++k) {

            for (int j = h; j < ny - h; ++j) {

                for (int i = h; i < nx - h; ++i) {

                    if (host(k, j, i) != real(0.0)) {

                        passed = false;

                        std::fprintf(stderr,
                            "constant %s tendency nonzero "
                            "at (%d,%d,%d): %.17e\n",
                            variable,
                            k,
                            j,
                            i,
                            static_cast<double>(host(k, j, i)));
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------
    // Dim == 2 top-zeta tendency
    // ------------------------------------------------------------

    fill_3d(state, "zeta", real(7.0));

    reset_tendency_2d(state, "zeta");

    auto& zeta_tendency = state.get_field<2>("fe_tendency_zeta");

    turbulence.calculate_tendencies<2>(state, "zeta", zeta_tendency);

    const auto zeta_host = zeta_tendency.get_host_data();

    for (int j = h; j < ny - h; ++j) {

        for (int i = h; i < nx - h; ++i) {

            if (zeta_host(j, i) != real(0.0)) {

                passed = false;

                std::fprintf(stderr,
                    "constant zeta tendency nonzero "
                    "at (%d,%d): %.17e\n",
                    j,
                    i,
                    static_cast<double>(zeta_host(j, i)));
            }
        }
    }

    report("RLL turbulence: constant fields -> zero", passed);

    check(passed, "constant-field turbulent diffusion");
}

// ============================================================================
// Test one 3-D manufactured solution
// ============================================================================

bool
check_manufactured_3d(TurbulenceProcess& turbulence,
    State& state,
    const Grid& grid,
    const std::string& variable,
    const HorizontalLocation location,
    const ManufacturedPattern pattern) {

    set_constant_diffusivity(state);

    set_manufactured_field(state, grid, variable, location, pattern);

    reset_tendency_3d(state, variable);

    auto& tendency = state.get_field<3>("fe_tendency_" + variable);

    turbulence.calculate_tendencies<3>(state, variable, tendency);

    const auto host = tendency.get_host_data();

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int h = grid.get_halo_cells();

    // Use a level away from both vertical boundaries.
    const int k = h + 1;

    Real max_normalized_error = real(0.0);

    const Real reference_scale = K0 / (EARTH_RADIUS * EARTH_RADIUS);

    // Stay one additional point away from physical boundaries.
    // This keeps this test about the metric diffusion stencil,
    // not about bounded-latitude boundary treatment.
    for (int j = h + 1; j < ny - h - 1; ++j) {

        for (int i = h + 1; i < nx - h - 1; ++i) {

            const Real expected = manufactured_expected(pattern, location, j, i, h);

            const Real actual = host(k, j, i);

            if (!std::isfinite(actual)) {
                return false;
            }

            max_normalized_error =
                std::max(max_normalized_error, std::abs(actual - expected) / reference_scale);
        }
    }

    // dphi and dlambda are 2.5 degrees here.
    // A second-order stencil should be comfortably below 1%.
    const Real tolerance = real(1.0e-2);

    std::printf("  %-8s %-14s max normalized error = %.17e\n",
        variable.c_str(),
        pattern == ManufacturedPattern::SinLatitude ? "sin(latitude)" : "sin(2*lon)",
        static_cast<double>(max_normalized_error));

    return max_normalized_error < tolerance;
}

// ============================================================================
// Test 2: latitude manufactured solutions
// ============================================================================

void
test_latitude_manufactured_solution(TurbulenceProcess& turbulence, State& state, const Grid& grid) {

    bool passed = true;

    passed &= check_manufactured_3d(turbulence,
        state,
        grid,
        "th",
        HorizontalLocation::T,
        ManufacturedPattern::SinLatitude);

    passed &= check_manufactured_3d(turbulence,
        state,
        grid,
        "xi",
        HorizontalLocation::V,
        ManufacturedPattern::SinLatitude);

    passed &= check_manufactured_3d(turbulence,
        state,
        grid,
        "eta",
        HorizontalLocation::U,
        ManufacturedPattern::SinLatitude);

    // ------------------------------------------------------------
    // top zeta
    // ------------------------------------------------------------

    set_constant_diffusivity(state);

    set_manufactured_field(state,
        grid,
        "zeta",
        HorizontalLocation::Z,
        ManufacturedPattern::SinLatitude);

    reset_tendency_2d(state, "zeta");

    auto& tendency = state.get_field<2>("fe_tendency_zeta");

    turbulence.calculate_tendencies<2>(state, "zeta", tendency);

    const auto host = tendency.get_host_data();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int h = grid.get_halo_cells();

    const Real reference_scale = K0 / (EARTH_RADIUS * EARTH_RADIUS);

    Real max_error = real(0.0);

    for (int j = h + 1; j < ny - h - 1; ++j) {

        for (int i = h + 1; i < nx - h - 1; ++i) {

            const Real expected = manufactured_expected(ManufacturedPattern::SinLatitude,
                HorizontalLocation::Z,
                j,
                i,
                h);

            const Real actual = host(j, i);

            if (!std::isfinite(actual)) {
                passed = false;
                continue;
            }

            max_error = std::max(max_error, std::abs(actual - expected) / reference_scale);
        }
    }

    std::printf("  %-8s %-14s max normalized error = %.17e\n",
        "zeta",
        "sin(latitude)",
        static_cast<double>(max_error));

    passed &= max_error < real(1.0e-2);

    report("RLL turbulence: latitude manufactured solution", passed);

    check(passed, "latitude manufactured turbulent diffusion");
}

// ============================================================================
// Test 3: longitude manufactured solutions
//
// This specifically tests the 1/cos(phi)^2 zonal metric.
// ============================================================================

void
test_longitude_manufactured_solution(
    TurbulenceProcess& turbulence, State& state, const Grid& grid) {

    bool passed = true;

    passed &= check_manufactured_3d(turbulence,
        state,
        grid,
        "th",
        HorizontalLocation::T,
        ManufacturedPattern::SinLongitude);

    passed &= check_manufactured_3d(turbulence,
        state,
        grid,
        "xi",
        HorizontalLocation::V,
        ManufacturedPattern::SinLongitude);

    passed &= check_manufactured_3d(turbulence,
        state,
        grid,
        "eta",
        HorizontalLocation::U,
        ManufacturedPattern::SinLongitude);

    report("RLL turbulence: longitude manufactured solution", passed);

    check(passed, "longitude manufactured turbulent diffusion");
}

// ============================================================================
// Test 4: solid-body rotation must have approximately zero deformation
//
// u = U0 cos(phi)
// v = 0
// w = 0
//
// For the spherical xy shear:
//
//   (1/R) du/dphi + u tan(phi)/R = 0
//
// continuously.
//
// This specifically validates the connection term in R_zeta.
// ============================================================================

void
test_solid_body_rotation(TurbulenceProcess& turbulence, State& state, const Grid& grid) {

    constexpr Real U0 = real(20.0);

    auto& u = state.get_field<3>("u").get_mutable_device_data();

    const auto geometry_u = grid.geometry().device_view(HorizontalLocation::U);

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int h = grid.get_halo_cells();

    Kokkos::parallel_for("SetSolidBodyRotation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{0, 0, 0}}, {{nz, ny, nx}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u(k, j, i) = U0 * Kokkos::cos(geometry_u.latitude(j, i));
        });

    fill_3d(state, "v", real(0.0));

    fill_3d(state, "w", real(0.0));

    fill_3d(state, "th", real(300.0));

    Kokkos::fence();

    turbulence.compute_coefficients(state, real(1.0));

    const auto R_xi = state.get_field<3>("R_xi").get_host_data();

    const auto R_eta = state.get_field<3>("R_eta").get_host_data();

    const auto R_zeta = state.get_field<3>("R_zeta").get_host_data();

    const auto RKM = state.get_field<3>("RKM").get_host_data();

    const auto RKH = state.get_field<3>("RKH").get_host_data();

    Real max_xi = real(0.0);

    Real max_eta = real(0.0);

    Real max_zeta_normalized = real(0.0);

    bool coefficients_ok = true;

    const Real deformation_scale = U0 / EARTH_RADIUS;

    for (int k = h; k < nz - h - 1; ++k) {

        for (int j = h + 1; j < ny - h - 1; ++j) {

            for (int i = h + 1; i < nx - h - 1; ++i) {

                max_xi = std::max(max_xi, std::abs(R_xi(k, j, i)));

                max_eta = std::max(max_eta, std::abs(R_eta(k, j, i)));

                max_zeta_normalized =
                    std::max(max_zeta_normalized, std::abs(R_zeta(k, j, i)) / deformation_scale);

                if (!std::isfinite(RKM(k, j, i)) || !std::isfinite(RKH(k, j, i)) ||
                    RKM(k, j, i) < real(0.0) || RKH(k, j, i) < real(0.0)) {

                    coefficients_ok = false;
                }
            }
        }
    }

    std::printf("  max |R_xi|                     = %.17e\n", static_cast<double>(max_xi));

    std::printf("  max |R_eta|                    = %.17e\n", static_cast<double>(max_eta));

    std::printf("  max |R_zeta| / (U0/R)          = %.17e\n",
        static_cast<double>(max_zeta_normalized));

    // R_xi and R_eta should be exactly zero for this state.
    //
    // R_zeta is not exactly zero because the d/dphi term is
    // discretized while the connection term uses tan(phi)
    // analytically. It should be second-order small.
    const bool passed = max_xi < real(1.0e-14) && max_eta < real(1.0e-14) &&
                        max_zeta_normalized < real(1.0e-3) && coefficients_ok;

    report("RLL turbulence: solid-body rotation deformation", passed);

    check(passed, "solid-body rotation deformation");
}

// ============================================================================
// Build a minimal RLL model configuration
// ============================================================================

Json
make_configuration() {

    Json config = Json::parse(R"(
{
  "grid": {
    "horizontal": {
      "nx": 24,
      "ny": 16,
      "n_halo_cells": 2,
      "geometry": {
        "kind": "regular_latlon",
        "earth_radius_m": 6371220,
        "longitude_bounds_deg": [-30, 30],
        "latitude_bounds_deg": [-20, 20]
      },
      "topology": {
        "q1": "periodic",
        "q2": "bounded"
      }
    },
    "vertical": {
      "nz": 8,
      "type": "default",
      "dz": 250,
      "dz1": 250
    }
  },

  "simulation": {
    "idealized_test": "jung2019_barotropic",
    "dt_s": 1,
    "total_time_s": 2,
    "output_interval_s": 1
  },

  "initial_conditions": {
    "jung2019": {
      "case": 1
    }
  },

  "dynamics": {
    "solver": {
      "w_solver_method": "tridiagonal",
      "iteration": 10,
      "initial_iterations": 10,
      "vertical_iterations": 10,
      "WRXMU": 100
    }
  },

  "constants": {
    "gravity": 9.806,
    "Rd": 287.04,
    "Cp": 1004.5,
    "P0": 100000
  },

  "physics": {},

  "output": {
    "engine": "HDF5"
  }
}
)");

    // State expects the prognostic-variable definitions.
    for (const auto* variable : {"xi", "eta", "zeta"}) {

        for (const auto* term : {"advection", "stretching", "twisting"}) {

            config["dynamics"]["prognostic_variables"][variable]["tendency_terms"][term] = {
                {"enable", true},
                {"spatial_scheme", "Takacs"},
                {"temporal_scheme", "AdamsBashforth2"}};
        }
    }

    return config;
}

} // namespace

// ============================================================================
// main
// ============================================================================

int
main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));

    const auto directory = std::filesystem::temp_directory_path() /
                           ("vvm_regular_latlon_turbulence_" + std::to_string(getpid()));

    std::filesystem::create_directories(directory);

    int result = 0;

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm = nullptr;
#endif

    try {
        // ------------------------------------------------------------
        // Configuration
        // ------------------------------------------------------------

        const auto json = make_configuration();

        const auto config_path = directory / "regular_latlon_turbulence.json";

        {
            std::ofstream file(config_path);

            file << json.dump(2);
        }

        ConfigurationManager config(config_path.string());

        // ------------------------------------------------------------
        // Core model objects
        // ------------------------------------------------------------

        Grid grid(config, MPI_COMM_WORLD);

        Parameters params(config, grid);

#if defined(ENABLE_NCCL)

        ncclUniqueId nccl_id;

        if (ncclGetUniqueId(&nccl_id) != ncclSuccess) {
            throw std::runtime_error("ncclGetUniqueId failed");
        }

        if (ncclCommInitRank(&nccl_comm, 1, nccl_id, 0) != ncclSuccess) {
            throw std::runtime_error("ncclCommInitRank failed");
        }

        const auto stream = Kokkos::Cuda().cuda_stream();

        State state(config, params, grid, nccl_comm, stream);

        VVM::Core::HaloExchanger halo(config, grid, nccl_comm, stream);

#else

        State state(config, params, grid);

        VVM::Core::HaloExchanger halo(grid);

#endif

        // ------------------------------------------------------------
        // Controlled turbulence fixture
        // ------------------------------------------------------------

        initialize_base_state(state, params, grid);

        TurbulenceProcess turbulence(config, grid, params, halo, state);

        turbulence.initialize(state);

        // ------------------------------------------------------------
        // Tests
        // ------------------------------------------------------------

        test_constant_fields(turbulence, state, grid);

        test_latitude_manufactured_solution(turbulence, state, grid);

        test_longitude_manufactured_solution(turbulence, state, grid);

        test_solid_body_rotation(turbulence, state, grid);

        // ------------------------------------------------------------
        // Summary
        // ------------------------------------------------------------

        if (failures == 0) {
            std::printf("\nPASS: regular latitude-longitude turbulence\n");
        }
        else {
            std::fprintf(stderr,
                "\nFAIL: %d regular latitude-longitude "
                "turbulence check(s) failed\n",
                failures);

            result = 1;
        }
    }
    catch (const std::exception& error) {

        std::fprintf(stderr, "Unexpected exception: %s\n", error.what());

        result = 1;
    }

#if defined(ENABLE_NCCL)

    if (nccl_comm != nullptr) {
        ncclCommDestroy(nccl_comm);
    }

#endif

    std::filesystem::remove_all(directory);

    Kokkos::finalize();

    MPI_Finalize();

    return result;
}
