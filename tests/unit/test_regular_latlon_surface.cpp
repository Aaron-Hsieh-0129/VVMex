#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "physics/surface/SurfaceProcess.hpp"
#include "utils/ConfigurationManager.hpp"

#include "../../externals/json/json.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

#if defined(ENABLE_NCCL)
#include <nccl.h>
#endif

namespace {

using VVM::Real;
using VVM::real;

using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Core::Parameters;
using VVM::Core::State;
using VVM::Physics::SurfaceProcess;
using VVM::Utils::ConfigurationManager;

using Json = nlohmann::json;

// ============================================================================
// Test constants
// ============================================================================

constexpr Real DZ = real(100.0);

constexpr Real WIND_SPEED = real(10.0);

constexpr Real THETA = real(300.0);
constexpr Real TG = real(300.0);

constexpr Real PRESSURE = real(100000.0);
constexpr Real RHO = real(1.0);

constexpr Real ZROUGH = real(2.0e-4);
constexpr Real VK = 0.4;

int failures = 0;

// ============================================================================
// Utilities
// ============================================================================

void
check(const bool condition, const std::string& message) {

    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

void
check_near(const Real actual,
    const Real expected,
    const std::string& message,
    const Real relative_tolerance = real(1.0e-11),
    const Real absolute_tolerance = real(1.0e-14)) {

    const Real tolerance =
        absolute_tolerance + relative_tolerance * std::max(std::abs(actual), std::abs(expected));

    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {

        ++failures;

        std::fprintf(stderr,
            "FAIL: %s\n"
            "      actual   = %.17e\n"
            "      expected = %.17e\n"
            "      error    = %.17e\n"
            "      tolerance= %.17e\n",
            message.c_str(),
            static_cast<double>(actual),
            static_cast<double>(expected),
            static_cast<double>(std::abs(actual - expected)),
            static_cast<double>(tolerance));
    }
}

void
report(const std::string& name, const int failures_before) {

    std::printf("%-58s %s\n", name.c_str(), failures == failures_before ? "PASS" : "FAIL");
}

// ============================================================================
// Temporary directory
// ============================================================================

class TemporaryDirectory {
public:
    TemporaryDirectory() {

        const std::string pattern =
            (std::filesystem::temp_directory_path() / "vvm_regular_latlon_surface_XXXXXX").string();

        std::vector<char> buffer(pattern.begin(), pattern.end());

        buffer.push_back('\0');

        const char* directory = ::mkdtemp(buffer.data());

        if (!directory) {
            throw std::runtime_error("Cannot create temporary test directory.");
        }

        path_ = directory;
    }

    ~TemporaryDirectory() {

        std::error_code error;

        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;

    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path&
    path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

// ============================================================================
// Generic field helpers
// ============================================================================

template <std::size_t Dim>
void
fill(State& state, const std::string& name, const Real value) {

    Kokkos::deep_copy(state.get_field<Dim>(name).get_mutable_device_data(), value);
}

void
ensure_2d_field(State& state, const std::string& name, const int ny, const int nx) {

    if (!state.has_field(name)) {
        state.add_field<2>(name, {ny, nx});
    }
}

void
ensure_3d_field(State& state, const std::string& name, const int nz, const int ny, const int nx) {

    if (!state.has_field(name)) {
        state.add_field<3>(name, {nz, ny, nx});
    }
}

// ============================================================================
// JSON
// ============================================================================

void
write_json(const std::filesystem::path& path, const Json& value) {

    std::ofstream output(path);

    if (!output) {
        throw std::runtime_error("Cannot create " + path.string());
    }

    output << value.dump(2) << '\n';

    output.close();

    if (!output) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

// ============================================================================
// Minimal regular latitude-longitude configuration
// ============================================================================

Json
make_configuration() {

    Json config = Json::parse(R"(
{
  "grid": {
    "horizontal": {
      "nx": 16,
      "ny": 12,
      "n_halo_cells": 2,

      "geometry": {
        "kind": "regular_latlon",
        "earth_radius_m": 6371220.0,
        "longitude_bounds_deg": [-30.0, 30.0],
        "latitude_bounds_deg": [-30.0, 30.0]
      },

      "topology": {
        "q1": "periodic",
        "q2": "bounded"
      }
    },

    "vertical": {
      "nz": 6,
      "type": "default",
      "dz": 100.0,
      "dz1": 100.0
    }
  },

  "simulation": {
    "idealized_test": "jung2019_barotropic",
    "dt_s": 1.0,
    "total_time_s": 1.0,
    "output_interval_s": 1.0
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
      "WRXMU": 100.0
    }
  },

  "constants": {
    "gravity": 9.806,
    "Rd": 287.04,
    "Cp": 1004.5,
    "P0": 100000.0,
    "Lv": 2500000.0
  },

  "physics": {
    "surface_process": {
      "ocean_scheme": "sflux_2d",
      "land_scheme": "none"
    }
  },

  "output": {
    "engine": "HDF5"
  }
}
)");

    // State expects the dynamical prognostic-variable configuration.
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

// ============================================================================
// Controlled flat-ocean state
// ============================================================================
//
// Important:
//
// topo = h - 1
//
// SurfaceProcess uses:
//
//     hxp = topo + 1
//
// therefore:
//
//     hxp = h
//
// which is the first physical atmospheric level.
//
// ============================================================================

void
initialize_surface_fixture(
    State& state, Parameters& params, const Grid& grid, SurfaceProcess& surface) {

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int h = grid.get_halo_cells();

    // SurfaceProcess creates qc, qi, flux fields, gwet,
    // zrough, VEN2D, ustar, molen, sea_land_ice_mask.
    surface.initialize(state);

    // ------------------------------------------------------------
    // Terrain-derived fields.
    // Normally created by initialize_topo().
    // This is a component test, so construct the flat state directly.
    // ------------------------------------------------------------

    ensure_2d_field(state, "topo", ny, nx);

    ensure_2d_field(state, "topou", ny, nx);

    ensure_2d_field(state, "topov", ny, nx);

    const Real flat_topo = static_cast<Real>(h - 1);

    fill<2>(state, "topo", flat_topo);

    fill<2>(state, "topou", flat_topo);

    fill<2>(state, "topov", flat_topo);

    // max_topo_idx <= h makes SurfaceProcess use the
    // flat-terrain momentum-flux branch.
    params.max_topo_idx = h - 1;

    // ------------------------------------------------------------
    // Simple vertical geometry.
    // ------------------------------------------------------------

    Kokkos::deep_copy(params.flex_height_coef_mid.get_mutable_device_data(), real(1.0));

    Kokkos::deep_copy(params.flex_height_coef_up.get_mutable_device_data(), real(1.0));

    Kokkos::deep_copy(params.z_mid.get_mutable_device_data(), real(50.0));

    Kokkos::deep_copy(params.z_up.get_mutable_device_data(), real(0.0));

    // ------------------------------------------------------------
    // Reference atmosphere.
    // ------------------------------------------------------------

    fill<1>(state, "pbar", PRESSURE);

    fill<1>(state, "pibar", real(1.0));

    fill<1>(state, "rhobar", RHO);

    fill<1>(state, "rhobar_up", RHO);

    fill<1>(state, "thbar", THETA);

    // ------------------------------------------------------------
    // Atmospheric fields.
    // ------------------------------------------------------------

    fill<3>(state, "u", real(0.0));

    fill<3>(state, "v", real(0.0));

    fill<3>(state, "th", THETA);

    fill<3>(state, "qv", real(0.0));

    fill<3>(state, "qc", real(0.0));

    fill<3>(state, "qi", real(0.0));

    // ------------------------------------------------------------
    // Flat ocean surface.
    //
    // Tg = theta and gwet = 0 gives thvsm = 0:
    // a neutral Monin-Obukhov case.
    // ------------------------------------------------------------

    fill<2>(state, "Tg", TG);

    fill<2>(state, "gwet", real(0.0));

    fill<2>(state, "zrough", ZROUGH);

    fill<2>(state, "sea_land_ice_mask", real(0.0));

    // ------------------------------------------------------------
    // Clear output fields before every calculation.
    // ------------------------------------------------------------

    fill<2>(state, "VEN2D", real(0.0));

    fill<2>(state, "sfc_flux_th", real(0.0));

    fill<2>(state, "sfc_flux_qv", real(0.0));

    fill<2>(state, "sfc_flux_u", real(0.0));

    fill<2>(state, "sfc_flux_v", real(0.0));

    // ------------------------------------------------------------
    // Test-only tendency arrays.
    // ------------------------------------------------------------

    ensure_3d_field(state, "surface_test_tendency_th", nz, ny, nx);

    ensure_3d_field(state, "surface_test_tendency_qv", nz, ny, nx);

    ensure_3d_field(state, "surface_test_tendency_xi", nz, ny, nx);

    ensure_3d_field(state, "surface_test_tendency_eta", nz, ny, nx);

    Kokkos::fence();
}

// ============================================================================
// Snapshot
// ============================================================================

struct SurfaceSnapshot {
    std::vector<Real> ven;
    std::vector<Real> flux_th;
    std::vector<Real> flux_qv;
    std::vector<Real> flux_u;
    std::vector<Real> flux_v;
};

SurfaceSnapshot
run_wind_case(SurfaceProcess& surface,
    State& state,
    const Grid& grid,
    const Real u_value,
    const Real v_value) {

    fill<3>(state, "u", u_value);

    fill<3>(state, "v", v_value);

    // Reinitialize output fields so this case does not
    // depend on results from a previous case.
    fill<2>(state, "VEN2D", real(0.0));

    fill<2>(state, "sfc_flux_th", real(0.0));

    fill<2>(state, "sfc_flux_qv", real(0.0));

    fill<2>(state, "sfc_flux_u", real(0.0));

    fill<2>(state, "sfc_flux_v", real(0.0));

    Kokkos::fence();

    surface.compute_coefficients(state);

    Kokkos::fence();

    const auto ven = state.get_field<2>("VEN2D").get_host_data();

    const auto flux_th = state.get_field<2>("sfc_flux_th").get_host_data();

    const auto flux_qv = state.get_field<2>("sfc_flux_qv").get_host_data();

    const auto flux_u = state.get_field<2>("sfc_flux_u").get_host_data();

    const auto flux_v = state.get_field<2>("sfc_flux_v").get_host_data();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int h = grid.get_halo_cells();

    SurfaceSnapshot snapshot;

    for (int j = h + 1; j < ny - h - 1; ++j) {

        for (int i = h; i < nx - h; ++i) {

            snapshot.ven.push_back(ven(j, i));

            snapshot.flux_th.push_back(flux_th(j, i));

            snapshot.flux_qv.push_back(flux_qv(j, i));

            snapshot.flux_u.push_back(flux_u(j, i));

            snapshot.flux_v.push_back(flux_v(j, i));
        }
    }

    return snapshot;
}

// ============================================================================
// Exact neutral surface-layer result
// ============================================================================
//
// For thvsm = 0:
//
//     Cu = kappa / ln(z / z0)
//
//     VEN2D = Cu^2 |V|
//
// The flat-domain momentum flux is:
//
//     tau_u = -VEN2D u
//     tau_v = -VEN2D v
//
// because rhobar_up = 1.
//
// ============================================================================

Real
expected_ven(const Real u, const Real v) {

    const Real speed = std::sqrt(u * u + v * v);

    const Real zr = real(0.5) * DZ;

    const Real cu = VK / std::log(zr / ZROUGH);

    return cu * cu * speed;
}

// ============================================================================
// Test 1: neutral zonal wind
// ============================================================================

void
test_zonal_surface_flux(SurfaceProcess& surface, State& state, const Grid& grid) {

    const int before = failures;

    const auto result = run_wind_case(surface, state, grid, WIND_SPEED, real(0.0));

    const Real ven_expected = expected_ven(WIND_SPEED, real(0.0));

    const Real flux_u_expected = -ven_expected * WIND_SPEED;

    for (std::size_t n = 0; n < result.ven.size(); ++n) {

        check(std::isfinite(result.ven[n]), "zonal: VEN2D must be finite");

        check(std::isfinite(result.flux_th[n]), "zonal: sfc_flux_th must be finite");

        check(std::isfinite(result.flux_qv[n]), "zonal: sfc_flux_qv must be finite");

        check(std::isfinite(result.flux_u[n]), "zonal: sfc_flux_u must be finite");

        check(std::isfinite(result.flux_v[n]), "zonal: sfc_flux_v must be finite");

        check(result.ven[n] > real(0.0), "zonal: VEN2D must be positive");

        check_near(result.ven[n], ven_expected, "zonal: neutral VEN2D");

        check_near(result.flux_u[n], flux_u_expected, "zonal: momentum drag");

        check_near(result.flux_v[n], real(0.0), "zonal: meridional momentum flux must vanish");

        check_near(result.flux_qv[n], real(0.0), "zonal: zero-wetness moisture flux must vanish");
    }

    report("RLL surface: neutral zonal momentum flux", before);
}

// ============================================================================
// Test 2: neutral meridional wind
// ============================================================================

void
test_meridional_surface_flux(SurfaceProcess& surface, State& state, const Grid& grid) {

    const int before = failures;

    const auto result = run_wind_case(surface, state, grid, real(0.0), WIND_SPEED);

    const Real ven_expected = expected_ven(real(0.0), WIND_SPEED);

    const Real flux_v_expected = -ven_expected * WIND_SPEED;

    for (std::size_t n = 0; n < result.ven.size(); ++n) {

        check_near(result.ven[n], ven_expected, "meridional: neutral VEN2D");

        check_near(result.flux_u[n], real(0.0), "meridional: zonal momentum flux must vanish");

        check_near(result.flux_v[n], flux_v_expected, "meridional: momentum drag");

        check_near(result.flux_qv[n],
            real(0.0),
            "meridional: zero-wetness moisture flux must vanish");
    }

    report("RLL surface: neutral meridional momentum flux", before);
}

// ============================================================================
// Test 3: rotational symmetry
//
// u and v are physical eastward/northward velocities in the RLL model.
// Therefore rotating a uniform physical wind by 90 degrees must not change
// VEN2D or the scalar surface fluxes.
// ============================================================================

void
test_horizontal_rotational_symmetry(SurfaceProcess& surface, State& state, const Grid& grid) {

    const int before = failures;

    const auto zonal = run_wind_case(surface, state, grid, WIND_SPEED, real(0.0));

    const auto meridional = run_wind_case(surface, state, grid, real(0.0), WIND_SPEED);

    check(zonal.ven.size() == meridional.ven.size(), "rotational symmetry: snapshot sizes");

    const std::size_t count = std::min(zonal.ven.size(), meridional.ven.size());

    for (std::size_t n = 0; n < count; ++n) {

        check_near(zonal.ven[n], meridional.ven[n], "rotational symmetry: VEN2D");

        check_near(zonal.flux_th[n], meridional.flux_th[n], "rotational symmetry: heat flux");

        check_near(zonal.flux_qv[n], meridional.flux_qv[n], "rotational symmetry: moisture flux");

        // Rotating u -> v should rotate the drag vector.
        check_near(zonal.flux_u[n],
            meridional.flux_v[n],
            "rotational symmetry: momentum flux magnitude");

        check_near(zonal.flux_v[n],
            meridional.flux_u[n],
            "rotational symmetry: zero cross component");
    }

    report("RLL surface: physical-wind rotational symmetry", before);
}

// ============================================================================
// Test 4: no direct latitude dependence
//
// SurfaceProcess uses physical u/v, not coordinate velocities.
//
// With identical physical atmospheric and surface states at all latitudes,
// the local surface exchange must be identical at every j.
//
// This catches accidental insertion of R cos(phi), cos(phi), etc.
// ============================================================================

void
test_latitude_independence(SurfaceProcess& surface, State& state, const Grid& grid) {

    const int before = failures;

    fill<3>(state, "u", WIND_SPEED);

    fill<3>(state, "v", real(0.0));

    fill<2>(state, "VEN2D", real(0.0));

    fill<2>(state, "sfc_flux_th", real(0.0));

    fill<2>(state, "sfc_flux_qv", real(0.0));

    fill<2>(state, "sfc_flux_u", real(0.0));

    fill<2>(state, "sfc_flux_v", real(0.0));

    surface.compute_coefficients(state);

    Kokkos::fence();

    const auto ven = state.get_field<2>("VEN2D").get_host_data();

    const auto flux_u = state.get_field<2>("sfc_flux_u").get_host_data();

    const auto flux_th = state.get_field<2>("sfc_flux_th").get_host_data();

    const auto flux_qv = state.get_field<2>("sfc_flux_qv").get_host_data();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int h = grid.get_halo_cells();

    // Use one interior point as the reference.
    const int reference_j = h;
    const int reference_i = h + 1;

    const Real reference_ven = ven(reference_j, reference_i);

    const Real reference_flux_u = flux_u(reference_j, reference_i);

    const Real reference_flux_th = flux_th(reference_j, reference_i);

    const Real reference_flux_qv = flux_qv(reference_j, reference_i);

    for (int j = h; j < ny - h; ++j) {

        // Stay away from the zonal halo interface when comparing.
        for (int i = h + 1; i < nx - h - 1; ++i) {

            check_near(ven(j, i), reference_ven, "latitude independence: VEN2D");

            check_near(flux_u(j, i),
                reference_flux_u,
                "latitude independence: zonal momentum flux");

            check_near(flux_th(j, i), reference_flux_th, "latitude independence: heat flux");

            check_near(flux_qv(j, i), reference_flux_qv, "latitude independence: moisture flux");
        }
    }

    report("RLL surface: no artificial latitude dependence", before);
}

// ============================================================================
// Test 5: tendency coupling
//
// For a flat surface:
//
//     topo = topou = topov = h - 1
//
// therefore all tendencies are deposited at:
//
//     target_k = h
//
// SurfaceProcess currently maps:
//
//     sfc_flux_th -> th
//     sfc_flux_qv -> qv
//     sfc_flux_v  -> xi
//     sfc_flux_u  -> eta
//
// ============================================================================

void
test_surface_tendency_coupling(SurfaceProcess& surface, State& state, const Grid& grid) {

    const int before = failures;

    // Use zonal wind.
    //
    // Expected:
    //
    //     sfc_flux_u < 0
    //     sfc_flux_v = 0
    //
    // therefore:
    //
    //     eta tendency < 0
    //     xi tendency  = 0
    //
    run_wind_case(surface, state, grid, WIND_SPEED, real(0.0));

    fill<3>(state, "surface_test_tendency_th", real(0.0));

    fill<3>(state, "surface_test_tendency_qv", real(0.0));

    fill<3>(state, "surface_test_tendency_xi", real(0.0));

    fill<3>(state, "surface_test_tendency_eta", real(0.0));

    auto& tendency_th = state.get_field<3>("surface_test_tendency_th");

    auto& tendency_qv = state.get_field<3>("surface_test_tendency_qv");

    auto& tendency_xi = state.get_field<3>("surface_test_tendency_xi");

    auto& tendency_eta = state.get_field<3>("surface_test_tendency_eta");

    surface.calculate_tendencies<3>(state, "th", tendency_th);

    surface.calculate_tendencies<3>(state, "qv", tendency_qv);

    surface.calculate_tendencies<3>(state, "xi", tendency_xi);

    surface.calculate_tendencies<3>(state, "eta", tendency_eta);

    Kokkos::fence();

    const auto flux_th = state.get_field<2>("sfc_flux_th").get_host_data();

    const auto flux_qv = state.get_field<2>("sfc_flux_qv").get_host_data();

    const auto flux_u = state.get_field<2>("sfc_flux_u").get_host_data();

    const auto flux_v = state.get_field<2>("sfc_flux_v").get_host_data();

    const auto th_host = tendency_th.get_host_data();

    const auto qv_host = tendency_qv.get_host_data();

    const auto xi_host = tendency_xi.get_host_data();

    const auto eta_host = tendency_eta.get_host_data();

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int h = grid.get_halo_cells();

    const Real rdz = real(1.0) / DZ;

    const Real rdz2 = real(1.0) / (DZ * DZ);

    for (int j = h; j < ny - h; ++j) {

        for (int i = h; i < nx - h; ++i) {

            // --------------------------------------------------------
            // First physical level
            // --------------------------------------------------------

            check_near(th_host(h, j, i), flux_th(j, i) * rdz / RHO, "surface tendency: th");

            check_near(qv_host(h, j, i), flux_qv(j, i) * rdz / RHO, "surface tendency: qv");

            check_near(xi_host(h, j, i),
                flux_v(j, i) * rdz2 / RHO,
                "surface tendency: xi <- v flux");

            check_near(eta_host(h, j, i),
                flux_u(j, i) * rdz2 / RHO,
                "surface tendency: eta <- u flux");

            check(eta_host(h, j, i) < real(0.0),
                "surface tendency: eastward wind must produce "
                "negative eta surface tendency");

            check_near(xi_host(h, j, i),
                real(0.0),
                "surface tendency: zero meridional wind -> zero xi");

            // --------------------------------------------------------
            // Surface forcing should not modify any other k.
            // --------------------------------------------------------

            for (int k = 0; k < nz; ++k) {

                if (k == h) {
                    continue;
                }

                check_near(th_host(k, j, i),
                    real(0.0),
                    "surface tendency: th only at surface level");

                check_near(qv_host(k, j, i),
                    real(0.0),
                    "surface tendency: qv only at surface level");

                check_near(xi_host(k, j, i),
                    real(0.0),
                    "surface tendency: xi only at surface level");

                check_near(eta_host(k, j, i),
                    real(0.0),
                    "surface tendency: eta only at surface level");
            }
        }
    }

    report("RLL surface: th/qv/xi/eta tendency coupling", before);
}

// ============================================================================
// Main test
// ============================================================================

} // namespace

int
main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));

    int result = 0;

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm = nullptr;
#endif

    try {

        int mpi_size = 0;

        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        if (mpi_size != 1) {
            throw std::runtime_error("test_regular_latlon_surface "
                                     "requires exactly one MPI rank.");
        }

        TemporaryDirectory temporary;

        const auto config_path = temporary.path() / "regular_latlon_surface.json";

        write_json(config_path, make_configuration());

        ConfigurationManager config(config_path.string());

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

        HaloExchanger halo(config, grid, nccl_comm, stream);

#else

        State state(config, params, grid);

        HaloExchanger halo(grid);

#endif

        SurfaceProcess surface(config, grid, params, halo, state);

        initialize_surface_fixture(state, params, grid, surface);

        // ------------------------------------------------------------
        // Execute independent checks.
        // ------------------------------------------------------------

        test_zonal_surface_flux(surface, state, grid);

        test_meridional_surface_flux(surface, state, grid);

        test_horizontal_rotational_symmetry(surface, state, grid);

        test_latitude_independence(surface, state, grid);

        test_surface_tendency_coupling(surface, state, grid);

        Kokkos::fence();

        if (failures == 0) {

            std::puts("\nPASS: regular latitude-longitude surface physics");
        }
        else {

            std::fprintf(stderr,
                "\nFAIL: %d regular latitude-longitude "
                "surface check(s) failed\n",
                failures);

            result = 1;
        }
    }
    catch (const std::exception& error) {

        std::fprintf(stderr, "test_regular_latlon_surface: %s\n", error.what());

        result = 1;
    }

#if defined(ENABLE_NCCL)

    if (nccl_comm != nullptr) {
        ncclCommDestroy(nccl_comm);
    }

#endif

    Kokkos::finalize();

    MPI_Finalize();

    return result;
}
