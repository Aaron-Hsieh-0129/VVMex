#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "physics/surface/SurfaceProcess.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Grid;
using VVM::Core::Parameters;
using VVM::Core::State;
using VVM::Core::HaloExchanger;
using VVM::Physics::SurfaceProcess;
using VVM::Utils::ConfigurationManager;

int failures = 0;

constexpr std::array<const char*, 5> output_names = {
    "VEN2D", "sfc_flux_th", "sfc_flux_qv", "sfc_flux_u", "sfc_flux_v"
};

using Snapshot = std::array<std::vector<Real>, output_names.size()>;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

void check_near(Real actual, Real expected, const std::string& message) {
    const Real scale = std::max(std::abs(expected), std::numeric_limits<Real>::min());
    const Real tolerance = real(128.0) * std::numeric_limits<Real>::epsilon() * scale;
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const std::string pattern = (std::filesystem::temp_directory_path() / "vvm_surface_vertical_XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');

        const char* directory = ::mkdtemp(buffer.data());
        if (!directory) throw std::runtime_error("Cannot create temporary test directory.");
        path_ = directory;
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

struct Communication {
#if defined(ENABLE_NCCL)
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;

    Communication() {
        ncclUniqueId id;
        require(ncclGetUniqueId(&id), "ncclGetUniqueId");
        require(ncclCommInitRank(&comm, 1, id, 0), "ncclCommInitRank");
        stream = Kokkos::Cuda().cuda_stream();
    }

    ~Communication() {
        if (comm) ncclCommDestroy(comm);
    }

    static void require(ncclResult_t status, const char* operation) {
        if (status != ncclSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
        }
    }
#endif
};

void write_json(const std::filesystem::path& path, const nlohmann::json& value) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot create " + path.string());
    output << value.dump(4) << '\n';
    output.close();
    if (!output) throw std::runtime_error("Cannot write " + path.string());
}

nlohmann::json make_configuration(const std::string& type, int mode, const std::string& rcemip_path) {
    nlohmann::json config;

    // Keep horizontal configuration identical to isolate vertical-type selection.
    config["grid"] = {
        {"nx", 4},
        {"ny", 4},
        {"n_halo_cells", 2},
        {"dx", 100.0},
        {"dy", 100.0},
        {"boundary_condition", {{"x", "periodic"}, {"y", "periodic"}}}
    };

    const double dz1 = type == "taiwanvvm" ? 50.0 : 100.0;

    if (mode == 0) {
        config["grid"]["nz"] = 4;
        config["grid"]["dz"] = 100.0;
        config["grid"]["dz1"] = dz1;
        config["grid"]["vertical_coordinate_type"] = type;
        config["grid"]["rcemip_grid_data_path"] = rcemip_path;
    } else {
        config["grid"]["vertical"] = {
            {"nz", 4},
            {"type", type},
            {"dz", 100.0},
            {"dz1", dz1},
            {"rcemip_grid_data_path", rcemip_path}
        };

        if (mode == 2) {
            config["grid"]["nz"] = 8;
            config["grid"]["dz"] = 200.0;
            config["grid"]["dz1"] = 100.0;
            config["grid"]["vertical_coordinate_type"] = type == "rcemip" ? "default" : "rcemip";
            config["grid"]["rcemip_grid_data_path"] = rcemip_path;
        }
    }

    config["constants"] = {
        {"gravity", 9.81},
        {"Rd", 287.0},
        {"P0", 100000.0},
        {"Cp", 1004.0},
        {"Lv", 2500000.0}
    };

    config["simulation"] = {
        {"dt_s", 1.0},
        {"idealized_test", "2dbubble"}
    };

    config["dynamics"]["solver"] = {
        {"WRXMU", 0.25},
        {"iteration", 4}
    };

    config["physics"]["surface_process"] = {
        {"ocean_scheme", "sflux_2d"},
        {"land_scheme", "none"}
    };

    return config;
}

template<std::size_t Dim>
void fill(State& state, const char* name, Real value) {
    Kokkos::deep_copy(state.get_field<Dim>(name).get_mutable_device_data(), value);
}

Snapshot run_case(const std::filesystem::path& config_path, const std::string& type, Real wind, const Communication& communication) {
    const ConfigurationManager config(config_path.string());
    const Grid grid(config);
    Parameters parameters(config, grid);

#if defined(ENABLE_NCCL)
    State state(config, parameters, grid, communication.comm, communication.stream);
    HaloExchanger halo(config, grid, communication.comm, communication.stream);
#else
    (void)communication;
    State state(config, parameters, grid);
    HaloExchanger halo(grid);
#endif

    SurfaceProcess surface(config, grid, parameters, halo, state);
    surface.initialize(state);

    const int h = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    // Prescribed component-test geometry: first-level height 50 m, flat surface.
    parameters.max_topo_idx = h - 1;
    Kokkos::deep_copy(parameters.z_mid.get_mutable_device_data(), real(50.0));
    Kokkos::deep_copy(parameters.z_up.get_mutable_device_data(), real(0.0));
    Kokkos::deep_copy(parameters.flex_height_coef_mid.get_mutable_device_data(), real(1.0));

    for (const char* name : {"topo", "topou", "topov"}) {
        state.add_field<2>(name, {ny, nx});
        fill<2>(state, name, static_cast<Real>(h - 1));
    }

    fill<1>(state, "pbar", real(100000.0));
    fill<1>(state, "pibar", real(1.0));
    fill<1>(state, "rhobar", real(1.0));
    fill<1>(state, "rhobar_up", real(1.0));
    fill<1>(state, "thbar", real(300.0));

    fill<3>(state, "u", wind);
    fill<3>(state, "v", real(0.0));
    fill<3>(state, "th", real(300.0));
    fill<3>(state, "qv", real(0.0));
    fill<3>(state, "qc", real(0.0));
    fill<3>(state, "qi", real(0.0));

    fill<2>(state, "Tg", real(300.0));
    fill<2>(state, "gwet", real(0.0));
    fill<2>(state, "zrough", real(2e-4));
    fill<2>(state, "sea_land_ice_mask", real(0.0));

    // Equal skin/air potential temperatures and zero wetness give thvsm = 0.
    surface.compute_coefficients(state);
    Kokkos::fence();

    Snapshot snapshot;
    for (std::size_t field = 0; field < output_names.size(); ++field) {
        const auto host = state.get_field<2>(output_names[field]).get_host_data();

        for (int j = h; j < ny - h; ++j) {
            for (int i = h; i < nx - h; ++i) {
                check(std::isfinite(host(j, i)), std::string(output_names[field]) + " must be finite.");
                snapshot[field].push_back(host(j, i));
            }
        }
    }

    const Real floor = type == "rcemip" ? real(1.0) : real(1e-3);
    const Real cu = real(0.4) / std::log(real(50.0) / real(2e-4));
    const Real expected_ven = cu * cu * std::max(wind, floor);

    for (std::size_t point = 0; point < snapshot[0].size(); ++point) {
        check_near(snapshot[0][point], expected_ven, type + ": neutral VEN2D");
        check_near(snapshot[3][point], -expected_ven * wind, type + ": zonal momentum flux");
        check(snapshot[2][point] == real(0.0), type + ": zero-wetness moisture flux");
        check(snapshot[4][point] == real(0.0), type + ": zero meridional momentum flux");
    }

    return snapshot;
}

void compare(const Snapshot& actual, const Snapshot& expected, const std::string& label) {
    for (std::size_t field = 0; field < output_names.size(); ++field) {
        check(actual[field] == expected[field], label + ": identical " + output_names[field]);
    }
}

void run_tests(const Communication& communication) {
    TemporaryDirectory temporary;
    const auto grid_path = temporary.path() / "rcemip_grid.txt";

    {
        std::ofstream output(grid_path);
        output << "zz zt\n100 50\n200 150\n300 250\n400 350\n";
        output.close();
        if (!output) throw std::runtime_error("Cannot write temporary RCEMIP grid.");
    }

    for (const std::string type : {"default", "taiwanvvm", "rcemip"}) {
        std::array<std::filesystem::path, 3> paths;

        for (int mode = 0; mode < 3; ++mode) {
            paths[mode] = temporary.path() / (type + "_" + std::to_string(mode) + ".json");
            write_json(paths[mode], make_configuration(type, mode, grid_path.string()));
        }

        for (Real wind : {real(0.1), real(2.0)}) {
            const Snapshot legacy = run_case(paths[0], type, wind, communication);
            const Snapshot structured = run_case(paths[1], type, wind, communication);
            const Snapshot mixed = run_case(paths[2], type, wind, communication);

            compare(structured, legacy, type + " structured");
            compare(mixed, legacy, type + " conflicting legacy");

            std::printf("Checked %s wind=%.1f: legacy, structured, conflicting legacy\n",
                type.c_str(), static_cast<double>(wind));
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    try {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size != 1) throw std::runtime_error("This test requires one MPI rank.");

        Communication communication;
        run_tests(communication);
        Kokkos::fence();
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_surface_vertical_configuration: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) {
        std::puts("test_surface_vertical_configuration: PASS");
    }

    Kokkos::finalize();
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
