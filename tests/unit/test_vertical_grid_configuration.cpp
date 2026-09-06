#include "core/Grid.hpp"
#include "core/Initializer.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "io/TxtReader.hpp"
#include "utils/ConfigurationManager.hpp"

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
#include <vector>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

using VVM::Real;
using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Core::Initializer;
using VVM::Core::Parameters;
using VVM::Core::State;
using VVM::Utils::ConfigurationManager;
using Json = nlohmann::json;

const std::array<const char*, 8> vertical_names = {
    "z_mid", "z_up",
    "flex_height_coef_mid", "flex_height_coef_up",
    "dz_mid", "dz_up",
    "fact1_xi_eta", "fact2_xi_eta"
};

const std::array<const char*, 13> profile_names = {
    "Tbar", "qvbar", "pbar", "pibar", "pibar_up",
    "thbar", "Tvbar", "rhobar", "rhobar_up",
    "U", "V", "Q1", "Q2"
};

struct Snapshot {
    std::array<std::vector<Real>, 8> vertical;
    std::array<std::vector<Real>, 13> profiles;
};

int failures = 0;

void check(const bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

void check_close(const Real actual, const Real expected, const std::string& message) {
    const Real scale = std::max(std::abs(expected), std::numeric_limits<Real>::min());
    const Real tolerance = Real(64) * std::numeric_limits<Real>::epsilon() * scale;

    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const std::string pattern = (std::filesystem::temp_directory_path() / "vvm_vertical_grid_XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');

        const char* created = ::mkdtemp(buffer.data());
        if (!created) {
            throw std::runtime_error("Cannot create temporary vertical-grid test directory.");
        }

        path_ = created;
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create test file: " + path.string());
    }

    output << text;
    output.close();

    if (!output) {
        throw std::runtime_error("Cannot finish writing test file: " + path.string());
    }
}

Json make_configuration(const std::string& type, const double dz1, const bool structured,
    const std::string& grid_data_path, const bool consistent_reference_state) {

    const int nz = type == "rcemip" ? 4 : 32;

    Json config = {
        {"constants", {
            {"gravity", 9.81},
            {"Rd", 287.0},
            {"P0", 100000.0},
            {"Cp", 1004.0},
            {"Lv", 2500000.0}
        }},
        {"simulation", {
            {"dt_s", 2.0},
            {"idealized_test", "2dbubble"}
        }},
        {"dynamics", {
            {"solver", {
                {"WRXMU", 0.25},
                {"iteration", 4}
            }}
        }}
    };

    // Leave the option absent for the compatibility case, checking that
    // its existing default remains unchanged.
    if (consistent_reference_state) {
        config["initial_conditions"]["rcemip_consistent_reference_state"] = true;
    }

    // No automatic input readers are configured. The test calls TxtReader
    // explicitly after the Initializer constructor has generated the grid.
    if (structured) {
        config["grid"] = {
            {"horizontal", {
                {"nx", 4},
                {"ny", 4},
                {"n_halo_cells", 2},
                {"geometry", {
                    {"kind", "cartesian"},
                    {"dx", 500.0},
                    {"dy", 500.0}
                }},
                {"topology", {
                    {"q1", "periodic"},
                    {"q2", "periodic"}
                }}
            }},
            {"vertical", {
                {"nz", nz},
                {"type", type},
                {"dz", 500.0},
                {"dz1", dz1},
                {"rcemip_grid_data_path", grid_data_path}
            }}
        };
    } else {
        config["grid"] = {
            {"nx", 4},
            {"ny", 4},
            {"nz", nz},
            {"n_halo_cells", 2},
            {"dx", 500.0},
            {"dy", 500.0},
            {"dz", 500.0},
            {"dz1", dz1},
            {"vertical_coordinate_type", type},
            {"rcemip_grid_data_path", grid_data_path}
        };
    }

    return config;
}

Snapshot take_snapshot(const Parameters& parameters, const State& state, const int halo) {
    const std::array<const Field<1>*, 8> fields = {
        &parameters.z_mid,
        &parameters.z_up,
        &parameters.flex_height_coef_mid,
        &parameters.flex_height_coef_up,
        &parameters.dz_mid,
        &parameters.dz_up,
        &parameters.fact1_xi_eta,
        &parameters.fact2_xi_eta
    };

    Snapshot result;

    for (std::size_t n = 0; n < fields.size(); ++n) {
        const auto host = fields[n]->get_host_data();
        result.vertical[n].resize(host.extent(0));

        for (std::size_t k = 0; k < host.extent(0); ++k) {
            result.vertical[n][k] = host(k);
        }
    }

    // TxtReader initializes forcing profiles from halo-1 upward.
    // Do not compare untouched lower forcing halos.
    for (std::size_t n = 0; n < profile_names.size(); ++n) {
        const auto host = state.get_field<1>(profile_names[n]).get_host_data();

        for (int k = halo - 1; k < static_cast<int>(host.extent(0)); ++k) {
            result.profiles[n].push_back(host(k));
        }
    }

    return result;
}

#if defined(ENABLE_NCCL)
Snapshot initialize_configuration(const std::filesystem::path& path, const std::filesystem::path& sounding_path, ncclComm_t communicator) {
#else
Snapshot initialize_configuration(const std::filesystem::path& path, const std::filesystem::path& sounding_path) {
#endif
    const ConfigurationManager config(path.string());
    const Grid grid(config);
    Parameters parameters(config, grid);

#if defined(ENABLE_NCCL)
    const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
    State state(config, parameters, grid, communicator, stream);
    HaloExchanger halo_exchanger(config, grid, communicator, stream);
#else
    State state(config, parameters, grid);
    HaloExchanger halo_exchanger(grid);
#endif

    Initializer initializer(config, grid, parameters, state, halo_exchanger);

    VVM::IO::TxtReader reader(sounding_path.string(), grid, parameters, config);
    reader.read_and_initialize(state);

    // Do not call initialize_state(): this test covers grid and profile
    // initialization, not spatial input, perturbations, or model dynamics.
    Kokkos::fence();
    return take_snapshot(parameters, state, grid.get_halo_cells());
}

void compare_vector(const std::vector<Real>& expected, const std::vector<Real>& actual, const std::string& label) {
    if (expected.size() != actual.size()) {
        check(false, label + ": different extent.");
        return;
    }

    for (std::size_t k = 0; k < expected.size(); ++k) {
        if (!std::isfinite(expected[k]) || !std::isfinite(actual[k]) || expected[k] != actual[k]) {
            check(false, label + ": mismatch at index " + std::to_string(k));
            return;
        }
    }
}

void compare_snapshots(const Snapshot& expected, const Snapshot& actual, const std::string& label) {
    for (std::size_t n = 0; n < vertical_names.size(); ++n) {
        compare_vector(expected.vertical[n], actual.vertical[n], label + ": " + vertical_names[n]);
    }

    for (std::size_t n = 0; n < profile_names.size(); ++n) {
        compare_vector(expected.profiles[n], actual.profiles[n], label + ": " + profile_names[n]);
    }
}

void check_physical_grid(const Snapshot& snapshot, const std::string& label) {
    const auto& values = snapshot.vertical;
    const int h = 2;
    const int nz = static_cast<int>(values[0].size());

    for (int k = h; k < nz - h; ++k) {
        check(values[0][k] > values[0][k - 1], label + ": z_mid must increase.");
        check(values[1][k] > values[1][k - 1], label + ": z_up must increase.");

        for (std::size_t n = 2; n <= 5; ++n) {
            check(std::isfinite(values[n][k]) && values[n][k] > Real(0),
                label + ": positive finite " + vertical_names[n]);
        }

        if (k < nz - h - 1) {
            check(std::isfinite(values[6][k]) && values[6][k] > Real(0), label + ": positive fact1_xi_eta.");
            check(std::isfinite(values[7][k]) && values[7][k] > Real(0), label + ": positive fact2_xi_eta.");
        }

        // Profile snapshots start at model index h-1.
        const int profile_index = k - (h - 1);
        check(snapshot.profiles[0][profile_index] > Real(0), label + ": positive temperature.");
        check(snapshot.profiles[2][profile_index] > Real(0), label + ": positive pressure.");
        check(snapshot.profiles[7][profile_index] > Real(0), label + ": positive density.");
    }
}

void check_uniform_grid(const Snapshot& snapshot) {
    const int h = 2;
    const int nz = static_cast<int>(snapshot.vertical[0].size());

    for (int k = h; k < nz - h; ++k) {
        check(snapshot.vertical[0][k] == Real(k - h + 0.5) * Real(500), "Uniform z_mid must match its expected height.");
        check(snapshot.vertical[1][k] == Real(k - h + 1) * Real(500), "Uniform z_up must match its expected height.");
        check(snapshot.vertical[2][k] == Real(1), "Uniform mid flex coefficient must equal one.");
        check(snapshot.vertical[3][k] == Real(1), "Uniform up flex coefficient must equal one.");
        check(snapshot.vertical[4][k] == Real(500), "Uniform dz_mid must equal 500 m.");
        check(snapshot.vertical[5][k] == Real(500), "Uniform dz_up must equal 500 m.");
    }
}

void check_rcemip_reference(const Snapshot& snapshot, const bool consistent_reference_state) {
    const int h = 2;
    const std::array<Real, 4> expected_mid = {Real(50), Real(170), Real(330), Real(530)};
    const std::array<Real, 4> expected_up = {Real(100), Real(240), Real(420), Real(640)};

    for (int k = 0; k < 4; ++k) {
        check(snapshot.vertical[0][h + k] == expected_mid[k], "RCEMIP z_mid must come from the selected grid file.");
        check(snapshot.vertical[1][h + k] == expected_up[k], "RCEMIP z_up must come from the selected grid file.");
    }

    // Snapshot profile index 1 corresponds to the first physical level,
    // which reads the second sounding row in the existing RCEMIP branch.
    const int index = 1;
    const Real pressure = Real(95000);
    const Real p0 = Real(100000);
    const Real rd = Real(287);
    const Real cp = Real(1004);
    const Real lv = Real(2500000);
    const Real qv = Real(0.01);
    const Real rd_over_cp = rd / cp;

    const Real source_exner = std::pow(pressure / p0, rd_over_cp);
    const Real theta = Real(300) / source_exner;
    const Real exner = consistent_reference_state
        ? std::pow(pressure / p0, rd_over_cp)
        : std::pow(pressure / p0, 2.0 / 7.0);

    const Real temperature = theta * exner;
    const Real virtual_temperature = temperature * (Real(1) + Real(0.608) * qv);
    const Real density = consistent_reference_state
        ? pressure / (rd * virtual_temperature)
        : pressure / (rd * theta * exner);

    check_close(snapshot.profiles[2][index], pressure, "RCEMIP pressure must use the matching sounding row.");
    check_close(snapshot.profiles[3][index], exner, "RCEMIP Exner compatibility setting must be preserved.");
    check_close(snapshot.profiles[7][index], density, "RCEMIP density compatibility setting must be preserved.");

    check_close(snapshot.profiles[9][index], Real(4), "RCEMIP U must use direct row indexing.");
    check_close(snapshot.profiles[10][index], Real(-2), "RCEMIP V must use direct row indexing.");
    check_close(snapshot.profiles[11][index], -Real(2) / exner / Real(86400), "RCEMIP Q1 conversion must remain unchanged.");
    check_close(snapshot.profiles[12][index], Real(1) / ((lv / cp) * Real(86400)), "RCEMIP Q2 conversion must remain unchanged.");
}

#if defined(ENABLE_NCCL)
void run_tests(ncclComm_t communicator) {
#else
void run_tests() {
#endif
    TemporaryDirectory temporary;
    const auto grid_data_path = temporary.path() / "rcemip_grid.txt";
    const auto sounding_path = temporary.path() / "sounding.txt";

    write_text(grid_data_path,
        "zz zt\n"
        "100 50\n"
        "240 170\n"
        "420 330\n"
        "640 530\n");

    // Constant temperature and humidity keep extrapolation well behaved
    // for the taller analytic grids. Winds and forcing vary by source row.
    write_text(sounding_path,
        "pbar Tbar qvbar U V Q1 Q2\n"
        "100000 300 0.01 2 -1 1 0.5\n"
        "95000 300 0.01 4 -2 2 1.0\n"
        "90000 300 0.01 6 -3 3 1.5\n"
        "85000 300 0.01 8 -4 4 2.0\n"
        "80000 300 0.01 10 -5 5 2.5\n");

    struct TestCase {
        const char* name;
        const char* type;
        double dz1;
        bool consistent_reference_state;
    };

    const std::array<TestCase, 5> cases = {{
        {"uniform", "default", 500.0, false},
        {"stretched", "default", 250.0, false},
        {"taiwanvvm", "taiwanvvm", 250.0, false},
        {"rcemip", "rcemip", 250.0, false},
        {"rcemip_consistent", "rcemip", 250.0, true}
    }};

    for (const auto& test_case : cases) {
        const int failures_before = failures;

        const Json legacy = make_configuration(test_case.type, test_case.dz1, false, grid_data_path.string(), test_case.consistent_reference_state);
        const Json structured = make_configuration(test_case.type, test_case.dz1, true, grid_data_path.string(), test_case.consistent_reference_state);

        Json mixed = structured;
        mixed["grid"]["nx"] = 12;
        mixed["grid"]["ny"] = 10;
        mixed["grid"]["nz"] = 8;
        mixed["grid"]["n_halo_cells"] = 3;
        mixed["grid"]["dx"] = 11.0;
        mixed["grid"]["dy"] = 22.0;
        mixed["grid"]["dz"] = 33.0;
        mixed["grid"]["dz1"] = 33.0;
        mixed["grid"]["vertical_coordinate_type"] = "deliberately_ignored";
        mixed["grid"]["rcemip_grid_data_path"] = (temporary.path() / "must_not_be_opened.txt").string();

        const auto legacy_path = temporary.path() / (std::string(test_case.name) + "_legacy.json");
        const auto structured_path = temporary.path() / (std::string(test_case.name) + "_structured.json");
        const auto mixed_path = temporary.path() / (std::string(test_case.name) + "_mixed.json");

        write_text(legacy_path, legacy.dump(4));
        write_text(structured_path, structured.dump(4));
        write_text(mixed_path, mixed.dump(4));

#if defined(ENABLE_NCCL)
        const auto legacy_values = initialize_configuration(legacy_path, sounding_path, communicator);
        const auto structured_values = initialize_configuration(structured_path, sounding_path, communicator);
        const auto mixed_values = initialize_configuration(mixed_path, sounding_path, communicator);
#else
        const auto legacy_values = initialize_configuration(legacy_path, sounding_path);
        const auto structured_values = initialize_configuration(structured_path, sounding_path);
        const auto mixed_values = initialize_configuration(mixed_path, sounding_path);
#endif

        compare_snapshots(legacy_values, structured_values, std::string(test_case.name) + " legacy/structured");
        compare_snapshots(structured_values, mixed_values, std::string(test_case.name) + " precedence");
        check_physical_grid(structured_values, test_case.name);

        if (std::string(test_case.name) == "uniform") {
            check_uniform_grid(structured_values);
        }

        if (std::string(test_case.type) == "rcemip") {
            check_rcemip_reference(structured_values, test_case.consistent_reference_state);
        }

        std::printf("%s grid_profiles_and_precedence %s\n",
            test_case.name, failures == failures_before ? "PASS" : "FAIL");
    }
}

#if defined(ENABLE_NCCL)
void require_nccl(const ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}
#endif

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    try {
        int mpi_size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        if (mpi_size != 1) {
            throw std::invalid_argument("This vertical-grid configuration test requires one MPI rank.");
        }

#if defined(ENABLE_NCCL)
        ncclUniqueId identifier;
        require_nccl(ncclGetUniqueId(&identifier), "Create NCCL identifier");

        ncclComm_t communicator = nullptr;
        require_nccl(ncclCommInitRank(&communicator, 1, identifier, 0), "Initialize NCCL communicator");

        try {
            run_tests(communicator);
        } catch (...) {
            ncclCommAbort(communicator);
            throw;
        }

        require_nccl(ncclCommDestroy(communicator), "Destroy NCCL communicator");
#else
        run_tests();
#endif
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_vertical_grid_configuration: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::finalize();
    MPI_Finalize();

    return global_failures == 0 ? 0 : 1;
}
