#include "core/GridSpecification.hpp"
#include "core/ModelConfigurationValidation.hpp"
#include "utils/ConfigurationManager.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

namespace {

using Json = nlohmann::json;
using VVM::Core::GridSpecification;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::validate_model_numerical_configuration;
using VVM::Utils::ConfigurationManager;

int failures = 0;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const std::string pattern = (std::filesystem::temp_directory_path() / "vvm_model_validation_XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');

        const char* created = ::mkdtemp(buffer.data());
        if (!created) {
            throw std::runtime_error("Cannot create temporary model-validation directory.");
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

std::filesystem::path write_configuration(const std::filesystem::path& directory, const std::string& name, const Json& config) {
    const auto path = directory / (name + ".json");
    std::ofstream output(path);

    if (!output) {
        throw std::runtime_error("Cannot create test configuration: " + path.string());
    }

    output << config.dump(4);
    output.close();

    if (!output) {
        throw std::runtime_error("Cannot finish writing test configuration: " + path.string());
    }

    return path;
}

Json make_legacy_configuration() {
    return {
        {"grid", {
            {"nx", 64},
            {"ny", 48},
            {"nz", 32},
            {"n_halo_cells", 2},
            {"dx", 100.0},
            {"dy", 200.0},
            {"dz", 50.0},
            {"dz1", 25.0},
            {"vertical_coordinate_type", "default"}
        }},
        {"simulation", {
            {"dt_s", 1.0},
            {"total_time_s", 120.0},
            {"output_interval_s", 60.0}
        }}
    };
}

Json make_structured_configuration() {
    Json config = make_legacy_configuration();

    config["grid"] = {
        {"horizontal", {
            {"nx", 64},
            {"ny", 48},
            {"n_halo_cells", 2},
            {"geometry", {
                {"kind", "cartesian"},
                {"dx", 100.0},
                {"dy", 200.0}
            }},
            {"topology", {
                {"q1", "periodic"},
                {"q2", "periodic"}
            }}
        }},
        {"vertical", {
            {"nz", 32},
            {"type", "default"},
            {"dz", 50.0},
            {"dz1", 25.0}
        }}
    };

    return config;
}

void check_validation(const std::filesystem::path& directory, const std::string& name, const Json& config,
    const int compute_ranks, const bool should_pass, const std::string& expected_error = "") {

    const auto path = write_configuration(directory, name, config);

    bool accepted = false;
    std::string error_message;

    try {
        const ConfigurationManager configuration(path.string());
        validate_model_numerical_configuration(configuration, compute_ranks);
        accepted = true;
    } catch (const std::exception& error) {
        error_message = error.what();
    }

    bool passed = accepted == should_pass;

    if (!should_pass && !expected_error.empty()) {
        passed = passed && error_message.find(expected_error) != std::string::npos;
    }

    if (!passed) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s accepted=%d expected_accept=%d error='%s'\n",
            name.c_str(), accepted ? 1 : 0, should_pass ? 1 : 0, error_message.c_str());
    } else {
        std::printf("%s PASS\n", name.c_str());
    }
}

void run_tests() {
    TemporaryDirectory temporary;
    const auto& directory = temporary.path();

    const Json legacy = make_legacy_configuration();
    const Json structured = make_structured_configuration();

    // These rank counts exercise decomposition arithmetic. The unit test
    // itself does not launch or construct a four-rank Grid.
    check_validation(directory, "legacy", legacy, 4, true);
    check_validation(directory, "structured", structured, 4, true);

    Json horizontal_only = legacy;
    horizontal_only["grid"]["horizontal"] = structured["grid"]["horizontal"];
    check_validation(directory, "structured_horizontal_legacy_vertical", horizontal_only, 4, true);

    Json vertical_only = legacy;
    vertical_only["grid"]["vertical"] = structured["grid"]["vertical"];
    check_validation(directory, "legacy_horizontal_structured_vertical", vertical_only, 4, true);

    Json mixed = structured;
    mixed["grid"]["nx"] = 0;
    mixed["grid"]["ny"] = 0;
    mixed["grid"]["nz"] = 0;
    mixed["grid"]["n_halo_cells"] = 0;
    mixed["grid"]["dx"] = 0.0;
    mixed["grid"]["dy"] = 0.0;
    mixed["grid"]["dz"] = 0.0;
    mixed["grid"]["dz1"] = 0.0;
    mixed["grid"]["vertical_coordinate_type"] = "deliberately_ignored";

    check_validation(directory, "structured_overrides_invalid_legacy", mixed, 4, true);

    // A present but incomplete structured scope must not borrow missing
    // values from the legacy scope.
    Json incomplete = legacy;
    incomplete["grid"]["horizontal"] = structured["grid"]["horizontal"];
    incomplete["grid"]["horizontal"]["geometry"].erase("dx");

    check_validation(directory, "incomplete_structured_horizontal", incomplete, 4, false, "grid.horizontal.geometry.dx");

    Json invalid = structured;
    invalid["simulation"]["dt_s"] = 0.0;
    check_validation(directory, "zero_timestep", invalid, 4, false, "simulation.dt_s");

    invalid = structured;
    invalid["simulation"]["output_interval_s"] = 0.0;
    check_validation(directory, "zero_output_interval", invalid, 4, false, "simulation.output_interval_s");

    invalid = structured;
    invalid["simulation"]["total_time_s"] = -1.0;
    check_validation(directory, "negative_total_time", invalid, 4, false, "simulation.total_time_s");

    check_validation(directory, "zero_compute_ranks", structured, 0, false, "compute ranks");

    invalid = structured;
    invalid["grid"]["horizontal"]["n_halo_cells"] = 1;
    check_validation(directory, "insufficient_stencil_halo", invalid, 4, false, "Takacs");

    invalid = structured;
    invalid["grid"]["horizontal"]["nx"] = 3;
    invalid["grid"]["horizontal"]["ny"] = 1;
    check_validation(directory, "insufficient_local_width", invalid, 2, false, "smallest local width");

    invalid = structured;
    invalid["grid"]["horizontal"]["nx"] = 1;
    invalid["grid"]["horizontal"]["ny"] = 1;
    check_validation(directory, "multirank_column", invalid, 2, false, "exactly one compute rank");

    Json rll = structured;
    rll["grid"]["horizontal"]["geometry"] = {
        {"kind", "regular_latlon"},
        {"fix_lonlat", false},
        {"earth_radius_m", 6371220.0},
        {"longitude_bounds_deg", {0.0, 360.0}},
        {"latitude_bounds_deg", {-45.0, 45.0}}
    };
    rll["grid"]["horizontal"]["topology"]["q2"] = "bounded";

    // Distinguish a valid RLL specification from permission to run the
    // still-unmigrated full model.
    const auto rll_path = write_configuration(directory, "rll_specification", rll);
    const ConfigurationManager rll_configuration(rll_path.string());
    const auto rll_specification = GridSpecification::from_config(rll_configuration);

    if (rll_specification.horizontal.geometry.kind != GeometryKind::RegularLatLon) {
        ++failures;
        std::fprintf(stderr, "FAIL: RLL specification must remain available to component tests.\n");
    }

    check_validation(directory, "rll_model_guard", rll, 1, false, "Non-Cartesian full-model execution");
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    try {
        run_tests();
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_model_configuration_validation: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) {
        std::puts("test_model_configuration_validation: PASS");
    }

    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
