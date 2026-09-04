#include "core/GridSpecification.hpp"
#include "utils/ConfigurationManager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include <mpi.h>

namespace {

using VVM::Core::GridSpecification;
using VVM::Core::HorizontalEdgeTopology;
using VVM::Core::VerticalCoordinateType;
using VVM::Core::Geometry::GeometryKind;

int failures = 0;

constexpr VVM::Real relative_tolerance =
    sizeof(VVM::Real) == sizeof(double)
        ? VVM::real(1.0e-12)
        : VVM::real(5.0e-5);

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

bool close(const VVM::Real actual, const VVM::Real expected) {
    const VVM::Real scale =
        std::max(VVM::real(1.0), std::abs(expected));

    return std::abs(actual - expected) <=
        relative_tolerance * scale;
}

nlohmann::json make_legacy_config() {
    return {
        {"grid", {
            {"nx", 384},
            {"ny", 384},
            {"nz", 74},
            {"n_halo_cells", 2},
            {"dx", 3000.0},
            {"dy", 3000.0},
            {"dz", 500.0},
            {"dz1", 100.0},
            {"fix_lonlat", true},
            {"vertical_coordinate_type", "rcemip"},
            {"rcemip_grid_data_path", "legacy_rcemip_grid.txt"},
            {"boundary_condition", {
                {"x", "periodic"},
                {"y", "zero_gradient"}
            }}
        }}
    };
}

nlohmann::json make_structured_rll_config() {
    nlohmann::json config =
        make_legacy_config();

    config["grid"]["horizontal"] = {
        {"nx", 400},
        {"ny", 100},
        {"n_halo_cells", 2},
        {"geometry", {
            {"kind", "regular_latlon"},
            {"fix_lonlat", false},
            {"earth_radius_m", 6371220.0},
            {"longitude_bounds_deg", {0.0, 360.0}},
            {"latitude_bounds_deg", {-45.0, 45.0}}
        }},
        {"topology", {
            {"q1", "periodic"},
            {"q2", "bounded"}
        }}
    };

    config["grid"]["vertical"] = {
        {"nz", 4},
        {"type", "rcemip"},
        {"__vertical_coordinate_type_options", "default, taiwanvvm, rcemip"},
        {"rcemip_grid_data_path", "structured_rcemip_grid.txt"},
        {"dz", 1000.0},
        {"dz1", 1000.0}
    };

    return config;
}

std::string write_config(
    const std::filesystem::path& directory,
    const std::string& name,
    const nlohmann::json& contents) {

    const std::filesystem::path path =
        directory / name;

    std::ofstream file(path);
    file << contents.dump(2) << '\n';
    file.close();

    if (!file) {
        throw std::runtime_error(
            "Failed to write test configuration: " +
            path.string());
    }

    return path.string();
}

GridSpecification parse_spec(
    const std::filesystem::path& directory,
    const std::string& name,
    const nlohmann::json& contents) {

    const VVM::Utils::ConfigurationManager config(
        write_config(directory, name, contents));

    return GridSpecification::from_config(config);
}

void expect_invalid(
    const std::filesystem::path& directory,
    const std::string& name,
    const nlohmann::json& contents,
    const char* message) {

    try {
        (void)parse_spec(
            directory,
            name,
            contents);

        check(false, message);
    } catch (const std::exception&) {
    }
}

void test_legacy_configuration(
    const std::filesystem::path& directory) {

    const GridSpecification spec =
        parse_spec(
            directory,
            "legacy.json",
            make_legacy_config());

    check(
        spec.horizontal.nx == 384,
        "legacy grid.nx must be used");

    check(
        spec.horizontal.ny == 384,
        "legacy grid.ny must be used");

    check(
        spec.horizontal.n_halo_cells == 2,
        "legacy halo width must be used");

    check(
        spec.horizontal.geometry.kind ==
            GeometryKind::Cartesian,
        "legacy configuration must select Cartesian geometry");

    check(
        close(
            spec.horizontal.geometry.dq1,
            VVM::real(3000.0)),
        "legacy grid.dx must become Cartesian dq1");

    check(
        close(
            spec.horizontal.geometry.dq2,
            VVM::real(3000.0)),
        "legacy grid.dy must become Cartesian dq2");

    check(
        spec.horizontal.fix_lonlat,
        "legacy grid.fix_lonlat must be retained");

    check(
        spec.horizontal.topology.q1 ==
            HorizontalEdgeTopology::Periodic,
        "legacy periodic x boundary must become periodic q1 topology");

    check(
        spec.horizontal.topology.q2 ==
            HorizontalEdgeTopology::Bounded,
        "legacy zero-gradient y boundary must become bounded q2 topology");

    check(
        spec.vertical.nz == 74,
        "legacy grid.nz must be used");

    check(
        close(
            spec.vertical.dz,
            VVM::real(500.0)),
        "legacy grid.dz must be used");

    check(
        close(
            spec.vertical.dz1,
            VVM::real(100.0)),
        "legacy grid.dz1 must be used");

    check(
        spec.vertical.type ==
            VerticalCoordinateType::RCEMIP,
        "legacy vertical coordinate type must be retained");

    check(
        spec.vertical.rcemip_grid_data_path ==
            "legacy_rcemip_grid.txt",
        "legacy RCEMIP path must be retained");
}

void test_structured_scopes_override_legacy(
    const std::filesystem::path& directory) {

    const GridSpecification spec =
        parse_spec(
            directory,
            "structured_rll.json",
            make_structured_rll_config());

    const VVM::Real pi =
        std::acos(VVM::real(-1.0));

    const VVM::Real expected_spacing =
        VVM::real(0.9) *
        pi /
        VVM::real(180.0);

    check(
        spec.horizontal.nx == 400,
        "structured horizontal nx must override legacy grid.nx");

    check(
        spec.horizontal.ny == 100,
        "structured horizontal ny must override legacy grid.ny");

    check(
        spec.horizontal.geometry.kind ==
            GeometryKind::RegularLatLon,
        "structured geometry must override legacy Cartesian geometry");

    check(
        close(
            spec.horizontal.geometry.dq1,
            expected_spacing),
        "structured longitude bounds must produce 0.9-degree dq1");

    check(
        close(
            spec.horizontal.geometry.dq2,
            expected_spacing),
        "structured latitude bounds must produce 0.9-degree dq2");

    check(
        close(
            spec.horizontal.geometry.regular_lat_lon.radius,
            VVM::real(6371220.0)),
        "structured Earth radius must be retained");

    check(
        !spec.horizontal.fix_lonlat,
        "structured fix_lonlat must override the legacy value");

    check(
        spec.horizontal.topology.q1 ==
            HorizontalEdgeTopology::Periodic,
        "structured q1 topology must be used");

    check(
        spec.horizontal.topology.q2 ==
            HorizontalEdgeTopology::Bounded,
        "structured q2 topology must be used");

    check(
        spec.vertical.nz == 4,
        "structured vertical nz must override legacy grid.nz");

    check(
        close(
            spec.vertical.dz,
            VVM::real(1000.0)),
        "structured vertical dz must override legacy grid.dz");

    check(
        close(
            spec.vertical.dz1,
            VVM::real(1000.0)),
        "structured vertical dz1 must override legacy grid.dz1");

    check(
        spec.vertical.spacing_parameters_are_equal(),
        "equal structured dz and dz1 must be recorded");

    check(
        spec.vertical.type ==
            VerticalCoordinateType::RCEMIP,
        "structured vertical type must override the legacy type");

    check(
        !spec.vertical.uses_uniform_analytic_coordinate(),
        "RCEMIP heights come from a file even when dz equals dz1");

    check(
        spec.vertical.rcemip_grid_data_path ==
            "structured_rcemip_grid.txt",
        "structured RCEMIP path must override the legacy path");
}

void test_independent_scope_fallback(
    const std::filesystem::path& directory) {

    nlohmann::json horizontal_only =
        make_legacy_config();

    horizontal_only["grid"]["horizontal"] =
        make_structured_rll_config()["grid"]["horizontal"];

    const GridSpecification horizontal_spec =
        parse_spec(
            directory,
            "horizontal_only.json",
            horizontal_only);

    check(
        horizontal_spec.horizontal.nx == 400,
        "new horizontal scope must be used when present");

    check(
        horizontal_spec.vertical.nz == 74,
        "legacy vertical scope must remain available independently");

    nlohmann::json vertical_only =
        make_legacy_config();

    vertical_only["grid"]["vertical"] =
        make_structured_rll_config()["grid"]["vertical"];

    const GridSpecification vertical_spec =
        parse_spec(
            directory,
            "vertical_only.json",
            vertical_only);

    check(
        vertical_spec.horizontal.nx == 384,
        "legacy horizontal scope must remain available independently");

    check(
        vertical_spec.vertical.nz == 4,
        "new vertical scope must be used when present");
}

void test_default_vertical_spacing(
    const std::filesystem::path& directory) {

    nlohmann::json uniform =
        make_legacy_config();

    uniform["grid"]["vertical"] = {
        {"nz", 33},
        {"type", "default"},
        {"dz", 500.0},
        {"dz1", 500.0}
    };

    const GridSpecification uniform_spec =
        parse_spec(
            directory,
            "uniform_vertical.json",
            uniform);

    check(
        uniform_spec.vertical.uses_uniform_analytic_coordinate(),
        "default vertical coordinate with dz equal to dz1 must be uniform");

    nlohmann::json stretched =
        make_legacy_config();

    stretched["grid"]["vertical"] = {
        {"nz", 74},
        {"type", "default"},
        {"dz", 500.0},
        {"dz1", 100.0}
    };

    const GridSpecification stretched_spec =
        parse_spec(
            directory,
            "stretched_vertical.json",
            stretched);

    check(
        stretched_spec.vertical.uses_default_stretching(),
        "default vertical coordinate with different dz and dz1 must use stretching");
}

void test_structured_cartesian(
    const std::filesystem::path& directory) {

    nlohmann::json config =
        make_legacy_config();

    config["grid"]["horizontal"] = {
        {"nx", 64},
        {"ny", 48},
        {"n_halo_cells", 2},
        {"geometry", {
            {"kind", "cartesian"},
            {"fix_lonlat", true},
            {"dx", 1000.0},
            {"dy", 2000.0}
        }},
        {"topology", {
            {"q1", "periodic"},
            {"q2", "bounded"}
        }}
    };

    const GridSpecification spec =
        parse_spec(
            directory,
            "structured_cartesian.json",
            config);

    check(
        spec.horizontal.geometry.kind ==
            GeometryKind::Cartesian,
        "structured Cartesian kind must be parsed");

    check(
        close(
            spec.horizontal.geometry.dq1,
            VVM::real(1000.0)),
        "structured Cartesian dx must become dq1");

    check(
        close(
            spec.horizontal.geometry.dq2,
            VVM::real(2000.0)),
        "structured Cartesian dy must become dq2");
}

void test_invalid_configurations(
    const std::filesystem::path& directory) {

    nlohmann::json config =
        make_structured_rll_config();

    config["grid"]["horizontal"]["geometry"]["kind"] =
        "cubed_sphere";

    expect_invalid(
        directory,
        "cubed_sphere.json",
        config,
        "unimplemented cubed-sphere geometry was accepted");

    config =
        make_structured_rll_config();

    config["grid"]["horizontal"]["topology"]["q2"] =
        "periodic";

    expect_invalid(
        directory,
        "periodic_latitude.json",
        config,
        "periodic RLL latitude was accepted");

    config =
        make_structured_rll_config();

    config["grid"]["horizontal"]["geometry"]["fix_lonlat"] =
        true;

    expect_invalid(
        directory,
        "fixed_rll_coordinates.json",
        config,
        "RLL geometry with fix_lonlat enabled was accepted");

    config =
        make_structured_rll_config();

    config["grid"]["horizontal"]["geometry"]["longitude_bounds_deg"] =
        {0.0, 180.0};

    expect_invalid(
        directory,
        "regional_periodic_longitude.json",
        config,
        "regional RLL longitude with periodic topology was accepted");

    config =
        make_structured_rll_config();

    config["grid"]["horizontal"]["geometry"]["latitude_bounds_deg"] =
        {-89.0, 1.0};

    expect_invalid(
        directory,
        "latitude_halo_crosses_pole.json",
        config,
        "RLL latitude halos crossing a pole were accepted");

    config =
        make_legacy_config();

    config["grid"]["vertical"] = {
        {"nz", 74},
        {"type", "taiwanvvm"},
        {"dz", 500.0},
        {"dz1", 500.0}
    };

    expect_invalid(
        directory,
        "uniform_taiwanvvm.json",
        config,
        "TaiwanVVM coordinate with equal dz and dz1 was accepted");

    config =
        make_legacy_config();

    config["grid"]["vertical"] = {
        {"nz", 74},
        {"type", "default"},
        {"dz", 100.0},
        {"dz1", 500.0}
    };

    expect_invalid(
        directory,
        "reversed_stretching.json",
        config,
        "analytic vertical coordinate with dz1 greater than dz was accepted");

    config =
        make_legacy_config();

    config["grid"]["vertical"] = {
        {"nz", 74},
        {"type", "rcemip"},
        {"dz", 500.0},
        {"dz1", 100.0}
    };

    expect_invalid(
        directory,
        "missing_rcemip_path.json",
        config,
        "RCEMIP coordinate without a source path was accepted");

    config =
        make_legacy_config();

    config["grid"]["vertical"] = {
        {"nz", 74},
        {"type", "terrain_following"},
        {"dz", 500.0},
        {"dz1", 100.0}
    };

    expect_invalid(
        directory,
        "unknown_vertical_type.json",
        config,
        "unknown vertical-coordinate type was accepted");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    namespace fs = std::filesystem;

    const fs::path directory =
        fs::temp_directory_path() /
        ("vvm_grid_specification_" +
         std::to_string(
             static_cast<long long>(getpid())));

    std::error_code error;
    fs::remove_all(directory, error);
    fs::create_directories(directory, error);

    if (error) {
        std::fprintf(
            stderr,
            "FAIL: unable to create test directory: %s\n",
            error.message().c_str());

        MPI_Finalize();
        return 1;
    }

    try {
        test_legacy_configuration(directory);
        test_structured_scopes_override_legacy(directory);
        test_independent_scope_fallback(directory);
        test_default_vertical_spacing(directory);
        test_structured_cartesian(directory);
        test_invalid_configurations(directory);
    } catch (const std::exception& exception) {
        ++failures;

        std::fprintf(
            stderr,
            "Unexpected exception: %s\n",
            exception.what());
    }

    fs::remove_all(directory, error);

    if (failures == 0) {
        std::puts("test_grid_specification: PASS");
    }

    MPI_Finalize();

    return failures == 0 ? 0 : 1;
}
