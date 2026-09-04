#include "core/GridSpecification.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Core {

namespace {

constexpr const char* default_rcemip_grid_data_path = "./rundata/initial_conditions/profiles/snd_rcemip_anal300_v3.txt";

[[noreturn]] void configuration_error(const std::string& message) {
    throw std::runtime_error("Configuration error: " + message);
}

void require_positive_integer(const int value, const char* key) {
    if (value <= 0) {
        configuration_error(std::string("'") + key + "' must be a positive integer.");
    }
}

void require_valid_halo_width(const int value, const char* key) {
    if (value < 2) {
        configuration_error(std::string("'") + key + "' must be at least 2 for the Takacs stencil.");
    }
}

void require_finite_positive(const VVM::Real value, const char* key) {
    if (!std::isfinite(value) || value <= VVM::real(0.0)) {
        configuration_error(std::string("'") + key + "' must be finite and greater than zero.");
    }
}

HorizontalEdgeTopology parse_structured_topology(const std::string& value, const char* key) {
    if (value == "periodic") {
        return HorizontalEdgeTopology::Periodic;
    }

    if (value == "bounded") {
        return HorizontalEdgeTopology::Bounded;
    }

    configuration_error(std::string("'") + key + "' must be 'periodic' or 'bounded'.");

    return HorizontalEdgeTopology::Bounded;
}

HorizontalEdgeTopology parse_legacy_boundary(const std::string& value, const char* key) {
    if (value == "periodic") {
        return HorizontalEdgeTopology::Periodic;
    }

    if (value == "zero_gradient" || value == "bounded") {
        return HorizontalEdgeTopology::Bounded;
    }

    configuration_error(std::string("'") + key + "' must be 'periodic', 'zero_gradient', or 'bounded'.");

    return HorizontalEdgeTopology::Bounded;
}

VerticalCoordinateType parse_vertical_coordinate_type(const std::string& value, const char* key) {
    if (value == "default") {
        return VerticalCoordinateType::Default;
    }

    if (value == "taiwanvvm") {
        return VerticalCoordinateType::TaiwanVVM;
    }

    if (value == "rcemip") {
        return VerticalCoordinateType::RCEMIP;
    }

    configuration_error(std::string("'") + key + "' must be 'default', 'taiwanvvm', or 'rcemip'.");

    return VerticalCoordinateType::Default;
}

Geometry::GeometryKind parse_geometry_kind(const std::string& value, const char* key) {

    if (value == "cartesian") {
        return Geometry::GeometryKind::Cartesian;
    }

    if (value == "regular_latlon") {
        return Geometry::GeometryKind::RegularLatLon;
    }

    if (value == "cubed_sphere") {
        return Geometry::GeometryKind::CubedSphere;
    }

    configuration_error(std::string("'") + key + "' must be 'cartesian', 'regular_latlon', or 'cubed_sphere'.");

    return Geometry::GeometryKind::Cartesian;
}

std::vector<double> read_bounds(const Utils::ConfigurationManager& config, const char* key) {
    const std::vector<double> bounds = config.get_value<std::vector<double>>(key);

    if (bounds.size() != 2) {
        configuration_error(
            std::string("'") + key + "' must contain exactly two values.");
    }

    if (!std::isfinite(bounds[0]) || !std::isfinite(bounds[1])) {
        configuration_error(
            std::string("'") + key + "' must contain finite values.");
    }

    return bounds;
}

void configure_cartesian_geometry(HorizontalDomainSpec& horizontal,
    const Utils::ConfigurationManager& config, const bool structured) {

    const char* dx_key = structured ? "grid.horizontal.geometry.dx" : "grid.dx";
    const char* dy_key = structured ? "grid.horizontal.geometry.dy" : "grid.dy";

    horizontal.geometry.kind = Geometry::GeometryKind::Cartesian;
    horizontal.geometry.dq1 = config.get_value<VVM::Real>(dx_key);
    horizontal.geometry.dq2 = config.get_value<VVM::Real>(dy_key);

    require_finite_positive(horizontal.geometry.dq1, dx_key);
    require_finite_positive(horizontal.geometry.dq2, dy_key);
}

void configure_regular_lat_lon_geometry(HorizontalDomainSpec& horizontal, const Utils::ConfigurationManager& config) {
    const VVM::Real radius = config.get_value<VVM::Real>("grid.horizontal.geometry.earth_radius_m");
    const std::vector<double> longitude_bounds = read_bounds(config, "grid.horizontal.geometry.longitude_bounds_deg");
    const std::vector<double> latitude_bounds = read_bounds(config, "grid.horizontal.geometry.latitude_bounds_deg");

    require_finite_positive(radius, "grid.horizontal.geometry.earth_radius_m");

    const double longitude_west = longitude_bounds[0];
    const double longitude_east = longitude_bounds[1];
    const double latitude_south = latitude_bounds[0];
    const double latitude_north = latitude_bounds[1];

    if (longitude_east <= longitude_west) {
        configuration_error("'grid.horizontal.geometry.longitude_bounds_deg' must be ordered west to east.");
    }

    const double longitude_span = longitude_east - longitude_west;

    if (longitude_span > 360.0) {
        configuration_error("'grid.horizontal.geometry.longitude_bounds_deg' must not span more than 360 degrees.");
    }

    if (latitude_north <= latitude_south) {
        configuration_error("'grid.horizontal.geometry.latitude_bounds_deg' must be ordered south to north.");
    }

    if (latitude_south <= -90.0 || latitude_north >= 90.0) {
        configuration_error("'grid.horizontal.geometry.latitude_bounds_deg' must remain strictly between the poles.");
    }

    const double dlongitude_degrees = longitude_span / static_cast<double>(horizontal.nx);
    const double dlatitude_degrees = (latitude_north - latitude_south) / static_cast<double>(horizontal.ny);

    const double minimum_halo_latitude = latitude_south + (0.5 - static_cast<double>(horizontal.n_halo_cells)) * dlatitude_degrees;
    const double maximum_halo_latitude = latitude_north + static_cast<double>(horizontal.n_halo_cells) * dlatitude_degrees;

    if (minimum_halo_latitude <= -90.0 || maximum_halo_latitude >= 90.0) {
        configuration_error("the regular latitude-longitude halo points reach or cross a pole.");
    }

    const bool longitude_is_global = std::abs(longitude_span - 360.0) <= 1.0e-10;

    if (horizontal.topology.q1 == HorizontalEdgeTopology::Periodic && !longitude_is_global) {
        configuration_error("periodic RLL q1 topology requires a 360-degree longitude span.");
    }

    if (horizontal.topology.q1 == HorizontalEdgeTopology::Bounded && longitude_is_global) {
        configuration_error("a 360-degree RLL longitude span requires periodic q1 topology.");
    }

    if (horizontal.topology.q2 != HorizontalEdgeTopology::Bounded) {
        configuration_error("regular latitude-longitude q2 topology must be bounded.");
    }

    if (horizontal.fix_lonlat) {
        configuration_error("'grid.horizontal.geometry.fix_lonlat' must be false for regular latitude-longitude geometry.");
    }

    const double degrees_to_radians = std::acos(-1.0) / 180.0;

    horizontal.geometry.kind = Geometry::GeometryKind::RegularLatLon;
    horizontal.geometry.dq1 = static_cast<VVM::Real>(dlongitude_degrees * degrees_to_radians);
    horizontal.geometry.dq2 = static_cast<VVM::Real>(dlatitude_degrees * degrees_to_radians);
    horizontal.geometry.regular_lat_lon.longitude_west_edge = static_cast<VVM::Real>(longitude_west * degrees_to_radians);
    horizontal.geometry.regular_lat_lon.latitude_south_edge = static_cast<VVM::Real>(latitude_south * degrees_to_radians);
    horizontal.geometry.regular_lat_lon.radius = radius;

    require_finite_positive(horizontal.geometry.dq1, "derived RLL longitude spacing");
    require_finite_positive(horizontal.geometry.dq2, "derived RLL latitude spacing");
}

HorizontalDomainSpec parse_structured_horizontal(
    const Utils::ConfigurationManager& config) {

    HorizontalDomainSpec horizontal;

    horizontal.nx = config.get_value<int>("grid.horizontal.nx");
    horizontal.ny = config.get_value<int>("grid.horizontal.ny");
    horizontal.n_halo_cells = config.get_value<int>("grid.horizontal.n_halo_cells");

    require_positive_integer(horizontal.nx, "grid.horizontal.nx");
    require_positive_integer(horizontal.ny, "grid.horizontal.ny");
    require_valid_halo_width(horizontal.n_halo_cells, "grid.horizontal.n_halo_cells");

    horizontal.fix_lonlat = config.get_value<bool>("grid.horizontal.geometry.fix_lonlat", false);
    horizontal.topology.q1 = parse_structured_topology(config.get_value<std::string>("grid.horizontal.topology.q1"), "grid.horizontal.topology.q1");
    horizontal.topology.q2 = parse_structured_topology(config.get_value<std::string>("grid.horizontal.topology.q2"), "grid.horizontal.topology.q2");

    const Geometry::GeometryKind kind = parse_geometry_kind(config.get_value<std::string>("grid.horizontal.geometry.kind"), "grid.horizontal.geometry.kind");

    switch (kind) {
        case Geometry::GeometryKind::Cartesian:
            configure_cartesian_geometry(horizontal, config, true);
            break;

        case Geometry::GeometryKind::RegularLatLon:
            configure_regular_lat_lon_geometry(horizontal, config);
            break;

        case Geometry::GeometryKind::CubedSphere:
            configuration_error("cubed-sphere geometry is recognized but is not implemented.");
    }

    return horizontal;
}

HorizontalDomainSpec parse_legacy_horizontal(const Utils::ConfigurationManager& config) {
    HorizontalDomainSpec horizontal;

    horizontal.nx = config.get_value<int>("grid.nx");
    horizontal.ny = config.get_value<int>("grid.ny");
    horizontal.n_halo_cells = config.get_value<int>("grid.n_halo_cells");

    require_positive_integer(horizontal.nx, "grid.nx");
    require_positive_integer(horizontal.ny, "grid.ny");
    require_valid_halo_width(horizontal.n_halo_cells, "grid.n_halo_cells");

    horizontal.fix_lonlat = config.get_value<bool>("grid.fix_lonlat", false);

    horizontal.topology.q1 = parse_legacy_boundary(config.get_value<std::string>("grid.boundary_condition.x", "periodic"), "grid.boundary_condition.x");
    horizontal.topology.q2 = parse_legacy_boundary(config.get_value<std::string>("grid.boundary_condition.y", "periodic"), "grid.boundary_condition.y");

    configure_cartesian_geometry(horizontal, config, false);

    return horizontal;
}

void validate_vertical_spec(const VerticalGridSpec& vertical, const char* nz_key,
    const char* dz_key, const char* dz1_key) {

    require_positive_integer(vertical.nz, nz_key);
    require_finite_positive(vertical.dz, dz_key);
    require_finite_positive(vertical.dz1, dz1_key);

    if (vertical.type != VerticalCoordinateType::RCEMIP &&
        vertical.dz1 > vertical.dz) {
        configuration_error(
            std::string("'") + dz1_key +
            "' must not exceed '" + dz_key +
            "' for an analytic vertical coordinate.");
    }

    if (vertical.type == VerticalCoordinateType::TaiwanVVM &&
        vertical.spacing_parameters_are_equal()) {
        configuration_error(
            "the current TaiwanVVM vertical-coordinate algorithm requires dz1 < dz; "
            "use type 'default' when dz and dz1 are equal.");
    }

    if (vertical.type == VerticalCoordinateType::RCEMIP && vertical.rcemip_grid_data_path.empty()) {
        configuration_error("RCEMIP vertical coordinates require a nonempty grid-data path.");
    }
}

VerticalGridSpec parse_structured_vertical(const Utils::ConfigurationManager& config) {
    VerticalGridSpec vertical;

    vertical.nz = config.get_value<int>("grid.vertical.nz");
    vertical.dz = config.get_value<VVM::Real>("grid.vertical.dz");
    vertical.dz1 = config.get_value<VVM::Real>("grid.vertical.dz1");
    vertical.type = parse_vertical_coordinate_type(config.get_value<std::string>("grid.vertical.type", "default"), "grid.vertical.type");
    vertical.rcemip_grid_data_path = config.get_value<std::string>("grid.vertical.rcemip_grid_data_path", "");

    validate_vertical_spec(vertical, "grid.vertical.nz", "grid.vertical.dz", "grid.vertical.dz1");

    return vertical;
}

VerticalGridSpec parse_legacy_vertical(const Utils::ConfigurationManager& config) {
    VerticalGridSpec vertical;

    vertical.nz = config.get_value<int>("grid.nz");
    vertical.dz = config.get_value<VVM::Real>("grid.dz");
    vertical.dz1 = config.get_value<VVM::Real>("grid.dz1");
    vertical.type = parse_vertical_coordinate_type(config.get_value<std::string>("grid.vertical_coordinate_type", "default"), "grid.vertical_coordinate_type");
    vertical.rcemip_grid_data_path = config.get_value<std::string>("grid.rcemip_grid_data_path", default_rcemip_grid_data_path);

    validate_vertical_spec(vertical, "grid.nz", "grid.dz", "grid.dz1");

    return vertical;
}

} // namespace

const char* horizontal_edge_topology_to_string(const HorizontalEdgeTopology topology) noexcept {

    switch (topology) {
        case HorizontalEdgeTopology::Periodic:
            return "periodic";

        case HorizontalEdgeTopology::Bounded:
            return "bounded";
    }

    return "unknown";
}

const char* vertical_coordinate_type_to_string(const VerticalCoordinateType type) noexcept {

    switch (type) {
        case VerticalCoordinateType::Default:
            return "default";

        case VerticalCoordinateType::TaiwanVVM:
            return "taiwanvvm";

        case VerticalCoordinateType::RCEMIP:
            return "rcemip";
    }

    return "unknown";
}

GridSpecification GridSpecification::from_config(const Utils::ConfigurationManager& config) {
    GridSpecification result;

    if (config.has_key("grid.horizontal")) {
        result.horizontal = parse_structured_horizontal(config);
    }
    else {
        result.horizontal = parse_legacy_horizontal(config);
    }

    if (config.has_key("grid.vertical")) {
        result.vertical = parse_structured_vertical(config);
    }
    else {
        result.vertical = parse_legacy_vertical(config);
    }

    return result;
}

} // namespace Core
} // namespace VVM
