#ifndef VVM_CORE_GRID_SPECIFICATION_HPP
#define VVM_CORE_GRID_SPECIFICATION_HPP

#include <cstdint>
#include <string>

#include "core/geometry/HorizontalGridSpec.hpp"
#include "core/vvm_types.hpp"

namespace VVM {
namespace Utils {
class ConfigurationManager;
}

namespace Core {

enum class HorizontalEdgeTopology : std::uint8_t {
    Periodic,
    Bounded
};

enum class VerticalCoordinateType : std::uint8_t {
    Default,
    TaiwanVVM,
    RCEMIP
};

const char* horizontal_edge_topology_to_string(HorizontalEdgeTopology topology) noexcept;
const char* vertical_coordinate_type_to_string(VerticalCoordinateType type) noexcept;

struct HorizontalTopologySpec {
    HorizontalEdgeTopology q1 = HorizontalEdgeTopology::Periodic;
    HorizontalEdgeTopology q2 = HorizontalEdgeTopology::Periodic;
};

struct HorizontalDomainSpec {
    int nx = 0;
    int ny = 0;
    int n_halo_cells = 0;

    bool fix_lonlat = false;

    Geometry::HorizontalGridSpec geometry;
    HorizontalTopologySpec topology;
};

struct VerticalGridSpec {
    int nz = 0;
    VVM::Real dz = VVM::real(0.0);
    VVM::Real dz1 = VVM::real(0.0);

    VerticalCoordinateType type = VerticalCoordinateType::Default;
    std::string rcemip_grid_data_path;

    bool spacing_parameters_are_equal() const noexcept {
        return dz == dz1;
    }

    bool uses_uniform_analytic_coordinate() const noexcept {
        return type == VerticalCoordinateType::Default && spacing_parameters_are_equal();
    }

    bool uses_default_stretching() const noexcept {
        return type == VerticalCoordinateType::Default && !spacing_parameters_are_equal();
    }
};

struct GridSpecification {
    HorizontalDomainSpec horizontal;
    VerticalGridSpec vertical;

    static GridSpecification from_config(const VVM::Utils::ConfigurationManager& config);
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GRID_SPECIFICATION_HPP
