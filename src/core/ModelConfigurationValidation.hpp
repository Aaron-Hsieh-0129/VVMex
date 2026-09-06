#ifndef VVM_CORE_MODEL_CONFIGURATION_VALIDATION_HPP
#define VVM_CORE_MODEL_CONFIGURATION_VALIDATION_HPP

#include <stdexcept>

#include "core/GridSpecification.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/NumericalConfigurationValidation.hpp"

namespace VVM {
namespace Core {

// Model-startup validation using the same configuration precedence as Grid.
//
// This adapter belongs in core because GridSpecification belongs in core.
// The existing value-level validator remains independent in vvm_utils.
//
// MPI must already be initialized. Kokkos initialization and Grid construction
// are not required.
inline void validate_model_numerical_configuration(const Utils::ConfigurationManager& config, const int compute_ranks) {
    const GridSpecification specification = GridSpecification::from_config(config);
    const auto& horizontal = specification.horizontal;
    const auto& vertical = specification.vertical;

    // Do not pass angular increments into fields that still describe
    // Cartesian metre-valued spacing for full-model validation.
    // RLL Grid, geometry, and standalone operator/solver tests remain usable.
    if (horizontal.geometry.kind != Geometry::GeometryKind::Cartesian) {
        throw std::runtime_error(
            "Non-Cartesian full-model execution is not enabled yet. "
            "RLL Grid and component tests remain available, but the model's "
            "spacing-dependent dynamics and physical lateral boundaries still require migration.");
    }

    const Utils::NumericalConfigurationValues values{
        horizontal.nx,
        horizontal.ny,
        vertical.nz,
        horizontal.n_halo_cells,
        static_cast<double>(horizontal.geometry.dq1),
        static_cast<double>(horizontal.geometry.dq2),
        static_cast<double>(vertical.dz),
        config.get_value<double>("simulation.dt_s"),
        config.get_value<double>("simulation.total_time_s"),
        config.get_value<double>("simulation.output_interval_s")
    };

    Utils::validate_numerical_configuration_values(values, compute_ranks);
}

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_MODEL_CONFIGURATION_VALIDATION_HPP
