#ifndef VVM_DYNAMICS_NUMERICAL_METHOD_FACTORY_HPP
#define VVM_DYNAMICS_NUMERICAL_METHOD_FACTORY_HPP

#include "NumericalMethod.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "core/Grid.hpp"
#include "core/HaloExchanger.hpp"
#include "utils/ConfigurationManager.hpp"

#include <array>
#include <memory>
#include <string>

namespace VVM {
namespace Dynamics {

class SpatialScheme;

class NumericalMethodFactory {
public:
    NumericalMethodFactory(
        const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        Core::HaloExchanger& halo_exchanger,
        const Core::BoundaryConditionManager& bc_manager)
        : config_(config), grid_(grid),
          halo_exchanger_(halo_exchanger), bc_manager_(bc_manager) {}

    std::unique_ptr<NumericalMethod> create(
        const std::string& variable_name,
        const nlohmann::json& variable_config,
        bool is_tracer,
        bool is_thermodynamic,
        const std::array<int, 3>& dimensions,
        bool has_external_forward_euler = false) const;

private:
    TemporalSchemeType parse_temporal_scheme(
        const std::string& variable_name,
        const std::string& term_name,
        const std::string& scheme_name,
        bool is_tracer) const;

    std::unique_ptr<SpatialScheme> create_spatial_scheme(
        const std::string& variable_name,
        const std::string& term_name,
        const nlohmann::json& term_config,
        const std::string& scheme_name,
        bool is_tracer) const;

    std::unique_ptr<TendencyTerm> create_tendency_term(
        const std::string& variable_name,
        const std::string& term_name,
        std::unique_ptr<SpatialScheme> spatial_scheme,
        bool normalize_anelastic_scalar) const;

    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    Core::HaloExchanger& halo_exchanger_;
    const Core::BoundaryConditionManager& bc_manager_;
};

} // namespace Dynamics
} // namespace VVM

#endif
