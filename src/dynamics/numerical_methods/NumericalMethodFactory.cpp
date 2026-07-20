#include "NumericalMethodFactory.hpp"

#include "dynamics/spatial_schemes/MUSCL.hpp"
#include "dynamics/spatial_schemes/Takacs.hpp"
#include "dynamics/temporal_schemes/SSPRK2.hpp"
#include "dynamics/tendency_processes/AdvectionTerm.hpp"
#include "dynamics/tendency_processes/BuoyancyTerm.hpp"
#include "dynamics/tendency_processes/CoriolisTerm.hpp"
#include "dynamics/tendency_processes/StretchingTerm.hpp"
#include "dynamics/tendency_processes/TwistingTerm.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace VVM {
namespace Dynamics {

TemporalSchemeType NumericalMethodFactory::parse_temporal_scheme(
    const std::string& variable_name,
    const std::string& term_name,
    const std::string& scheme_name,
    bool is_tracer) const {
    if (scheme_name == "AdamsBashforth2") {
        return TemporalSchemeType::AdamsBashforth2;
    }
    if (scheme_name == "ForwardEuler") {
        return TemporalSchemeType::ForwardEuler;
    }
    if (scheme_name == "SSPRK2") {
        return TemporalSchemeType::Multistage;
    }

    if (is_tracer) {
        throw std::runtime_error(
            "Tracer '" + variable_name + "', tendency term '" + term_name +
            "': unsupported temporal method '" + scheme_name + "'.");
    }
    throw std::runtime_error("Unknown temporal scheme: " + scheme_name);
}

std::unique_ptr<SpatialScheme>
NumericalMethodFactory::create_spatial_scheme(
    const std::string& variable_name,
    const std::string& term_name,
    const nlohmann::json& term_config,
    const std::string& scheme_name,
    bool is_tracer) const {
    if (scheme_name == "Takacs") {
        return std::make_unique<Takacs>(
            config_, grid_, halo_exchanger_, bc_manager_);
    }
    if (scheme_name == "MUSCL") {
        return std::make_unique<MUSCL>(
            variable_name, term_config, grid_);
    }

    if (is_tracer) {
        throw std::runtime_error(
            "Tracer '" + variable_name + "', tendency term '" + term_name +
            "': unsupported spatial method '" + scheme_name + "'.");
    }
    throw std::runtime_error("Unknown spatial scheme: " + scheme_name);
}

std::unique_ptr<TendencyTerm>
NumericalMethodFactory::create_tendency_term(
    const std::string& variable_name,
    const std::string& term_name,
    std::unique_ptr<SpatialScheme> spatial_scheme,
    bool normalize_anelastic_scalar) const {
    if (term_name == "advection") {
        return std::make_unique<AdvectionTerm>(std::move(spatial_scheme), variable_name, halo_exchanger_, bc_manager_, normalize_anelastic_scalar);
    }
    if (term_name == "stretching") {
        return std::make_unique<StretchingTerm>(std::move(spatial_scheme), variable_name, halo_exchanger_);
    }
    if (term_name == "twisting") {
        return std::make_unique<TwistingTerm>(std::move(spatial_scheme), variable_name, halo_exchanger_);
    }
    if (term_name == "buoyancy") {
        return std::make_unique<BuoyancyTerm>(std::move(spatial_scheme), variable_name, halo_exchanger_);
    }
    if (term_name == "coriolis") {
        return std::make_unique<CoriolisTerm>(std::move(spatial_scheme), variable_name, halo_exchanger_);
    }
    throw std::runtime_error(
        "Unknown tendency term '" + term_name +
        "' for prognostic variable '" + variable_name + "'.");
}

std::unique_ptr<NumericalMethod> NumericalMethodFactory::create(
    const std::string& variable_name,
    const nlohmann::json& variable_config,
    bool is_tracer,
    bool is_thermodynamic,
    const std::array<int, 3>& dimensions,
    bool has_external_forward_euler) const {
    std::vector<ConfiguredTendency> configured_tendencies;
    std::unique_ptr<TemporalScheme> multistage_scheme;

    size_t enabled_tendency_count = 0;
    if (variable_config.contains("tendency_terms")) {
        for (const auto& item :
             variable_config.at("tendency_terms").items()) {
            if (item.value().value("enable", true)) {
                ++enabled_tendency_count;
            }
        }
    }

    if (variable_config.contains("tendency_terms")) {
        for (const auto& item :
             variable_config.at("tendency_terms").items()) {
            const std::string& term_name = item.key();
            const auto& term_config = item.value();

            if (!term_config.value("enable", true)) {
                if (grid_.get_mpi_rank() == 0) {
                    std::cout << "    - [Disabled] Tendency term: "
                              << term_name << " is skipped." << std::endl;
                }
                continue;
            }

            if (is_tracer && term_name != "advection") {
                throw std::runtime_error(
                    "Tracer '" + variable_name +
                    "' has unsupported tendency term '" + term_name +
                    "'; passive tracers currently support advection only.");
            }

            const std::string spatial_scheme_name = term_config.at("spatial_scheme");
            const std::string temporal_scheme_name = term_config.value("temporal_scheme", "AdamsBashforth2");
            const TemporalSchemeType temporal_scheme = parse_temporal_scheme(variable_name, term_name, temporal_scheme_name, is_tracer);

            if (grid_.get_mpi_rank() == 0) {
                std::cout << "    - Tendency term: " << term_name
                          << " | Temporal Scheme: "
                          << temporal_scheme_name
                          << " | Spatial Scheme: "
                          << spatial_scheme_name << std::endl;
            }

            const bool requests_muscl = spatial_scheme_name == "MUSCL";
            const bool requests_ssprk2 = temporal_scheme == TemporalSchemeType::Multistage;
            if (requests_muscl || requests_ssprk2) {
                if (!is_thermodynamic || term_name != "advection") {
                    throw std::runtime_error(
                        "Field '" + variable_name +
                        "' requested spatial scheme '" +
                        spatial_scheme_name + "' and temporal scheme '" +
                        temporal_scheme_name +
                        "'; supported use is advection of a configured "
                        "tracer or thermodynamic scalar with spatial scheme "
                        "'MUSCL' and temporal scheme 'SSPRK2'.");
                }
                (void) MUSCL::validate_configuration(
                    variable_name, spatial_scheme_name,
                    temporal_scheme_name, term_config,
                    enabled_tendency_count, grid_.get_halo_cells());
            }
            if (requests_ssprk2 && !multistage_scheme) {
                multistage_scheme =
                    std::make_unique<SSPRK2>(variable_name, dimensions);
            }

            auto spatial_scheme = create_spatial_scheme(
                variable_name, term_name, term_config,
                spatial_scheme_name, is_tracer);
            const bool normalize_anelastic_scalar =
                is_thermodynamic &&
                spatial_scheme->
                    produces_anelastic_scalar_flux_divergence();
            auto tendency_term = create_tendency_term(
                variable_name, term_name, std::move(spatial_scheme),
                normalize_anelastic_scalar);
            configured_tendencies.push_back({
                temporal_scheme, std::move(tendency_term)});
        }
    }

    return std::make_unique<NumericalMethod>(
        variable_name, std::move(configured_tendencies),
        std::move(multistage_scheme), has_external_forward_euler);
}

} // namespace Dynamics
} // namespace VVM
