#ifndef VVM_CORE_REGULAR_LAT_LON_MODEL_CONFIGURATION_HPP
#define VVM_CORE_REGULAR_LAT_LON_MODEL_CONFIGURATION_HPP

#include "core/GridSpecification.hpp"
#include "utils/ConfigurationManager.hpp"
#include <cmath>
#include <stdexcept>

namespace VVM::Core {
inline bool is_jung2019_rll(const Utils::ConfigurationManager& config) {
    return config.get_value<std::string>("simulation.idealized_test", "none") == "jung2019_barotropic";
}

// Deliberately limited scientific configuration. This does not enable general
// dry RLL, moist physics, restart, terrain or Cartesian metre-based forcing.
inline void validate_jung2019_rll(const Utils::ConfigurationManager& config, const GridSpecification& specification) {
    const auto& h = specification.horizontal;
    const auto& v = specification.vertical;
    if (!is_jung2019_rll(config) || h.geometry.kind != Geometry::GeometryKind::RegularLatLon
        || h.topology.q1 != HorizontalEdgeTopology::Periodic || h.topology.q2 != HorizontalEdgeTopology::Bounded
        || h.n_halo_cells < 2 || h.nx < 8 || h.ny < 4 || v.nz < 3 || !v.uses_uniform_analytic_coordinate())
        throw std::runtime_error("Full-model RLL requires the supported Jung 2019 flat uniform-level periodic-longitude channel configuration.");
    for (const char* key : {"physics.p3.enable_p3", "physics.turbulence.enable_turbulence", "physics.rrtmgp.enable_rrtmgp",
         "physics.surface_process.enable", "dynamics.forcings.sponge_layer.enable", "dynamics.forcings.areamn.enable",
         "dynamics.forcings.random_perturbation.enable", "dynamics.forcings.lateral_boundary_nudging.enable", "restart.enable"})
        if (config.get_value<bool>(key, false)) throw std::runtime_error(std::string("Unsupported RLL option: ") + key);
    if (config.has_key("netcdf_reader.source_file") || config.has_key("initial_conditions.source_file") || config.has_key("dynamics.tracers"))
        throw std::runtime_error("Jung RLL uses analytic initial conditions; external input and tracers are not enabled.");
    if (config.get_value<std::string>("output.engine", "HDF5") != "HDF5" || h.fix_lonlat)
        throw std::runtime_error("Jung RLL currently requires HDF5 output and geometry-derived geographic coordinates.");
    const int experiment = config.get_value<int>("initial_conditions.jung2019.case");
    if (experiment != 1 && experiment != 2) throw std::runtime_error("Jung 2019 case must be 1 or 2.");
    for (const char* key : {"initial_conditions.jung2019.jet_scale", "initial_conditions.jung2019.perturbation_scale"}) {
        const double scale = config.get_value<double>(key, 1.0);
        if (!std::isfinite(scale) || (scale != 0.0 && scale != 1.0))
            throw std::runtime_error("Jung RLL amplitudes must select the prescribed state (1) or its diagnostic removal (0).");
    }
    if (config.get_value<std::string>("dynamics.solver.w_solver_method") != "tridiagonal"
        || config.get_value<int>("dynamics.solver.iteration") <= 0
        || config.get_value<int>("dynamics.solver.initial_iterations") <= 0
        || config.get_value<int>("dynamics.solver.vertical_iterations") <= 0)
        throw std::runtime_error("RLL requires positive fixed solver iteration counts and the original vertical line solver.");
    const auto variables = config.get_value<nlohmann::json>("dynamics.prognostic_variables");
    for (const char* name : {"xi", "eta", "zeta"}) {
        if (!variables.contains(name) || !variables.at(name).contains("tendency_terms"))
            throw std::runtime_error("Jung RLL requires all three vorticity variables and their shared tendencies.");
        const auto& terms = variables.at(name).at("tendency_terms");
        for (const char* term : {"advection", "stretching", "twisting"})
            if (!terms.contains(term) || !terms.at(term).value("enable", true))
                throw std::runtime_error("Jung RLL requires enabled advection, stretching and twisting for every vorticity variable.");
    }
    for (const auto& variable : variables.items()) {
        if (variable.key() != "xi" && variable.key() != "eta" && variable.key() != "zeta")
            throw std::runtime_error("Section 4.2 enables only xi, eta and zeta tendencies.");
        for (const auto& term : variable.value().at("tendency_terms").items()) {
            if (!term.value().value("enable", true)) continue;
            if (term.key() != "advection" && term.key() != "stretching" && term.key() != "twisting")
                throw std::runtime_error("Section 4.2 has no Coriolis, buoyancy, or diffusion tendency.");
            if (term.value().value("spatial_scheme", std::string("")) != "Takacs"
                || term.value().value("temporal_scheme", std::string("")) != "AdamsBashforth2")
                throw std::runtime_error("Section 4.2 requires Takacs transport and AdamsBashforth2.");
        }
    }
}
} // namespace VVM::Core
#endif
