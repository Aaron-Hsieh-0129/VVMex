#ifndef VVM_CORE_REGULAR_LAT_LON_MODEL_CONFIGURATION_HPP
#define VVM_CORE_REGULAR_LAT_LON_MODEL_CONFIGURATION_HPP

#include "core/GridSpecification.hpp"
#include "utils/ConfigurationManager.hpp"
#include <cmath>
#include <stdexcept>

namespace VVM::Core {
inline bool
is_jung2019_rll(const Utils::ConfigurationManager& config) {
    return config.get_value<std::string>("simulation.idealized_test", "none") ==
           "jung2019_barotropic";
}

inline bool
is_rll_mountain(const Utils::ConfigurationManager& config) {
    return config.get_value<std::string>("simulation.idealized_test", "none") == "rll_mountain";
}

inline bool
is_rll_idealized(const Utils::ConfigurationManager& config) {
    return is_jung2019_rll(config) || is_rll_mountain(config);
}

// Deliberately limited scientific configuration. This does not enable general
// dry RLL, moist physics, restart or Cartesian metre-based forcing. Terrain is
// admitted only by the separate rll_mountain experiment and its masked-curl path.
inline void
validate_jung2019_rll(const Utils::ConfigurationManager& config,
    const GridSpecification& specification) {
    const auto& h = specification.horizontal;
    const auto& v = specification.vertical;
    if (!is_rll_idealized(config) || h.geometry.kind != Geometry::GeometryKind::RegularLatLon ||
        h.topology.q1 != HorizontalEdgeTopology::Periodic ||
        h.topology.q2 != HorizontalEdgeTopology::Bounded || h.n_halo_cells < 2 || h.nx < 8 ||
        h.ny < 4 || v.nz < 3 || !v.uses_uniform_analytic_coordinate()) {
        throw std::runtime_error("Full-model RLL requires the supported Jung 2019 flat "
                                 "uniform-level periodic-longitude channel configuration.");
    }
    if (is_rll_mountain(config)) {
        const double height = config.get_value<double>("initial_conditions.rll_mountain.height_m");
        const double width =
            config.get_value<double>("initial_conditions.rll_mountain.half_width_m");
        const double south = h.geometry.regular_lat_lon.latitude_south_edge;
        const double north = south + h.geometry.dq2 * h.ny;
        const double center =
            config.get_value<double>("initial_conditions.rll_mountain.center_latitude_deg",
                (south + north) * 90. / std::acos(-1.)) *
            std::acos(-1.) / 180.;
        const double half_band =
            h.geometry.regular_lat_lon.radius * std::min(center - south, north - center);
        if (v.nz < 8 || !std::isfinite(height) || height < 0. ||
            height > (v.nz - h.n_halo_cells - 3) * v.dz || !std::isfinite(center) ||
            !std::isfinite(width) || width <= 0. || 3. * width >= half_band) {
            throw std::runtime_error(
                "RLL mountain requires nz >= 8, finite nonnegative height below the lid, and "
                "positive width with 3 widths inside the latitude walls.");
        }
    }
    else if (config.has_key("initial_conditions.rll_mountain")) {
        throw std::runtime_error("Mountain terrain requires simulation.idealized_test = "
                                 "rll_mountain; Jung reproduction remains flat.");
    }
    for (const char* key : {"physics.rrtmgp.enable_rrtmgp",
             "dynamics.forcings.sponge_layer.enable",
             "dynamics.forcings.areamn.enable",
             "dynamics.forcings.random_perturbation.enable",
             "dynamics.forcings.lateral_boundary_nudging.enable",
             "restart.enable"}) {
        if (config.get_value<bool>(key, false)) {
            throw std::runtime_error(std::string("Unsupported RLL option: ") + key);
        }
    }
    if (config.has_key("netcdf_reader.source_file") ||
        config.has_key("initial_conditions.source_file") || config.has_key("dynamics.tracers")) {
        throw std::runtime_error("Jung RLL uses analytic initial conditions; external input and "
                                 "tracers are not enabled.");
    }
    const auto engine = config.get_value<std::string>("output.engine", "HDF5");
    if ((engine != "HDF5" && !(is_rll_mountain(config) && engine == "BP5")) || h.fix_lonlat) {
        throw std::runtime_error(
            "Jung RLL currently requires HDF5 output and geometry-derived geographic coordinates.");
    }
    const int experiment = config.get_value<int>("initial_conditions.jung2019.case");
    if (experiment != 1 && experiment != 2) {
        throw std::runtime_error("Jung 2019 case must be 1 or 2.");
    }
    for (const char* key : {"initial_conditions.jung2019.jet_scale",
             "initial_conditions.jung2019.perturbation_scale"}) {
        const double scale = config.get_value<double>(key, 1.0);
        if (!std::isfinite(scale) || (scale != 0.0 && scale != 1.0)) {
            throw std::runtime_error("Jung RLL amplitudes must select the prescribed state (1) or "
                                     "its diagnostic removal (0).");
        }
    }
    if (config.get_value<std::string>("dynamics.solver.w_solver_method") != "tridiagonal" ||
        config.get_value<int>("dynamics.solver.iteration") <= 0 ||
        config.get_value<int>("dynamics.solver.initial_iterations") <= 0 ||
        config.get_value<int>("dynamics.solver.vertical_iterations") <= 0) {
        throw std::runtime_error("RLL requires positive fixed solver iteration counts and the "
                                 "original vertical line solver.");
    }
    const auto variables = config.get_value<nlohmann::json>("dynamics.prognostic_variables");
    if (is_rll_mountain(config)) {
        const double omega = config.get_value<double>("constants.OMEGA", 0.);
        if (!std::isfinite(omega)) {
            throw std::runtime_error("RLL mountain OMEGA must be finite.");
        }
        for (const char* name : {"xi", "eta", "zeta"}) {
            const std::string key =
                std::string("dynamics.prognostic_variables.") + name + ".tendency_terms.coriolis";
            const bool enabled =
                config.has_key(key) && config.get_value<bool>(key + ".enable", true);
            if (omega != 0. && !enabled) {
                throw std::runtime_error(
                    "Rotating RLL mountain requires Coriolis for all three vorticity components.");
            }
        }
    }
    for (const char* name : {"xi", "eta", "zeta"}) {
        if (!variables.contains(name) || !variables.at(name).contains("tendency_terms")) {
            throw std::runtime_error(
                "Jung RLL requires all three vorticity variables and their shared tendencies.");
        }
        const auto& terms = variables.at(name).at("tendency_terms");
        for (const char* term : {"advection", "stretching", "twisting"}) {
            if (!terms.contains(term) || !terms.at(term).value("enable", true)) {
                throw std::runtime_error("Jung RLL requires enabled advection, stretching and "
                                         "twisting for every vorticity variable.");
            }
        }
    }

    const bool p3_enabled = config.get_value<bool>("physics.p3.enable_p3", false);

    const auto is_vorticity_variable = [](const std::string& name) {
        return name == "xi" || name == "eta" || name == "zeta";
    };

    const auto is_p3_thermodynamic_variable = [](const std::string& name) {
        return name == "th" || name == "qv" || name == "qc" || name == "qr" || name == "qi" ||
               name == "qm" || name == "nc" || name == "nr" || name == "ni" || name == "bm";
    };

    for (const auto& variable : variables.items()) {
        const std::string& name = variable.key();

        const bool is_vorticity = is_vorticity_variable(name);

        const bool is_p3_scalar = p3_enabled && is_p3_thermodynamic_variable(name);

        if (!is_vorticity && !is_p3_scalar) {
            throw std::runtime_error("Unsupported RLL prognostic variable: " + name);
        }

        if (!variable.value().contains("tendency_terms")) {
            throw std::runtime_error(
                "RLL prognostic variable '" + name + "' requires tendency_terms.");
        }

        const auto& terms = variable.value().at("tendency_terms");

        // ---------------------------------------------------------------------
        // P3 thermodynamic variables
        //
        // RLL currently supports these as transported scalars. Keep this
        // deliberately narrow: advection only, using the validated RLL Takacs
        // scalar operator and the existing AB2 temporal integration.
        // ---------------------------------------------------------------------
        if (is_p3_scalar) {
            for (const auto& term : terms.items()) {
                if (!term.value().value("enable", true)) {
                    continue;
                }

                if (term.key() != "advection") {
                    throw std::runtime_error("RLL P3 thermodynamic variable '" + name +
                                             "' currently supports advection only.");
                }

                if (term.value().value("spatial_scheme", std::string("")) != "Takacs" ||
                    term.value().value("temporal_scheme", std::string("")) != "AdamsBashforth2") {

                    throw std::runtime_error("RLL P3 scalar transport requires "
                                             "Takacs spatial transport and AdamsBashforth2.");
                }
            }

            continue;
        }

        // ---------------------------------------------------------------------
        // Existing Jung/RLL vorticity validation
        // ---------------------------------------------------------------------
        for (const auto& term : terms.items()) {
            if (!term.value().value("enable", true)) {
                continue;
            }

            if (term.key() != "advection" && term.key() != "stretching" &&
                term.key() != "twisting" &&
                !(is_rll_mountain(config) && term.key() == "coriolis")) {

                throw std::runtime_error("RLL vorticity currently supports "
                                         "advection, stretching and twisting"
                                         " (plus Coriolis for the RLL mountain case).");
            }

            if (term.value().value("spatial_scheme", std::string("")) != "Takacs" ||
                term.value().value("temporal_scheme", std::string("")) != "AdamsBashforth2") {

                throw std::runtime_error("RLL vorticity requires Takacs transport "
                                         "and AdamsBashforth2.");
            }
        }
    }
}
} // namespace VVM::Core
#endif
