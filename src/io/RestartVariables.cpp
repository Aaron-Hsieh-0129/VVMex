#include "io/RestartVariables.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace VVM {
namespace IO {

namespace {

std::string
join_variable_names(const std::vector<std::string>& names) {
    if (names.empty()) {
        return "(none)";
    }

    std::ostringstream oss;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << names[i];
    }
    return oss.str();
}

void
append_unique_if_present(
    std::vector<std::string>& names, const Core::State& state, const std::string& name) {
    if (!state.has_field(name)) {
        return;
    }

    if (std::find(names.begin(), names.end(), name) == names.end()) {
        names.push_back(name);
    }
}

void
append_unique(std::vector<std::string>& names, const std::string& name) {
    if (std::find(names.begin(), names.end(), name) == names.end()) {
        names.push_back(name);
    }
}

} // namespace

RestartVariables
infer_restart_variables(const Utils::ConfigurationManager& config, const Core::State& state) {

    RestartVariables variables;

    // -------------------------------------------------------------
    // Atmospheric prognostic state
    // -------------------------------------------------------------
    //
    // Restart files store physical components. Generalized-coordinate
    // contravariant shadow state is rebuilt after loading.
    const auto prognostic_config =
        config.get_value<nlohmann::json>("dynamics.prognostic_variables");

    for (const auto& item : prognostic_config.items()) {
        append_unique_if_present(variables.vars_3d, state, item.key());
    }

    for (const auto& tracer_name : state.get_tracer_names()) {
        append_unique_if_present(variables.vars_3d, state, tracer_name);
    }

    // Physical wind must also be checkpointed.
    //
    // In particular, RLL vorticity alone does not determine the harmonic
    // circulation component of the horizontal wind.
    for (const char* name : {"u", "v", "w"}) {
        append_unique_if_present(variables.vars_3d, state, name);
    }

    // Both Cartesian and RLL wind solvers extrapolate the top-boundary
    // potentials and vertical wind between solves. These histories are stored
    // automatically in the output file's restart/ namespace.
    for (const char* name : {"psi", "psinm1", "chi", "chinm1"}) {
        append_unique_if_present(variables.vars_2d, state, name);
    }
    append_unique_if_present(variables.vars_3d, state, "W3DNM1");

    // -------------------------------------------------------------
    // P3
    // -------------------------------------------------------------
    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        for (const char* name : {"qc", "qr", "qi", "qm", "nc", "nr", "ni", "bm"}) {
            append_unique_if_present(variables.vars_3d, state, name);
        }

        // These accumulate between output times and therefore are persistent
        // state, not merely diagnostics.
        for (const char* name : {"precip_liq_surf_mass", "precip_ice_surf_mass"}) {
            append_unique_if_present(variables.vars_2d, state, name);
        }
    }

    // -------------------------------------------------------------
    // RRTMGP
    // -------------------------------------------------------------
    //
    // net_heating is held between radiation calls.
    //
    // Surface radiative fluxes are additionally consumed by Noah between
    // radiation calls.
    if (config.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false)) {
        append_unique_if_present(variables.vars_3d, state, "net_heating");

        for (const char* name : {"swdn_sfc", "swup_sfc", "lwdn_sfc", "lwup_sfc"}) {
            append_unique_if_present(variables.vars_2d, state, name);
        }
    }

    // -------------------------------------------------------------
    // Surface process
    // -------------------------------------------------------------
    //
    // These fluxes are held and reused between calls to
    // compute_coefficients().
    if (config.get_value<bool>("physics.surface_process.enable", false)) {
        for (const char* name :
            {"sfc_flux_th", "sfc_flux_qv", "sfc_flux_u", "sfc_flux_v", "VEN2D"}) {
            append_unique_if_present(variables.vars_2d, state, name);
        }
    }

    // -------------------------------------------------------------
    // Noah LSM
    // -------------------------------------------------------------
    const std::string land_scheme =
        config.get_value<std::string>("physics.surface_process.land_scheme", "none");

    if (config.get_value<bool>("physics.surface_process.enable", false) &&
        land_scheme == "noahlsm") {

        for (const char* name : {"Tg",
                 "hfx",
                 "le",
                 "st1",
                 "st2",
                 "st3",
                 "st4",
                 "sm1",
                 "sm2",
                 "sm3",
                 "sm4",
                 "sl1",
                 "sl2",
                 "sl3",
                 "sl4",
                 "canopy",
                 "snwdph",
                 "sneqv",
                 "zorl",
                 "cmx",
                 "chx",
                 "sfemis",
                 "albedo",
                 "lai"}) {
            append_unique_if_present(variables.vars_2d, state, name);
        }
    }
    return variables;
}

RestartVariables
select_restart_variables(const Utils::ConfigurationManager& config,
    const Core::State& state,
    int rank,
    const char* tag) {

    RestartVariables variables = infer_restart_variables(config, state);

    const bool explicit_1d = config.has_key("restart.variables_to_read.1d");
    const bool explicit_2d = config.has_key("restart.variables_to_read.2d");
    const bool explicit_3d = config.has_key("restart.variables_to_read.3d");

    if (explicit_1d) {
        for (const auto& name :
            config.get_value<std::vector<std::string>>("restart.variables_to_read.1d")) {
            append_unique(variables.vars_1d, name);
        }
    }

    if (explicit_2d) {
        for (const auto& name :
            config.get_value<std::vector<std::string>>("restart.variables_to_read.2d")) {
            append_unique(variables.vars_2d, name);
        }
    }

    if (explicit_3d) {
        for (const auto& name :
            config.get_value<std::vector<std::string>>("restart.variables_to_read.3d")) {
            append_unique(variables.vars_3d, name);
        }
    }

    if (rank == 0 && (explicit_1d || explicit_2d || explicit_3d)) {
        std::cout << "  [" << tag << "] "
                  << "restart.variables_to_read adds extra fields; "
                  << "required restart state remains mandatory." << std::endl;
    }

    return variables;
}

std::vector<std::string>
append_restart_output_fields(const Utils::ConfigurationManager& config,
    const Core::State& state,
    std::vector<std::string> fields_to_output) {

    if (!config.get_value<bool>("output.restart_capable", false)) {
        return fields_to_output;
    }

    const RestartVariables restart = infer_restart_variables(config, state);

    for (const auto& name : restart.vars_1d) {
        append_unique(fields_to_output, name);
    }
    for (const auto& name : restart.vars_2d) {
        append_unique(fields_to_output, name);
    }
    for (const auto& name : restart.vars_3d) {
        append_unique(fields_to_output, name);
    }
    return fields_to_output;
}

void
print_restart_variables(
    const RestartVariables& variables, const std::string& source, int rank, const char* tag) {
    if (rank != 0) {
        return;
    }

    std::cout << "  [" << tag << "] "
              << "Restart variables to read from " << source << ":" << std::endl;
    std::cout << "    1D: " << join_variable_names(variables.vars_1d) << std::endl;
    std::cout << "    2D: " << join_variable_names(variables.vars_2d) << std::endl;
    std::cout << "    3D: " << join_variable_names(variables.vars_3d) << std::endl;
}

} // namespace IO
} // namespace VVM
