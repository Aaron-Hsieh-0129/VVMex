#include "io/RestartVariables.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace VVM {
namespace IO {

namespace {
std::string join_variable_names(const std::vector<std::string>& names) {
    if (names.empty()) return "(none)";

    std::ostringstream oss;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << names[i];
    }
    return oss.str();
}

std::vector<std::string> infer_3d_variables(
    const Utils::ConfigurationManager& config,
    const Core::State& state,
    int rank,
    const char* tag) {
    std::vector<std::string> result;
    std::unordered_set<std::string> output_fields;
    std::vector<std::string> skipped_output_fields;

    if (config.has_key("output.fields_to_output")) {
        for (const auto& name : config.get_value<std::vector<std::string>>("output.fields_to_output")) {
            output_fields.insert(name);
        }
    }

    auto prognostic_config = config.get_value<nlohmann::json>("dynamics.prognostic_variables");
    for (const auto& item : prognostic_config.items()) {
        const std::string& var_name = item.key();
        if (!state.has_field(var_name)) continue;
        if (!output_fields.empty() && output_fields.count(var_name) == 0) {
            skipped_output_fields.push_back(var_name);
            continue;
        }
        result.push_back(var_name);
    }

    // User tracers are prognostic even though they are deliberately not
    // duplicated under dynamics.prognostic_variables. In inferred mode they
    // are therefore always required from the restart file.
    for (const auto& tracer_name : state.get_tracer_names()) {
        if (std::find(result.begin(), result.end(), tracer_name) == result.end()) {
            result.push_back(tracer_name);
        }
    }

    // u/v/w are diagnostic rather than prognostic, but they are still needed for
    // a physically consistent restart. They are written to the restart/output
    // file, so include them explicitly.
    for (const auto& var_name : {"u", "v", "w"}) {
        if (!state.has_field(var_name)) continue;
        if (!output_fields.empty() && output_fields.count(var_name) == 0) {
            skipped_output_fields.push_back(var_name);
            continue;
        }
        if (std::find(result.begin(), result.end(), var_name) == result.end()) {
            result.push_back(var_name);
        }
    }

    if (rank == 0 && !skipped_output_fields.empty()) {
        std::cout << "  [" << tag << "] Skipping restart variables not listed in "
                     "output.fields_to_output: "
                  << join_variable_names(skipped_output_fields) << std::endl;
    }
    return result;
}
} // namespace

RestartVariables select_restart_variables(
    const Utils::ConfigurationManager& config,
    const Core::State& state,
    int rank,
    const char* tag) {
    RestartVariables variables;

    if (config.has_key("restart.variables_to_read.1d")) {
        variables.vars_1d =
            config.get_value<std::vector<std::string>>("restart.variables_to_read.1d");
    }
    if (config.has_key("restart.variables_to_read.2d")) {
        variables.vars_2d =
            config.get_value<std::vector<std::string>>("restart.variables_to_read.2d");
    }
    if (config.has_key("restart.variables_to_read.3d")) {
        variables.vars_3d =
            config.get_value<std::vector<std::string>>("restart.variables_to_read.3d");
    } else {
        variables.vars_3d = infer_3d_variables(config, state, rank, tag);
    }

    return variables;
}

void print_restart_variables(
    const RestartVariables& variables,
    const std::string& source,
    int rank,
    const char* tag) {
    if (rank != 0) return;

    std::cout << "  [" << tag << "] Restart variables to read from " << source << ":" << std::endl;
    std::cout << "    1D: " << join_variable_names(variables.vars_1d) << std::endl;
    std::cout << "    2D: " << join_variable_names(variables.vars_2d) << std::endl;
    std::cout << "    3D: " << join_variable_names(variables.vars_3d) << std::endl;
}

} // namespace IO
} // namespace VVM
