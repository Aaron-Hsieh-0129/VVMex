#ifndef VVM_IO_RESTART_VARIABLES_HPP
#define VVM_IO_RESTART_VARIABLES_HPP

#include <string>
#include <vector>

#include "core/State.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace IO {

// Physical/persistent State fields that must survive a restart.
// Generalized-coordinate shadow fields are rebuilt after the physical state is
// restored. Wind-solver history is persistent and must be checkpointed.
struct RestartVariables {
    std::vector<std::string> vars_1d;
    std::vector<std::string> vars_2d;
    std::vector<std::string> vars_3d;

    bool
    empty() const {
        return vars_1d.empty() && vars_2d.empty() && vars_3d.empty();
    }
};

RestartVariables infer_restart_variables(const Utils::ConfigurationManager& config,
    const Core::State& state);

// Explicit restart.variables_to_read entries may add extra fields, but may not
// remove state required by the active model configuration.
RestartVariables select_restart_variables(
    const Utils::ConfigurationManager& config, const Core::State& state, int rank, const char* tag);

// When output.restart_capable=true, automatically append every field required
// by the restart contract to the normal output field list.
std::vector<std::string> append_restart_output_fields(const Utils::ConfigurationManager& config,
    const Core::State& state,
    std::vector<std::string> fields_to_output);

void print_restart_variables(
    const RestartVariables& variables, const std::string& source, int rank, const char* tag);

} // namespace IO
} // namespace VVM

#endif
