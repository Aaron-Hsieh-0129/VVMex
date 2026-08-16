#ifndef VVM_IO_RESTART_VARIABLES_HPP
#define VVM_IO_RESTART_VARIABLES_HPP

#include <string>
#include <vector>

#include "core/State.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace IO {

// Which fields a restart has to recover, by rank. Shared by every restart
// reader so that a run restarted from BP5 loads exactly what the same run
// restarted from HDF5 would.
struct RestartVariables {
    std::vector<std::string> vars_1d;
    std::vector<std::string> vars_2d;
    std::vector<std::string> vars_3d;

    bool empty() const {
        return vars_1d.empty() && vars_2d.empty() && vars_3d.empty();
    }
};

// restart.variables_to_read.{1d,2d,3d} when configured; otherwise the 3-D set is
// inferred from the prognostic variables, the tracers, and u/v/w, restricted to
// what the run actually writes. `tag` names the caller in log lines.
RestartVariables select_restart_variables(
    const Utils::ConfigurationManager& config,
    const Core::State& state,
    int rank,
    const char* tag);

void print_restart_variables(
    const RestartVariables& variables,
    const std::string& source,
    int rank,
    const char* tag);

} // namespace IO
} // namespace VVM

#endif
