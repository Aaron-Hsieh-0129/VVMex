#ifndef VVM_DYNAMICS_TRACER_SOURCE_HPP
#define VVM_DYNAMICS_TRACER_SOURCE_HPP

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/vvm_types.hpp"

#include <string>
#include <vector>

namespace VVM {
namespace Dynamics {

class TracerSource {
public:
    TracerSource(const Core::Grid& grid, const Core::State& state);
    void apply(Core::State& state, VVM::Real dt) const;

    const std::vector<std::string>& get_target_vars() const {
        return target_vars_;
    }

private:
    const Core::Grid& grid_;
    std::vector<std::string> target_vars_;
};

} // namespace Dynamics
} // namespace VVM

#endif
