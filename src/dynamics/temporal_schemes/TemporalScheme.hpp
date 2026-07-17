#ifndef VVM_DYNAMICS_TEMPORAL_SCHEME_HPP
#define VVM_DYNAMICS_TEMPORAL_SCHEME_HPP

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "dynamics/tendency_processes/TendencyTerm.hpp"
#include <vector>
#include <memory>
#include <stdexcept>

namespace VVM {
namespace Dynamics {

class TemporalScheme {
public:
    virtual ~TemporalScheme() = default;

    virtual void step(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        VVM::Real dt
    ) const = 0;

    virtual std::vector<std::string> get_required_state_suffixes() const {
        return {};
    }

    virtual bool requires_tendency_recomputation() const { return false; }

    virtual void begin_multistage_step(
        Core::State&, const Core::Grid&, const Core::Parameters&) const {
        throw std::runtime_error(
            "Temporal scheme does not implement multistage initialization.");
    }

    virtual void advance_multistage(
        Core::State&, const Core::Grid&, const Core::Parameters&,
        const Core::Field<3>&, VVM::Real, int) const {
        throw std::runtime_error(
            "Temporal scheme does not implement multistage advancement.");
    }
};

} // namespace Dynamics
} // namespace VVM
#endif
