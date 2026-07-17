#ifndef VVM_DYNAMICS_SSPRK2_HPP
#define VVM_DYNAMICS_SSPRK2_HPP

#include "TemporalScheme.hpp"
#include "core/Field.hpp"

#include <array>
#include <string>

namespace VVM {
namespace Dynamics {

class SSPRK2 final : public TemporalScheme {
public:
    SSPRK2(std::string var_name, const std::array<int, 3>& dimensions);

    void step(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        VVM::Real dt) const override;

    bool requires_tendency_recomputation() const override { return true; }

    void begin_multistage_step(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params) const override;

    void advance_multistage(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        const Core::Field<3>& tendency,
        VVM::Real dt,
        int stage) const override;

private:
    std::string variable_name_;
    mutable Core::Field<3> original_;
    mutable Core::Field<3> stage_one_;
};

} // namespace Dynamics
} // namespace VVM

#endif
