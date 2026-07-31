#ifndef VVM_DYNAMICS_NUMERICAL_METHOD_HPP
#define VVM_DYNAMICS_NUMERICAL_METHOD_HPP

#include "dynamics/temporal_schemes/TimeIntegrator.hpp"
#include "dynamics/tendency_processes/TendencyCalculator.hpp"
#include "dynamics/tendency_processes/TendencyTerm.hpp"

#include <memory>
#include <string>
#include <vector>

namespace VVM {
namespace Dynamics {

enum class TemporalSchemeType {
    AdamsBashforth2,
    ForwardEuler,
    Multistage
};

struct ConfiguredTendency {
    TemporalSchemeType temporal_scheme;
    std::unique_ptr<TendencyTerm> term;
};

class NumericalMethod {
public:
    struct StateRequirements {
        bool previous_state = false;
        bool ab2_tendency_history = false;
        bool forward_euler_tendency = false;
    };

    NumericalMethod(
        std::string variable_name,
        std::vector<ConfiguredTendency> tendencies,
        std::unique_ptr<TemporalScheme> multistage_scheme,
        bool has_external_forward_euler = false);
    ~NumericalMethod();

    NumericalMethod(const NumericalMethod&) = delete;
    NumericalMethod& operator=(const NumericalMethod&) = delete;

    void calculate_tendencies(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params);

    void advance(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        VVM::Real dt,
        const TimeIntegrator::StageProcessor& process_stage = {}) const;

    bool uses_multistage_scheme() const {
        return integrator_->uses_multistage_scheme();
    }

    StateRequirements state_requirements() const {
        return {
            has_ab2_terms_ || has_forward_euler_terms_,
            has_ab2_terms_,
            has_forward_euler_terms_
        };
    }

private:
    std::string variable_name_;
    bool has_ab2_terms_ = false;
    bool has_forward_euler_terms_ = false;
    std::unique_ptr<TendencyCalculator> tendency_calculator_;
    std::unique_ptr<TimeIntegrator> integrator_;
};

} // namespace Dynamics
} // namespace VVM

#endif
