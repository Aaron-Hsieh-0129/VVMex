#include "NumericalMethod.hpp"

#include <stdexcept>
#include <utility>

namespace VVM {
namespace Dynamics {

NumericalMethod::NumericalMethod(
    std::string variable_name,
    std::vector<ConfiguredTendency> tendencies,
    std::unique_ptr<TemporalScheme> multistage_scheme,
    bool has_external_forward_euler)
    : variable_name_(std::move(variable_name)),
      has_forward_euler_terms_(has_external_forward_euler) {
    std::vector<std::unique_ptr<TendencyTerm>> ab2_terms;
    std::vector<std::unique_ptr<TendencyTerm>> forward_euler_terms;
    std::vector<std::unique_ptr<TendencyTerm>> multistage_terms;

    bool has_multistage_terms = false;
    for (auto& configured : tendencies) {
        if (!configured.term) {
            throw std::runtime_error(
                "Numerical method for '" + variable_name_ +
                "' received an empty tendency term.");
        }

        switch (configured.temporal_scheme) {
        case TemporalSchemeType::AdamsBashforth2:
            has_ab2_terms_ = true;
            ab2_terms.push_back(std::move(configured.term));
            break;
        case TemporalSchemeType::ForwardEuler:
            has_forward_euler_terms_ = true;
            forward_euler_terms.push_back(std::move(configured.term));
            break;
        case TemporalSchemeType::Multistage:
            has_multistage_terms = true;
            multistage_terms.push_back(std::move(configured.term));
            break;
        }
    }

    if (has_multistage_terms &&
        (has_ab2_terms_ || has_forward_euler_terms_)) {
        throw std::runtime_error(
            "Advected variable '" + variable_name_ +
            "' cannot combine a multistage temporal scheme with "
            "AdamsBashforth2 or ForwardEuler.");
    }

    tendency_calculator_ = std::make_unique<TendencyCalculator>(
        variable_name_, std::move(ab2_terms),
        std::move(forward_euler_terms), std::move(multistage_terms));

    if (has_multistage_terms != static_cast<bool>(multistage_scheme)) {
        throw std::runtime_error(
            "Numerical method for '" + variable_name_ +
            "' has inconsistent multistage tendency and temporal "
            "scheme configuration.");
    }
    integrator_ = std::make_unique<TimeIntegrator>(
        variable_name_, has_ab2_terms_, has_forward_euler_terms_,
        std::move(multistage_scheme));
}

NumericalMethod::~NumericalMethod() = default;

void NumericalMethod::calculate_tendencies(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params) {
    tendency_calculator_->calculate_tendencies(state, grid, params);
}

void NumericalMethod::advance(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    VVM::Real dt,
    const TimeIntegrator::StageProcessor& process_stage) const {
    if (!uses_multistage_scheme()) {
        integrator_->step(state, grid, params, dt);
        return;
    }

    auto evaluate_tendency = [&](VVM::Real stage_dt) -> Core::Field<3>& {
        return tendency_calculator_->calculate_multistage_tendency(
            state, grid, params, stage_dt);
    };
    integrator_->step(
        state, grid, params, dt, evaluate_tendency, process_stage);
}

} // namespace Dynamics
} // namespace VVM

