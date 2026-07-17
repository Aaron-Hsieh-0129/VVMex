#ifndef VVM_DYNAMICS_TENDENCY_CALCULATOR_HPP
#define VVM_DYNAMICS_TENDENCY_CALCULATOR_HPP

#include "TendencyTerm.hpp"
#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include <vector>
#include <memory>
#include <string>

namespace VVM {
namespace Dynamics {

class TendencyCalculator {
public:
    explicit TendencyCalculator(std::string var_name,
                                std::vector<std::unique_ptr<TendencyTerm>> ab2_terms,
                                std::vector<std::unique_ptr<TendencyTerm>> fe_terms,
                                std::vector<std::unique_ptr<TendencyTerm>> ssprk_terms = {});

    void calculate_tendencies(Core::State& state, const Core::Grid& grid, const Core::Parameters& params);
    Core::Field<3>& calculate_ssprk_tendency(
        Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, VVM::Real stage_dt);

private:
    std::string variable_name_;
    std::vector<std::unique_ptr<TendencyTerm>> ab2_tendency_terms_;
    std::vector<std::unique_ptr<TendencyTerm>> fe_tendency_terms_;
    std::vector<std::unique_ptr<TendencyTerm>> ssprk_tendency_terms_;

    std::unique_ptr<Core::Field<3>> temp_tendency_field_ = nullptr;
    std::unique_ptr<Core::Field<3>> ssprk_tendency_field_ = nullptr;
};

} // namespace Dynamics
} // namespace VVM
#endif
