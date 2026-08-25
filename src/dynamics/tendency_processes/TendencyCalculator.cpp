#include "TendencyCalculator.hpp"
#include "core/Field.hpp"
#include <stdexcept>

namespace VVM {
namespace Dynamics {

TendencyCalculator::TendencyCalculator(std::string var_name,
                                       std::vector<std::unique_ptr<TendencyTerm>> ab2_terms,
                                       std::vector<std::unique_ptr<TendencyTerm>> fe_terms,
                                       std::vector<std::unique_ptr<TendencyTerm>> multistage_terms)
    : variable_name_(std::move(var_name)),
      ab2_tendency_terms_(std::move(ab2_terms)),
      fe_tendency_terms_(std::move(fe_terms)),
      multistage_tendency_terms_(std::move(multistage_terms)),
      hist_0_name_("d_" + variable_name_ + "_0"),
      hist_1_name_("d_" + variable_name_ + "_1"),
      fe_tendency_name_("fe_tendency_" + variable_name_) {}

void TendencyCalculator::calculate_tendencies(Core::State& state, const Core::Grid& grid, const Core::Parameters& params) {
    if (ab2_tendency_terms_.empty() && fe_tendency_terms_.empty()) {
        return;
    }

    const int& nz = grid.get_local_total_points_z();
    const int& ny = grid.get_local_total_points_y();
    const int& nx = grid.get_local_total_points_x();

    auto& field_to_update = var_ref_.get(state, variable_name_);
    auto& field_current_view = field_to_update.get_mutable_device_data();

    // Calculate AB2 tendencies
    if (!ab2_tendency_terms_.empty()) {
        if (!temp_tendency_field_) {
             temp_tendency_field_ = std::make_unique<Core::Field<3>>(
                 "temp_ab2_tendency_" + variable_name_, 
                 std::array<int, 3>{nz, ny, nx}
             );
        }

        const size_t now_idx = state.get_step() % 2;
        auto& current_tendency_field = (now_idx == 0)
            ? hist_0_ref_.get(state, hist_0_name_)
            : hist_1_ref_.get(state, hist_1_name_);
        current_tendency_field.set_to_zero();

        for (const auto& term : ab2_tendency_terms_) {
            term->compute_tendency(state, grid, params, current_tendency_field);
        }
    }

    // Calculate Forward Euler tendencies
    if (!fe_tendency_terms_.empty()) {
        auto& fe_tendency_field = fe_tendency_ref_.get(state, fe_tendency_name_);
        fe_tendency_field.set_to_zero();
        for (const auto& term : fe_tendency_terms_) {
            term->compute_tendency(state, grid, params, fe_tendency_field);
        }
    }
}

Core::Field<3>& TendencyCalculator::calculate_multistage_tendency(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    VVM::Real stage_dt) {
    if (multistage_tendency_terms_.empty()) {
        throw std::runtime_error(
            "No multistage tendency terms are configured for '" +
            variable_name_ + "'.");
    }
    if (!multistage_tendency_field_) {
        multistage_tendency_field_ = std::make_unique<Core::Field<3>>(
            "multistage_tendency_" + variable_name_,
            std::array<int, 3>{
                grid.get_local_total_points_z(),
                grid.get_local_total_points_y(),
                grid.get_local_total_points_x()});
    }
    multistage_tendency_field_->set_to_zero();
    for (const auto& term : multistage_tendency_terms_) {
        term->compute_stage_tendency(state, grid, params, *multistage_tendency_field_, stage_dt);
    }
    return *multistage_tendency_field_;
}

} // namespace Dynamics
} // namespace VVM
