#include "TimeIntegrator.hpp"
#include "core/Field.hpp"
#include <stdexcept>

namespace VVM {
namespace Dynamics {

TimeIntegrator::TimeIntegrator(
    std::string var_name, bool has_ab2, bool has_fe,
    std::unique_ptr<TemporalScheme> multistage_scheme)
    : variable_name_(std::move(var_name)), has_ab2_terms_(has_ab2),
      has_fe_terms_(has_fe),
      multistage_scheme_(std::move(multistage_scheme)),
      prev_state_name_(variable_name_ + "_m"),
      hist_0_name_("d_" + variable_name_ + "_0"),
      hist_1_name_("d_" + variable_name_ + "_1"),
      fe_tendency_name_("fe_tendency_" + variable_name_) {}

TimeIntegrator::~TimeIntegrator() = default; 

void TimeIntegrator::step(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    VVM::Real dt) const {

    if (multistage_scheme_) {
        throw std::runtime_error(
            "Multistage integration for '" + variable_name_ +
            "' requires a tendency evaluator and stage processor.");
    }

    auto& field_to_update = var_ref_.get(state, variable_name_);
    auto& field_new_view = field_to_update.get_mutable_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    int k_start = h;
    int k_end = nz-h;
    if (variable_name_ == "xi" || variable_name_ == "eta") {
        k_end = nz-h-1;
    }

    if (has_ab2_terms_) {
        // Variable uses Adams-Bashforth (and possibly also Forward Euler)
        auto& field_prev_step = var_prev_ref_.get(state, prev_state_name_);

        auto& var_view = field_to_update.get_mutable_device_data();
        auto& var_m_view = field_prev_step.get_mutable_device_data();

        // swap previous and now
        auto temp_view = var_view;
        var_view = var_m_view;
        var_m_view = temp_view;
        auto field_new_view = var_view;
        auto field_old_view = var_m_view;

        size_t now_idx = state.get_step() % 2;
        size_t prev_idx = (state.get_step() + 1) % 2;
        
        auto& hist_0 = hist_0_ref_.get(state, hist_0_name_);
        auto& hist_1 = hist_1_ref_.get(state, hist_1_name_);
        auto hist_now  = (now_idx  == 0 ? hist_0 : hist_1).get_mutable_device_data();
        auto hist_prev = (prev_idx == 0 ? hist_0 : hist_1).get_mutable_device_data();

        const auto& ITYPEU = itypeu_ref_.get(state, "ITYPEU").get_device_data();
        const auto& ITYPEV = itypev_ref_.get(state, "ITYPEV").get_device_data();
        const auto& ITYPEW = itypew_ref_.get(state, "ITYPEW").get_device_data();
        const auto& max_topo_idx = params.max_topo_idx;
        if (variable_name_ == "xi") {
            Kokkos::parallel_for("topo",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx+1, ny-h, nx-h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    // Set tendency to 0 if ITYPEV = 0
                    if (ITYPEV(k,j,i) != 1) {
                        hist_now(k,j,i)  = real(0.);
                        hist_prev(k,j,i) = real(0.);
                    }
                }
            );
        }
        else if (variable_name_ == "eta") {
            Kokkos::parallel_for("topo",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx+1, ny-h, nx-h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    // Set tendency to 0 if ITYPEU = 0
                    if (ITYPEU(k,j,i) != 1) {
                        hist_now(k,j,i)  = real(0.);
                        hist_prev(k,j,i) = real(0.);
                    }
                }
            );
        }
        else {
            Kokkos::parallel_for("topo",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx+1, ny-h, nx-h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    // Set tendency to 0 if ITYPEW = 0
                    if (ITYPEW(k,j,i) != 1) {
                        hist_now(k,j,i)  = real(0.);
                        hist_prev(k,j,i) = real(0.);
                    }
                }
            );
        }

        if (state.get_step() == 0) {
            if (variable_name_ == "zeta") {
                Kokkos::parallel_for("AB2_Forward_Step", 
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
                    KOKKOS_LAMBDA(const int j, const int i) {
                        field_new_view(nz-h-1, j, i) = field_old_view(nz-h-1, j, i) + dt * hist_now(nz-h-1, j, i);
                    }
                );
            }
            else {
                Kokkos::parallel_for("AB2_Forward_Step", 
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny - h, nx - h}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        field_new_view(k, j, i) = field_old_view(k, j, i) + dt * hist_now(k, j, i);
                    }
                );
            }

        } 
        else {
            if (variable_name_ == "zeta") {
                Kokkos::parallel_for("AdamsBashforth2_Step", 
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
                    KOKKOS_LAMBDA(const int j, const int i) {
                        field_new_view(nz-h-1, j, i) = field_old_view(nz-h-1, j, i) 
                                                + dt * (real(1.5) * hist_now(nz-h-1, j, i) - real(0.5) * hist_prev(nz-h-1, j, i));
                    }
                );
            }
            else {
                Kokkos::parallel_for("AdamsBashforth2_Step", 
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        field_new_view(k, j, i) = field_old_view(k, j, i) 
                                                + dt * (real(1.5) * hist_now(k, j, i) - real(0.5) * hist_prev(k, j, i));
                    }
                );
            }
        }

        // --- Add Forward Euler tendencies on top of AB2 update if applicable ---
        if (has_fe_terms_) {
            const auto& fe_tendency_data = fe_tendency_ref_.get(state, fe_tendency_name_).get_device_data();

            Kokkos::parallel_for("Forward_Euler_FE_Terms_Additive",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    field_new_view(k, j, i) += dt * fe_tendency_data(k, j, i);
                }
            );
        }
    } 

    if (has_fe_terms_ && !has_ab2_terms_) {
        // Variable *only* uses Forward Euler
        auto& field_new_view = var_ref_.get(state, variable_name_).get_mutable_device_data();

        auto& fe_tendency_field = fe_tendency_ref_.get(state, fe_tendency_name_);
        auto fe_tendency_data = fe_tendency_field.get_device_data();
        
        Kokkos::parallel_for("Pure_Forward_Euler_Step",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                field_new_view(k, j, i) += dt * fe_tendency_data(k, j, i);
            }
        );
    }
}

void TimeIntegrator::step(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    VVM::Real dt,
    const TendencyEvaluator& evaluate_tendency,
    const StageProcessor& process_stage) const {
    if (!multistage_scheme_) {
        step(state, grid, params, dt);
        return;
    }
    if (!evaluate_tendency || !process_stage) {
        throw std::runtime_error(
            "Multistage integration callbacks are missing for '" +
            variable_name_ + "'.");
    }

    multistage_scheme_->begin_multistage_step(state, grid, params);

    const int stage_count = multistage_scheme_->stage_count();
    if (stage_count < 1) {
        throw std::runtime_error(
            "Multistage integration for '" + variable_name_ +
            "' configured an invalid stage count.");
    }

    for (int stage = 0; stage < stage_count; ++stage) {
        const VVM::Real stage_dt =
            multistage_scheme_->stage_timestep(dt, stage);
        Core::Field<3>& tendency = evaluate_tendency(stage_dt);
        multistage_scheme_->advance_multistage(
            state, grid, params, tendency, stage_dt, stage);
        process_stage();
    }
}

} // namespace Dynamics
} // namespace VVM
