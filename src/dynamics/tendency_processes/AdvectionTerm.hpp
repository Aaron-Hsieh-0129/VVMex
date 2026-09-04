#ifndef VVM_DYNAMICS_ADVECTION_TERM_HPP
#define VVM_DYNAMICS_ADVECTION_TERM_HPP

#include "TendencyTerm.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/BoundaryConditionManager.hpp"
#include <memory>
#include <string>
#include <vector>

namespace VVM {
namespace Dynamics {

// Scalar advections within a step share identical density-weighted wind fields.
class MeanWindState {
public:
    enum class Variant { None, Xi, Eta, Zeta, Scalar };

    bool holds(Variant variant, size_t step) const {
        return variant_ != Variant::None && variant_ == variant && step_ == step;
    }
    void set(Variant variant, size_t step) { variant_ = variant; step_ = step; }
    void invalidate() { variant_ = Variant::None; }

private:
    Variant variant_ = Variant::None;
    size_t step_ = 0;
};

class AdvectionTerm : public TendencyTerm {
public:
    AdvectionTerm(
        std::unique_ptr<SpatialScheme> scheme,
        std::string var_name,
        VVM::Core::HaloExchanger& halo_exchanger,
        const Core::BoundaryConditionManager& bc_manager,
        std::shared_ptr<MeanWindState> mean_wind_state,
        bool force_anelastic_scalar_normalization = false);
    ~AdvectionTerm() override;

    void compute_tendency(
        Core::State& state, 
        const Core::Grid& grid,
        const Core::Parameters& params, 
        Core::Field<3>& out_tendency) const override;
    void compute_stage_tendency(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency,
        VVM::Real stage_dt) const override;
    void compute_tendency_impl(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency,
        VVM::Real stage_dt) const;
private:
    std::unique_ptr<SpatialScheme> scheme_;
    std::string variable_name_;
    std::vector<std::string> dynamics_vars_;
    std::vector<std::string> thermodynamics_vars_;
    bool force_anelastic_scalar_normalization_;

    Core::HaloExchanger& halo_exchanger_;
    const Core::BoundaryConditionManager& bc_manager_;

    std::shared_ptr<MeanWindState> mean_wind_state_;
    MeanWindState::Variant mean_wind_variant_ = MeanWindState::Variant::Scalar;

    Core::ConstFieldRef<3> advected_ref_;
    Core::FieldRef<3> u_ref_;
    Core::FieldRef<3> v_ref_;
    Core::FieldRef<3> w_ref_;
    Core::FieldRef<3> u_mean_ref_;
    Core::FieldRef<3> v_mean_ref_;
    Core::FieldRef<3> w_mean_ref_;
    Core::ConstFieldRef<1> rhobar_ref_;
    Core::ConstFieldRef<1> rhobar_up_ref_;

    mutable int normalize_by_rhobar_ = -1; // -1 means unresolved
};

} // namespace Dynamics
} // namespace VVM
#endif
