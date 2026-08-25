#ifndef VVM_DYNAMICS_TAKACS_HPP
#define VVM_DYNAMICS_TAKACS_HPP

#include "SpatialScheme.hpp"
#include "core/HaloExchanger.hpp"
#include "core/vvm_types.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"

namespace VVM {
namespace Dynamics {

class Takacs : public SpatialScheme {
public:
    explicit Takacs(const Utils::ConfigurationManager& config, const Core::Grid& grid, Core::HaloExchanger& halo_exchanger, const Core::BoundaryConditionManager& bc_manager);

    void calculate_advection_tendency(
        const Core::State& state,
        const Core::Field<3>& scalar,
        const Core::Field<3>& mass_flux_x,
        const Core::Field<3>& mass_flux_y,
        const Core::Field<3>& mass_flux_z,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency,
        const std::string& var_name,
        VVM::Real stage_dt) const override;

    void calculate_flux_convergence_x(
        const Core::Field<3>& scalar, const Core::Field<3>& u,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;
    void calculate_flux_convergence_y(
        const Core::Field<3>& scalar, const Core::Field<3>& v,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;
    void calculate_flux_convergence_z(
        const Core::Field<3>& scalar, const Core::Field<3>& w,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;

    void calculate_stretching_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;
    void calculate_stretching_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;
    void calculate_stretching_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;

    void calculate_twisting_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;
    void calculate_twisting_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;
    void calculate_twisting_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const override;

    void calculate_R_xi(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_xi) const override;
    void calculate_R_eta(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_eta) const override;
    void calculate_R_zeta(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_zeta) const override;

    void calculate_vorticity_divergence(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_field) const override;

    void calculate_buoyancy_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_buoyancy_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;

    void calculate_coriolis_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_coriolis_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_coriolis_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
private:
    const Utils::ConfigurationManager& config_;
    Core::HaloExchanger& halo_exchanger_;
    const Core::BoundaryConditionManager& bc_manager_;

    using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using MemberType = TeamPolicy::member_type;

    Core::ConstFieldRef<3> u_ref_;
    Core::ConstFieldRef<3> v_ref_;
    Core::ConstFieldRef<3> w_ref_;
    Core::ConstFieldRef<3> xi_ref_;
    Core::ConstFieldRef<3> eta_ref_;
    Core::ConstFieldRef<3> zeta_ref_;
    Core::ConstFieldRef<3> R_xi_ref_;
    Core::ConstFieldRef<3> R_eta_ref_;
    Core::ConstFieldRef<3> R_zeta_ref_;
    Core::ConstFieldRef<3> th_ref_;
    Core::ConstFieldRef<3> qv_ref_;
    Core::ConstFieldRef<3> qp_ref_;
    Core::ConstFieldRef<3> ITYPEU_ref_;
    Core::ConstFieldRef<3> ITYPEV_ref_;
    Core::ConstFieldRef<2> f_2d_ref_;
    Core::ConstFieldRef<1> rhobar_ref_;
    Core::ConstFieldRef<1> rhobar_up_ref_;
    Core::ConstFieldRef<1> thbar_ref_;
};

} // namespace Dynamics
} // namespace VVM
#endif
