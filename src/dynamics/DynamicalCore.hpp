#ifndef VVM_DYNAMICS_DYNAMICAL_CORE_HPP
#define VVM_DYNAMICS_DYNAMICAL_CORE_HPP

#include <map>
#include <string>
#include <memory>
#include <vector>
#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"
#include "spatial_schemes/Takacs.hpp"
#include "tendency_processes/AdvectionTerm.hpp"
#include "numerical_methods/NumericalMethod.hpp"
#include "solvers/WindSolver.hpp"
#include "core/BoundaryConditionManager.hpp"

namespace VVM {
namespace Dynamics {

struct IntegrationStep {
    int step;
    std::string description;
    std::vector<std::string> vars_to_calculate_tendency;
    std::vector<std::string> vars_to_update;
};

class DynamicalCore {
public:
    DynamicalCore(const Utils::ConfigurationManager& config, 
                  const Core::Grid& grid, 
                  const Core::Parameters& params,
                  Core::State& state, 
                  Core::HaloExchanger& halo_exchanger,
                  const Core::BoundaryConditionManager& bc_manager);
    ~DynamicalCore();

    void compute_diagnostic_fields() const;
    void compute_zeta_vertical_structure(Core::State& state) const;
    void compute_uvtopmn();
    void compute_wind_fields();
    // void step(Core::State& state, VVM::Real dt);

    void calculate_thermo_tendencies();
    void update_thermodynamics(VVM::Real dt);
    void calculate_vorticity_tendencies();
    void update_vorticity(VVM::Real dt);
    void diagnose_wind_fields(Core::State& state);
    void initialize_restart_history();


private:
    const Utils::ConfigurationManager& config_;
    Core::State& state_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    const Core::BoundaryConditionManager& bc_manager_;
    std::vector<std::string> thermo_vars_;
    std::vector<std::string> vorticity_vars_;
    
    std::map<std::string, std::unique_ptr<NumericalMethod>> numerical_methods_;
    std::vector<IntegrationStep> integration_procedure_;

    struct VariableCache {
        std::string name;
        Core::Field<3>* field = nullptr;
        NumericalMethod* method = nullptr;
        Core::Field<3>* fe_tendency = nullptr;
        bool zero_gradient_top = false;
        bool is_th = false;
        bool is_xi = false;
        bool is_eta = false;
    };
    std::vector<VariableCache> thermo_cache_;
    std::vector<VariableCache> vorticity_cache_;
    std::vector<const VariableCache*> single_stage_thermo_;
    std::vector<Core::Field<3>*> single_stage_thermo_fields_;
    bool field_cache_ready_ = false;
    void ensure_field_cache();

    std::shared_ptr<MeanWindState> mean_wind_state_;

    std::unique_ptr<WindSolver> wind_solver_;
    std::unique_ptr<Takacs> diagnostic_scheme_;

    Core::HaloExchanger& halo_exchanger_;

    bool enable_coriolis_ = false;
    bool enable_turbulence_ = false;

    Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space> tempumn_{"tempumn"};
    Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space> tempvmn_{"tempvmn"};
    Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space> mean_u_turb_{"mean_u_turb"};
    Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space> mean_v_turb_{"mean_v_turb"};
    Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space> mean_u_coriolis_{"mean_u_coriolis"};
    Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space> mean_v_coriolis_{"mean_v_coriolis"};

    Core::FieldRef<0> utopmn_ref_;
    Core::FieldRef<0> utopmn_m_ref_;
    Core::FieldRef<0> vtopmn_ref_;
    Core::FieldRef<0> vtopmn_m_ref_;
    Core::FieldRef<1> f_ref_;
    Core::FieldRef<1> thbar_ref_;
    Core::FieldRef<1> rhobar_ref_;
    Core::FieldRef<1> rhobar_up_ref_;
    Core::FieldRef<1> d_utopmn_ref_;
    Core::FieldRef<1> d_vtopmn_ref_;
    Core::FieldRef<2> tempu_ref_;
    Core::FieldRef<2> tempv_ref_;
    Core::FieldRef<3> u_ref_;
    Core::FieldRef<3> v_ref_;
    Core::FieldRef<3> w_ref_;
    Core::FieldRef<3> xi_ref_;
    Core::FieldRef<3> eta_ref_;
    Core::FieldRef<3> zeta_ref_;
    Core::FieldRef<3> u_topo_ref_;
    Core::FieldRef<3> v_topo_ref_;
    Core::FieldRef<3> w_topo_ref_;
    Core::FieldRef<3> xi_topo_ref_;
    Core::FieldRef<3> eta_topo_ref_;
    Core::FieldRef<3> R_xi_ref_;
    Core::FieldRef<3> R_eta_ref_;
    Core::FieldRef<3> R_zeta_ref_;
    Core::FieldRef<3> ITYPEU_ref_;
    Core::FieldRef<3> ITYPEV_ref_;
    Core::FieldRef<3> ITYPEW_ref_;
    Core::FieldRef<3> RKM_ref_;
    Core::FieldRef<3> RKH_ref_;
};

} // namespace Dynamics
} // namespace VVM
#endif
