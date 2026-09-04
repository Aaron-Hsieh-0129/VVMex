#ifndef VVM_PHYSICS_SURFACE_PROCESS_HPP
#define VVM_PHYSICS_SURFACE_PROCESS_HPP

#include <vector>
#include <string>
#include <Kokkos_Core.hpp>

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/Field.hpp"
#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Physics {

class SurfaceProcess {
public:

    SurfaceProcess(const Utils::ConfigurationManager& config, 
                      const Core::Grid& grid, 
                      const Core::Parameters& params,
                      Core::HaloExchanger& halo_exchanger,
                      Core::State& state);

    void initialize(Core::State& state);
    void compute_coefficients(Core::State& state);

    template<size_t Dim>
    void calculate_tendencies(Core::State& state, 
                              const std::string& var_name, 
                              Core::Field<Dim>& out_tendency);

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    Core::HaloExchanger& halo_exchanger_;

    static KOKKOS_INLINE_FUNCTION
    void sflux_2d(VVM::Real sigmau, VVM::Real thvm, VVM::Real thvsm, VVM::Real speed1, 
                  VVM::Real zr, VVM::Real zrough, VVM::Real speed1_filter,
                  VVM::Real& ustar, VVM::Real ventfc[2], VVM::Real& molen);

    static KOKKOS_INLINE_FUNCTION
    void sflux_tc_2d(VVM::Real sigmau, VVM::Real thvm, VVM::Real thvsm, VVM::Real speed1, 
                     VVM::Real zr, VVM::Real zrough, VVM::Real speed1_filter, 
                     VVM::Real& ustar, VVM::Real ventfc[2], VVM::Real& molen);

    static KOKKOS_INLINE_FUNCTION
    VVM::Real compute_es(VVM::Real t);

    std::string mode_;
    std::string land_scheme_;
    std::string v_coord_type_;
    VVM::Real speed1_filter_;

    Core::FieldRef<1> pbar_ref_;
    Core::FieldRef<1> pibar_ref_;
    Core::FieldRef<1> rhobar_ref_;
    Core::FieldRef<1> rhobar_up_ref_;
    Core::FieldRef<1> thbar_ref_;
    Core::FieldRef<2> Tg_ref_;
    Core::FieldRef<2> VEN2D_ref_;
    Core::FieldRef<2> cmx_ref_;
    Core::FieldRef<2> gwet_ref_;
    Core::FieldRef<2> molen_ref_;
    Core::FieldRef<2> sea_land_ice_mask_ref_;
    Core::FieldRef<2> sfc_flux_th_ref_;
    Core::FieldRef<2> sfc_flux_qv_ref_;
    Core::FieldRef<2> sfc_flux_u_ref_;
    Core::FieldRef<2> sfc_flux_v_ref_;
    Core::FieldRef<2> topo_ref_;
    Core::FieldRef<2> topou_ref_;
    Core::FieldRef<2> topov_ref_;
    Core::FieldRef<2> ustar_ref_;
    Core::FieldRef<2> zrough_ref_;
    Core::FieldRef<3> u_ref_;
    Core::FieldRef<3> v_ref_;
    Core::FieldRef<3> th_ref_;
    Core::FieldRef<3> qv_ref_;
    Core::FieldRef<3> qc_ref_;
    Core::FieldRef<3> qi_ref_;
};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_SURFACE_PROCESS_HPP
