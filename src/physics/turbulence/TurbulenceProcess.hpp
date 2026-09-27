#ifndef VVM_PHYSICS_TURBULENCE_PROCESS_HPP
#define VVM_PHYSICS_TURBULENCE_PROCESS_HPP

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

enum CoeffID {
    UU1 = 0,
    UU2,
    UV1,
    UV2,
    UW1,
    UW2,
    VU1,
    VU2,
    VV1,
    VV2,
    VW1,
    VW2,
    WU1,
    WU2,
    WV1,
    WV2,
    WW1,
    WW2,
    TOTAL_BITS
};

struct TerrainMasks {
    using MaskView = Kokkos::View<unsigned int***>;
    MaskView data;

    TerrainMasks() = default;

    TerrainMasks(int nz, int ny, int nx) : data("Terrain_Masks", nz, ny, nx) {
        reset_to_ones();
    }

    void
    reset_to_ones() {
        unsigned int all_ones = (1u << TOTAL_BITS) - 1;

        Kokkos::deep_copy(data, all_ones);
    }

    KOKKOS_INLINE_FUNCTION
    void
    turn_off(int k, int j, int i, CoeffID id) const {

        Kokkos::atomic_fetch_and(&data(k, j, i), ~(1u << id));
    }

    KOKKOS_INLINE_FUNCTION
    void
    turn_off_all(int k, int j, int i) const {

        data(k, j, i) = 0u;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    val(int k, int j, int i, CoeffID id) const {

        return ((data(k, j, i) >> id) & 1) ? VVM::real(1.0) : VVM::real(0.0);
    }

    KOKKOS_INLINE_FUNCTION
    bool
    is_active(int k, int j, int i, CoeffID id) const {

        return (data(k, j, i) >> id) & 1;
    }

    MaskView
    get_raw_view() {
        return data;
    }
};

class TurbulenceProcess {
public:
    TurbulenceProcess(const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::HaloExchanger& halo_exchanger,
        Core::State& state);

    void compute_coefficients(Core::State& state, VVM::Real dt);

    template <size_t Dim>
    void calculate_tendencies(
        Core::State& state, const std::string& var_name, Core::Field<Dim>& out_tendency);

    void initialize(Core::State& state);

    void init_boundary_masks(Core::State& state);

    const std::vector<std::string>&
    get_thermodynamics_vars() const {
        return thermodynamics_vars_;
    }

    const std::vector<std::string>&
    get_dynamics_vars() const {
        return dynamics_vars_;
    }

    // ------------------------------------------------------------
    // Generalized-coordinate turbulence
    //
    // These routines are used for mapped horizontal geometries.
    // They must not assume RegularLatLon, orthogonality, or g12 = 0.
    // ------------------------------------------------------------

    void compute_generalized_deformation(Core::State& state);

    void compute_generalized_coefficients(Core::State& state, VVM::Real dt);

    void calculate_generalized_xi_tendency(Core::State& state, Core::Field<3>& out_tendency);

    void calculate_generalized_eta_tendency(Core::State& state, Core::Field<3>& out_tendency);

    void calculate_generalized_zeta_tendency(Core::State& state, Core::Field<2>& out_tendency);

    void calculate_generalized_scalar_tendency(
        Core::State& state, const std::string& var_name, Core::Field<3>& out_tendency);

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;

    Core::HaloExchanger& halo_exchanger_;

    std::vector<std::string> dynamics_vars_;
    std::vector<std::string> thermodynamics_vars_;

    VVM::Real dx_;
    VVM::Real dy_;
    VVM::Real dz_;

    VVM::Real rdx_;
    VVM::Real rdy_;
    VVM::Real rdz_;

    VVM::Real rdx2_;
    VVM::Real rdy2_;
    VVM::Real rdz2_;

    VVM::Real deld_;   // Grid scale length
    VVM::Real ramd0s_; // Asymptotic mixing length squared
    VVM::Real critmn_; // Minimum viscosity

    VVM::Real grav_;
    VVM::Real vk_;

    TerrainMasks masks_;

    Core::FieldRef<1> rhobar_ref_;
    Core::FieldRef<1> rhobar_up_ref_;

    Core::FieldRef<2> topo_ref_;
    Core::FieldRef<2> topov_ref_;

    // Physical horizontal velocity components.
    //
    // These are retained because the legacy Cartesian turbulence path
    // still uses the original physical-component formulation.
    Core::FieldRef<3> u_ref_;
    Core::FieldRef<3> v_ref_;

    // Canonical generalized-coordinate horizontal velocity components.
    //
    // These are used by compute_generalized_deformation() and
    // compute_generalized_coefficients().  Keeping them separate from
    // u/v is important: a mapped grid must not reinterpret physical
    // components as contravariant components.
    Core::FieldRef<3> u_con_ref_;
    Core::FieldRef<3> v_con_ref_;

    Core::FieldRef<3> w_ref_;
    Core::FieldRef<3> th_ref_;

    Core::FieldRef<3> R_xi_ref_;
    Core::FieldRef<3> R_eta_ref_;
    Core::FieldRef<3> R_zeta_ref_;

    Core::FieldRef<3> ITYPEU_ref_;
    Core::FieldRef<3> ITYPEV_ref_;
    Core::FieldRef<3> ITYPEW_ref_;

    Core::FieldRef<3> RKM_ref_;
    Core::FieldRef<3> RKH_ref_;
};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_TURBULENCE_PROCESS_HPP
