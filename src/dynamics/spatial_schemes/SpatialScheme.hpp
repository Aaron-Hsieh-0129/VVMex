#ifndef VVM_DYNAMICS_SPATIAL_SCHEME_HPP
#define VVM_DYNAMICS_SPATIAL_SCHEME_HPP

#include "core/Field.hpp"
#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include <string>
#include <stdexcept>

namespace VVM {
namespace Dynamics {

class SpatialScheme {
public:
    virtual ~SpatialScheme() = default;

    // Directionally split schemes retain the existing hooks below. Schemes
    // with a coupled multidimensional limiter override this capability and
    // calculate the complete advection tendency in one evaluation.
    virtual bool handles_multidimensional_advection() const { return false; }

    // Density-normalized prognostic scalars need this result divided by
    // rhobar after the spatial flux divergence is evaluated.
    virtual bool produces_anelastic_scalar_flux_divergence() const {
        return false;
    }

    virtual void calculate_advection_tendency(
        const Core::State& state,
        const Core::Field<3>& scalar,
        const Core::Field<3>& mass_flux_x,
        const Core::Field<3>& mass_flux_y,
        const Core::Field<3>& mass_flux_z,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency,
        const std::string& var_name,
        VVM::Real stage_dt) const {
        (void)state;
        (void)stage_dt;
        calculate_flux_convergence_x(scalar, mass_flux_x, grid, params, out_tendency, var_name);
        calculate_flux_convergence_y(scalar, mass_flux_y, grid, params, out_tendency, var_name);
        calculate_flux_convergence_z(scalar, mass_flux_z, grid, params, out_tendency, var_name);
    }

    // ∇⋅(ρvφ)
    virtual void calculate_flux_convergence_x(
        const Core::Field<3>& scalar, const Core::Field<3>& u,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_flux_convergence_y(
        const Core::Field<3>& scalar, const Core::Field<3>& v,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_flux_convergence_z(
        const Core::Field<3>& scalar, const Core::Field<3>& w,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }

    // ρω∇⋅(v)
    virtual void calculate_stretching_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_stretching_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_stretching_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }

    // Twisting Term
    // 0.5ρ(eta*Rzeta + zeta*Reta)
    virtual void calculate_twisting_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    // 0.5ρ(xi*Rzeta + zeta*Rxi)
    virtual void calculate_twisting_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    // 0.5ρ(xi*Reta + eta*Rxi)
    virtual void calculate_twisting_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }

    // Rotation (Diagnostic)
    virtual void calculate_R_xi(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_xi) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_R_eta(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_eta) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_R_zeta(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_zeta) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }

    // Vorticity divergence
    virtual void calculate_vorticity_divergence(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_field) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }

    // Buoyancy Term
    virtual void calculate_buoyancy_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_buoyancy_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }


    // Coriolis Term
    virtual void calculate_coriolis_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_coriolis_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
    virtual void calculate_coriolis_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const {
        throw std::runtime_error("Spatial scheme does not implement this operation.");
    }
};

} // namespace Dynamics
} // namespace VVM
#endif
