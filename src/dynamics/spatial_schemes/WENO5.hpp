#ifndef VVM_DYNAMICS_WENO5_HPP
#define VVM_DYNAMICS_WENO5_HPP

#include "SpatialScheme.hpp"
#include "Takacs.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <string>

namespace VVM {
namespace Dynamics {

// Fifth-order finite-volume WENO-JS reconstruction for passive tracers.
//
// Only the uniform horizontal directions use WENO5. Vertical transport is
// deliberately delegated to Takacs because VVMex's vertical coordinate may
// be stretched, while these WENO coefficients assume uniform cell spacing.
class WENO5 final : public SpatialScheme {
public:
    struct Options {
        // Kept at the supplied/reference value for regression testing. This
        // fixed dimensional epsilon is configurable because its appropriate
        // size depends on tracer magnitude and floating-point precision.
        VVM::Real epsilon = VVM::real(1.0e-6);
    };

    // The divergence in cell i uses faces i-1/2 and i+1/2. The complete
    // positive/negative-velocity stencil is q_{i-3},...,q_{i+3}.
    static constexpr int required_halo_width = 3;

    static Options validate_configuration(
        const std::string& variable_name,
        const std::string& spatial_scheme,
        const std::string& temporal_scheme,
        const nlohmann::json& advection_config,
        size_t enabled_tendency_count,
        int configured_halo_width);

    WENO5(
        std::string variable_name,
        const nlohmann::json& advection_config,
        const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        Core::HaloExchanger& halo_exchanger,
        const Core::BoundaryConditionManager& bc_manager);

    bool handles_multidimensional_advection() const override { return true; }
    bool produces_anelastic_scalar_flux_divergence() const override {
        return true;
    }

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

    // q^-_{i+1/2} from cell averages q_{i-2},...,q_{i+2}.
    KOKKOS_INLINE_FUNCTION
    static VVM::Real reconstruct_left(
        VVM::Real q_im2,
        VVM::Real q_im1,
        VVM::Real q_i,
        VVM::Real q_ip1,
        VVM::Real q_ip2,
        VVM::Real epsilon) {
        const VVM::Real one_sixth = VVM::real(1.0) / VVM::real(6.0);
        const VVM::Real p0 =
            one_sixth *
            (VVM::real(2.0) * q_im2 - VVM::real(7.0) * q_im1 +
             VVM::real(11.0) * q_i);
        const VVM::Real p1 =
            one_sixth *
            (-q_im1 + VVM::real(5.0) * q_i +
             VVM::real(2.0) * q_ip1);
        const VVM::Real p2 =
            one_sixth *
            (VVM::real(2.0) * q_i + VVM::real(5.0) * q_ip1 -
             q_ip2);

        const VVM::Real d20 = q_im2 - VVM::real(2.0) * q_im1 + q_i;
        const VVM::Real d10 =
            q_im2 - VVM::real(4.0) * q_im1 + VVM::real(3.0) * q_i;
        const VVM::Real d21 = q_im1 - VVM::real(2.0) * q_i + q_ip1;
        const VVM::Real d11 = q_im1 - q_ip1;
        const VVM::Real d22 = q_i - VVM::real(2.0) * q_ip1 + q_ip2;
        const VVM::Real d12 =
            VVM::real(3.0) * q_i - VVM::real(4.0) * q_ip1 + q_ip2;

        const VVM::Real beta0 =
            (VVM::real(13.0) / VVM::real(12.0)) * d20 * d20 +
            VVM::real(0.25) * d10 * d10;
        const VVM::Real beta1 =
            (VVM::real(13.0) / VVM::real(12.0)) * d21 * d21 +
            VVM::real(0.25) * d11 * d11;
        const VVM::Real beta2 =
            (VVM::real(13.0) / VVM::real(12.0)) * d22 * d22 +
            VVM::real(0.25) * d12 * d12;

        const VVM::Real e0 = epsilon + beta0;
        const VVM::Real e1 = epsilon + beta1;
        const VVM::Real e2 = epsilon + beta2;
        const VVM::Real alpha0 = VVM::real(0.1) / (e0 * e0);
        const VVM::Real alpha1 = VVM::real(0.6) / (e1 * e1);
        const VVM::Real alpha2 = VVM::real(0.3) / (e2 * e2);
        const VVM::Real alpha_sum = alpha0 + alpha1 + alpha2;

        return
            (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / alpha_sum;
    }

    // q^+_{i+1/2} uses q_{i-1},...,q_{i+3}. The indexing is the explicit
    // mirror x -> -x of reconstruct_left about face i+1/2:
    //   (q_{i-2},q_{i-1},q_i,q_{i+1},q_{i+2})
    // ->(q_{i+3},q_{i+2},q_{i+1},q_i,q_{i-1}).
    KOKKOS_INLINE_FUNCTION
    static VVM::Real reconstruct_right(
        VVM::Real q_im1,
        VVM::Real q_i,
        VVM::Real q_ip1,
        VVM::Real q_ip2,
        VVM::Real q_ip3,
        VVM::Real epsilon) {
        return reconstruct_left(
            q_ip3, q_ip2, q_ip1, q_i, q_im1, epsilon);
    }

    const Options& options() const { return options_; }

private:
    static Options parse_options(
        const std::string& variable_name,
        const nlohmann::json& advection_config);

    std::string variable_name_;
    Options options_;
    const Core::Grid& grid_;
    Takacs vertical_scheme_;

    Core::ConstFieldRef<3> ITYPEU_ref_;
    Core::ConstFieldRef<3> ITYPEV_ref_;
    Core::ConstFieldRef<3> ITYPEW_ref_;
};

} // namespace Dynamics
} // namespace VVM

#endif
