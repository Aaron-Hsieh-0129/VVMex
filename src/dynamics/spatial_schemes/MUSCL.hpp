#ifndef VVM_DYNAMICS_MUSCL_HPP
#define VVM_DYNAMICS_MUSCL_HPP

#include "SpatialScheme.hpp"
#include "core/Field.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <string>

namespace VVM {
namespace Dynamics {

class MUSCL final : public SpatialScheme {
public:
    // MUSCL is scalar-category agnostic; DynamicalCore owns the current
    // configuration eligibility policy for tracers and future water variables.
    struct Options {
        VVM::Real lower_bound = VVM::real(0.0);
        VVM::Real max_cfl = VVM::real(0.9);
    };

    static constexpr int required_halo_width = 2;

    // This literal-based helper is safe in host and device compilation. Some
    // CUDA compilers reject std::numeric_limits::epsilon in device functions.
    KOKKOS_INLINE_FUNCTION
    static VVM::Real machine_epsilon() {
#ifdef VVM_USE_DOUBLE_PRECISION
        return VVM::real(2.2204460492503130808472633361816e-16);
#else
        return VVM::real(1.1920928955078125e-7);
#endif
    }

    static Options validate_configuration(
        const std::string& variable_name,
        const std::string& spatial_scheme,
        const std::string& temporal_scheme,
        const nlohmann::json& advection_config,
        size_t enabled_tendency_count,
        int configured_halo_width);

    MUSCL(std::string variable_name,
          const nlohmann::json& advection_config,
          const Core::Grid& grid);

    bool handles_multidimensional_advection() const override { return true; }

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

    KOKKOS_INLINE_FUNCTION
    static VVM::Real van_leer_slope(VVM::Real backward, VVM::Real forward) {
        const bool both_positive =
            backward > VVM::real(0.0) && forward > VVM::real(0.0);
        const bool both_negative =
            backward < VVM::real(0.0) && forward < VVM::real(0.0);
        if (!both_positive && !both_negative) return VVM::real(0.0);

        const VVM::Real abs_backward = Kokkos::abs(backward);
        const VVM::Real abs_forward = Kokkos::abs(forward);
        const VVM::Real lo =
            abs_backward < abs_forward ? abs_backward : abs_forward;
        const VVM::Real hi =
            abs_backward < abs_forward ? abs_forward : abs_backward;
        if (hi <= VVM::real(0.0)) return VVM::real(0.0);

        const VVM::Real magnitude =
            VVM::real(2.0) * lo / (VVM::real(1.0) + lo / hi);
        return backward > VVM::real(0.0) ? magnitude : -magnitude;
    }

    KOKKOS_INLINE_FUNCTION
    static VVM::Real reconstructed_state_factor_from_deviations(
        VVM::Real q, VVM::Real lower_bound,
        VVM::Real dxm, VVM::Real dxp,
        VVM::Real dym, VVM::Real dyp,
        VVM::Real dzm, VVM::Real dzp) {
        VVM::Real theta = VVM::real(1.0);
        const VVM::Real faces[6] = {
            q + dxm, q + dxp,
            q + dym, q + dyp,
            q + dzm, q + dzp
        };
        for (int n = 0; n < 6; ++n) {
            if (faces[n] < lower_bound) {
                const VVM::Real denominator = q - faces[n];
                if (denominator <= VVM::real(0.0)) return VVM::real(0.0);
                const VVM::Real candidate =
                    (q - lower_bound) / denominator;
                if (candidate < theta) theta = candidate;
            }
        }
        if (theta < VVM::real(0.0)) return VVM::real(0.0);
        return theta > VVM::real(1.0) ? VVM::real(1.0) : theta;
    }

    KOKKOS_INLINE_FUNCTION
    static VVM::Real reconstructed_state_factor(
        VVM::Real q, VVM::Real lower_bound,
        VVM::Real sx, VVM::Real sy, VVM::Real sz) {
        const VVM::Real half = VVM::real(0.5);
        return reconstructed_state_factor_from_deviations(
            q, lower_bound,
            -half * sx, half * sx,
            -half * sy, half * sy,
            -half * sz, half * sz);
    }

    KOKKOS_INLINE_FUNCTION
    static VVM::Real donor_outflow_factor(
        VVM::Real available, VVM::Real outgoing_depletion) {
        const VVM::Real tolerance =
            VVM::real(64.0) * machine_epsilon() *
            (Kokkos::abs(available) + VVM::real(1.0));
        if (outgoing_depletion <= tolerance) return VVM::real(1.0);
        if (available <= VVM::real(0.0)) return VVM::real(0.0);
        const VVM::Real ratio = available / outgoing_depletion;
        return ratio < VVM::real(1.0) ? ratio : VVM::real(1.0);
    }

    KOKKOS_INLINE_FUNCTION
    static VVM::Real lower_bound_tolerance(VVM::Real lower_bound) {
        return VVM::real(64.0) *
            machine_epsilon() *
            (Kokkos::abs(lower_bound) + VVM::real(1.0));
    }

    KOKKOS_INLINE_FUNCTION
    static bool violates_lower_bound(
        VVM::Real value, VVM::Real lower_bound) {
        return value < lower_bound -
            lower_bound_tolerance(lower_bound);
    }

    const Options& options() const { return options_; }

    // These validation entry points launch Kokkos reductions. They remain
    // public for scalar-scheme reuse and focused numerical testing.
    void validate_initial_field(
        const Core::State& state, const Core::Field<3>& scalar) const;
    void validate_cfl(
        const Core::State& state,
        const Core::Field<3>& mass_flux_x,
        const Core::Field<3>& mass_flux_y,
        const Core::Field<3>& mass_flux_z,
        const Core::Parameters& params,
        VVM::Real stage_dt) const;

private:
    static Options parse_options(
        const std::string& variable_name,
        const nlohmann::json& advection_config);


    std::string variable_name_;
    Options options_;
    const Core::Grid& grid_;
    mutable bool initial_field_validated_ = false;

    // Scheme-owned reusable device scratch. None of these fields is registered
    // in State, so output and restart discovery cannot expose them.
    mutable Core::Field<3> slope_x_;
    mutable Core::Field<3> slope_y_;
    mutable Core::Field<3> slope_z_;
    mutable Core::Field<3> donor_alpha_;
    mutable Core::Field<3> flux_x_;
    mutable Core::Field<3> flux_y_;
    mutable Core::Field<3> flux_z_;
    mutable Core::Field<0> reduction_local_;
    mutable Core::Field<0> reduction_global_;
};

} // namespace Dynamics
} // namespace VVM

#endif
