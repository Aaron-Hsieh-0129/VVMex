#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_HORIZONTAL_ELLIPTIC_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_HORIZONTAL_ELLIPTIC_HPP

#include "dynamics/operators/GeneralizedWindRecovery.hpp"

namespace VVM::Dynamics::Operators {

// Potentials use different discrete operators:
//
//   L_T chi = div(reconstruct_contravariant(0, chi)),
//   L_Z psi = curl(reconstruct_covariant(psi, 0)).
//
// Both are composed from the accepted native-face reconstruction, including
// metric-times-derivative interpolation. This is not a replacement for every
// scalar Laplace-Beltrami discretization in the model.
//
// Stationary horizontal chart, physical z unchanged, one valid scalar/metric
// halo (corners included). Geometry/topology must supply compatible halos.
//
// Methods below return J*L, not L. A physical RHS is multiplied by J once.
struct GeneralizedHorizontalEllipticDeviceView {
    GeneralizedHorizontalWindReconstructionDeviceView reconstruction;

    Core::Geometry::GeometryField2D jacobian_t;
    Core::Geometry::GeometryField2D jacobian_z;

    struct ZeroPotential {
        KOKKOS_INLINE_FUNCTION Real
        operator()(int, int) const noexcept {
            return real(0.0);
        }
    };

    struct UnitImpulse {
        int row;
        int column;

        KOKKOS_INLINE_FUNCTION Real
        operator()(int j, int i) const noexcept {
            return j == row && i == column ? real(1.0) : real(0.0);
        }
    };

    template <typename ScalarView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_jacobian_weighted_at_t(const ScalarView& chi, int j, int i) const noexcept {
        const auto& r = reconstruction;
        const ZeroPotential zero{};

        const Real east = r.jacobian_u(j, i) * r.calculate_contravariant_q1_at_u(zero, chi, j, i);

        const Real west =
            r.jacobian_u(j, i - 1) * r.calculate_contravariant_q1_at_u(zero, chi, j, i - 1);

        const Real north = r.jacobian_v(j, i) * r.calculate_contravariant_q2_at_v(zero, chi, j, i);

        const Real south =
            r.jacobian_v(j - 1, i) * r.calculate_contravariant_q2_at_v(zero, chi, j - 1, i);

        return (east - west) / r.dq1 + (north - south) / r.dq2;
    }

    template <typename ScalarView>
    KOKKOS_INLINE_FUNCTION Real
    calculate_jacobian_weighted_at_z(const ScalarView& psi, int j, int i) const noexcept {
        const auto& r = reconstruction;
        const ZeroPotential zero{};

        const Real east = r.calculate_covariant_q2_at_v(psi, zero, j, i + 1);

        const Real west = r.calculate_covariant_q2_at_v(psi, zero, j, i);

        const Real north = r.calculate_covariant_q1_at_u(psi, zero, j + 1, i);

        const Real south = r.calculate_covariant_q1_at_u(psi, zero, j, i);

        return (east - west) / r.dq1 - (north - south) / r.dq2;
    }

    // Obtain the actual center coefficient of the same linear stencil.
    // Mixed terms can contribute to the T-point diagonal when J and g^12
    // vary. Do not reuse an orthogonal five-point diagonal in that case.
    //
    // UnitImpulse reads no field and allocates nothing.
    KOKKOS_INLINE_FUNCTION Real
    jacobian_weighted_diagonal_at_t(int j, int i) const noexcept {
        return calculate_jacobian_weighted_at_t(UnitImpulse{j, i}, j, i);
    }

    KOKKOS_INLINE_FUNCTION Real
    jacobian_weighted_diagonal_at_z(int j, int i) const noexcept {
        return calculate_jacobian_weighted_at_z(UnitImpulse{j, i}, j, i);
    }

    template <typename ScalarView>
    KOKKOS_INLINE_FUNCTION Real
    relaxed_at_t(const ScalarView& previous, Real physical_rhs, Real diagonal_shift, int j, int i)
        const noexcept {
        return previous(j, i) + (calculate_jacobian_weighted_at_t(previous, j, i) -
                                    jacobian_t(j, i) * physical_rhs) /
                                    (diagonal_shift - jacobian_weighted_diagonal_at_t(j, i));
    }

    template <typename ScalarView>
    KOKKOS_INLINE_FUNCTION Real
    relaxed_at_z(const ScalarView& previous, Real physical_rhs, Real diagonal_shift, int j, int i)
        const noexcept {
        return previous(j, i) + (calculate_jacobian_weighted_at_z(previous, j, i) -
                                    jacobian_z(j, i) * physical_rhs) /
                                    (diagonal_shift - jacobian_weighted_diagonal_at_z(j, i));
    }
};

inline GeneralizedHorizontalEllipticDeviceView
make_generalized_horizontal_elliptic_device_view(
    const Core::Geometry::HorizontalGeometry& geometry) {
    using Core::Geometry::HorizontalLocation;

    return {make_generalized_horizontal_wind_reconstruction_device_view(geometry),
        geometry.device_view(HorizontalLocation::T).sqrt_g,
        geometry.device_view(HorizontalLocation::Z).sqrt_g};
}

} // namespace VVM::Dynamics::Operators

#endif
