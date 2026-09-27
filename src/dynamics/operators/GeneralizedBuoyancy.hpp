#ifndef VVM_DYNAMICS_OPERATORS_GENERALIZED_BUOYANCY_HPP
#define VVM_DYNAMICS_OPERATORS_GENERALIZED_BUOYANCY_HPP

#include <Kokkos_Core.hpp>

#include "core/Field.hpp"
#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM::Dynamics::Operators {

// Curl of the vertical buoyancy force in a stationary, right-handed,
// z-independent horizontal chart; physical z is unchanged.
//
// s = th/thbar + 0.608*qv - qp (omit moisture in dry mode).
// S^1 =  g/J * partial_2(s),  S^2 = -g/J * partial_1(s).
//
// Inputs below are unscaled scalar differences across the positive face,
// separately at the lower and upper T levels. Outputs are TRUE tensor
// components, not VVM eta. The half-sum preserves the existing vertical
// staggering. No density normalization, metric raising or physical-component
// conversion belongs here. Nonzero g12 is allowed; only J enters this curl.
struct GeneralizedBuoyancyDeviceView {
    Core::Geometry::GeometryField2D inverse_jacobian_u;
    Core::Geometry::GeometryField2D inverse_jacobian_v;

    Real dq1 = real(0.0);
    Real dq2 = real(0.0);

    KOKKOS_INLINE_FUNCTION Real
    calculate_omega1_at_v(int j,
        int i,
        Real lower_difference_q2,
        Real upper_difference_q2,
        Real gravity) const noexcept {
        return real(0.5) * gravity * inverse_jacobian_v(j, i) *
               ((lower_difference_q2 + upper_difference_q2) / dq2);
    }

    KOKKOS_INLINE_FUNCTION Real
    calculate_omega2_at_u(int j,
        int i,
        Real lower_difference_q1,
        Real upper_difference_q1,
        Real gravity) const noexcept {
        return -real(0.5) * gravity * inverse_jacobian_u(j, i) *
               ((lower_difference_q1 + upper_difference_q1) / dq1);
    }
};

GeneralizedBuoyancyDeviceView make_generalized_buoyancy_device_view(
    const Core::Geometry::HorizontalGeometry& geometry);

// Field-level dry/moist buoyancy launcher. Default output is canonical:
// xi_con = omega^1, eta_con = -omega^2. Inputs th, qv, qp are physical;
// qp is the condensate diagnostic supplied by physics, not recomputed here.
//
// Caller owns halos, positive finite thbar/J and finite gravity
class GeneralizedBuoyancy {
public:
    explicit GeneralizedBuoyancy(const Core::Geometry::HorizontalGeometry& geometry);

    static void prepare_execution();

    void add_xi_tendency(const Core::Field<3>& th,
        const Core::Field<1>& thbar,
        const Kokkos::View<Real>& gravity,
        Core::Field<3>& output,
        int k_begin,
        int k_end) const;

    void add_eta_tendency(const Core::Field<3>& th,
        const Core::Field<1>& thbar,
        const Kokkos::View<Real>& gravity,
        Core::Field<3>& output,
        int k_begin,
        int k_end) const;

    void add_moist_tendency(const Core::Field<3>& th,
        const Core::Field<1>& thbar,
        const Kokkos::View<Real>& gravity,
        const Core::Field<3>& qv,
        const Core::Field<3>& qp,
        const Core::Field<3>& face_mask,
        Core::Field<3>& output,
        int k_begin,
        int k_end,
        int max_topo_idx,
        bool xi_component) const;

private:
    void add_tendency(const Core::Field<3>& th,
        const Core::Field<1>& thbar,
        const Kokkos::View<Real>& gravity,
        Core::Field<3>& output,
        int k_begin,
        int k_end,
        bool xi_component,
        const Core::Field<3>* qv = nullptr,
        const Core::Field<3>* qp = nullptr,
        const Core::Field<3>* face_mask = nullptr,
        int max_topo_idx = -1) const;

    void validate_volume(const Core::Field<3>& field, int nz, const char* role) const;

    Core::Geometry::HorizontalDomainLayout layout_;

    GeneralizedBuoyancyDeviceView operator_;
};

} // namespace VVM::Dynamics::Operators

#endif
