#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_DRY_BUOYANCY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_DRY_BUOYANCY_HPP

#include <Kokkos_Core.hpp>

#include "core/Field.hpp"
#include "core/geometry/HorizontalGeometry.hpp"
#include "dynamics/operators/HorizontalScalarGradient.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Dry baroclinic vorticity source on orthogonal regular latitude-longitude
// geometry.
//
// xi is the physical eastward vorticity component at V.
// eta is negative physical northward vorticity at U, preserving VVMex's
// existing State convention.
//
// Consequently:
//
//     d(xi)/dt  += g * physical_gradient_q2(th / thbar)
//     d(eta)/dt += g * physical_gradient_q1(th / thbar)
//
// Both gradients are averaged between adjacent T levels to reach the
// horizontal-vorticity vertical staggering.
struct RegularLatLonDryBuoyancyDeviceView {
    HorizontalScalarGradientDeviceView gradient;

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    calculate_xi_at_v(const int j,
        const int i,
        const ScalarStencilAtT& lower,
        const ScalarStencilAtT& upper,
        const VVM::Real gravity) const noexcept {

        const VVM::Real lower_contravariant = gradient.calculate_q2_at_v(j, i, lower);
        const VVM::Real upper_contravariant = gradient.calculate_q2_at_v(j, i, upper);

        const VVM::Real physical_scale = gradient.v.contravariant_to_physical.a22(j, i);

        return VVM::real(0.5) * gravity * physical_scale *
               (lower_contravariant + upper_contravariant);
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real
    calculate_eta_at_u(const int j,
        const int i,
        const ScalarStencilAtT& lower,
        const ScalarStencilAtT& upper,
        const VVM::Real gravity) const noexcept {

        const VVM::Real lower_contravariant = gradient.calculate_q1_at_u(j, i, lower);
        const VVM::Real upper_contravariant = gradient.calculate_q1_at_u(j, i, upper);

        const VVM::Real physical_scale = gradient.u.contravariant_to_physical.a11(j, i);

        return VVM::real(0.5) * gravity * physical_scale *
               (lower_contravariant + upper_contravariant);
    }
};

RegularLatLonDryBuoyancyDeviceView make_regular_lat_lon_dry_buoyancy_device_view(
    const Core::Geometry::HorizontalGeometry& geometry);

// Field-level dry buoyancy launcher.
//
// th:
//     Potential temperature at T.
//
// thbar:
//     Horizontally uniform reference potential temperature.
//
// gravity:
//     Existing device scalar from Parameters.
//
// The methods add to physical horizontal-vorticity cells over
// [k_begin, k_end). They do not initialize output, exchange halos, apply
// boundaries, synchronize, or include moisture buoyancy.
class RegularLatLonDryBuoyancy {
public:
    explicit RegularLatLonDryBuoyancy(const Core::Geometry::HorizontalGeometry& geometry);

    // Prepare the exact CUDA launch functor before manual graph capture.
    // Repeated calls are allowed. This is a no-op on CPU.
    static void prepare_execution();

    void add_xi_tendency(const Core::Field<3>& th,
        const Core::Field<1>& thbar,
        const Kokkos::View<VVM::Real>& gravity,
        Core::Field<3>& out_tendency,
        int k_begin,
        int k_end) const;

    void add_eta_tendency(const Core::Field<3>& th,
        const Core::Field<1>& thbar,
        const Kokkos::View<VVM::Real>& gravity,
        Core::Field<3>& out_tendency,
        int k_begin,
        int k_end) const;

    // Physical (not density-weighted) mixing ratios. qp is total condensate
    // from the existing physics owner; do not sum P3 categories here.
    // face_mask is ITYPEV for xi or ITYPEU for eta. As in Cartesian Takacs,
    // masked cells at/below max_topo_idx reset the accumulated tendency.
    void add_moist_tendency(const Core::Field<3>& th,
        const Core::Field<1>& thbar,
        const Kokkos::View<VVM::Real>& gravity,
        const Core::Field<3>& qv,
        const Core::Field<3>& qp,
        const Core::Field<3>& face_mask,
        Core::Field<3>& output,
        int k_begin, int k_end, int max_topo_idx, bool xi_component) const;

private:
    void add_tendency(const Core::Field<3>& th,
        const Core::Field<1>& thbar,
        const Kokkos::View<VVM::Real>& gravity,
        Core::Field<3>& out_tendency,
        int k_begin,
        int k_end,
        bool xi_component,
        const Core::Field<3>* qv = nullptr,
        const Core::Field<3>* qp = nullptr,
        const Core::Field<3>* face_mask = nullptr,
        int max_topo_idx = -1) const;

    void validate_volume(const Core::Field<3>& field, int nz, const char* role) const;

    Core::Geometry::HorizontalDomainLayout layout_;
    RegularLatLonDryBuoyancyDeviceView operator_;
};

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_DRY_BUOYANCY_HPP
