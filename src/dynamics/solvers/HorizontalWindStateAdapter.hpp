#ifndef VVM_DYNAMICS_HORIZONTAL_WIND_STATE_ADAPTER_HPP
#define VVM_DYNAMICS_HORIZONTAL_WIND_STATE_ADAPTER_HPP

#include "core/Field.hpp"
#include "core/geometry/HorizontalGeometry.hpp"
#include "dynamics/operators/HorizontalWindReconstruction.hpp"

namespace VVM {
namespace Dynamics {

// Physical-field interface for flat, height-independent Cartesian/RLL geometry.
// u/v/w are physical velocities. xi is physical eastward vorticity;
// eta is NEGATIVE physical northward vorticity, preserving VVMex's sign.
// Owns geometry handles only: no State fields, solver, or volume scratch.
class HorizontalWindStateAdapter {
public:
    explicit HorizontalWindStateAdapter(const Core::Geometry::HorizontalGeometry& geometry);

    // Explicit backend preparation outside capture. No model fields are touched.
    static void prepare_execution();

    // Initialization only, outside capture: spacing(k) = dz / flex_up(k).
    // Output must already be allocated; validates all values before writing.
    static void initialize_spacing(VVM::Real dz, const Core::Field<1>& flex_up, Core::Field<1>& spacing);

    // Compute xi/eta on interfaces first..last, inclusive, using wind k and k+1.
    // Writes physical horizontal cells only. Does not diagnose vertical zeta.
    void diagnose_vorticity(const Core::Field<3>& u, const Core::Field<3>& v, const Core::Field<3>& w,
        const Core::Field<1>& spacing, Core::Field<3>& xi, Core::Field<3>& eta, int first, int last) const;

    // Convert the existing covariant potential reconstruction to physical u/v.
    // Writes only level top. No mean subtraction or circulation adjustment.
    void reconstruct_top(const Core::Field<2>& psi, const Core::Field<2>& chi,
        Core::Field<3>& u, Core::Field<3>& v, int top) const;

    // Preserve prescribed physical u/v at top; integrate down through bottom.
    // Can therefore retain a caller-supplied compatible nonzero top circulation.
    void integrate_from_top(const Core::Field<3>& w, const Core::Field<3>& xi, const Core::Field<3>& eta,
        const Core::Field<1>& spacing, Core::Field<3>& u, Core::Field<3>& v, int bottom, int top) const;

    // All execution methods require valid input halos and positive finite used
    // spacing. Outputs must not overlap each other or inputs. No allocations,
    // halo exchange, wall conditions, vertical ghost policy, or fences occur.
    // Call prepare_execution() outside capture before capturing these methods.

private:
    void validate_volume(const Core::Field<3>& field, int nz) const;
    void validate_plane(const Core::Field<2>& field) const;

    Core::Geometry::HorizontalDomainLayout layout_;
    Core::Geometry::GeometryField2D inverse_h1_at_u_;
    Core::Geometry::GeometryField2D inverse_h2_at_v_;
    Operators::HorizontalWindReconstructionDeviceView reconstruction_;
    VVM::Real dq1_;
    VVM::Real dq2_;
};

} // namespace Dynamics
} // namespace VVM

#endif
