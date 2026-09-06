#ifndef VVM_DYNAMICS_HORIZONTAL_WIND_COLUMN_RECOVERY_HPP
#define VVM_DYNAMICS_HORIZONTAL_WIND_COLUMN_RECOVERY_HPP

#include "core/Field.hpp"
#include "core/geometry/HorizontalGeometry.hpp"
#include "dynamics/operators/HorizontalVorticity.hpp"
#include "dynamics/operators/HorizontalWindReconstruction.hpp"

namespace VVM {
namespace Dynamics {

// Recover covariant horizontal wind from supplied top-boundary potentials
// and contravariant horizontal vorticity.
//
// This is a wind-recovery component, not an elliptic solver.
// It does not own prognostic fields or change their representation.
class HorizontalWindColumnRecovery {
public:
    explicit HorizontalWindColumnRecovery(const Core::Geometry::HorizontalGeometry& geometry);

    // psi:       streamfunction at horizontal Z points
    // chi:       velocity potential at horizontal T points
    // w:         physical vertical velocity, horizontally at T
    // omega1:    contravariant vorticity omega^1, horizontally at V
    // omega2:    contravariant vorticity omega^2, horizontally at U
    // spacing:   physical distance between wind levels k and k+1
    // output1:   covariant wind u_1, horizontally at U
    // output2:   covariant wind u_2, horizontally at V
    //
    // Writes physical horizontal columns, from bottom_level through top_level.
    // Horizontal halos and other vertical levels are left untouched.
    //
    // Required input halos must already be valid. Used spacing values must
    // be positive and finite. Inputs and outputs must not overlap.
    //
    // No halo exchange, mean correction, or vertical ghost policy is applied.
    // The caller owns synchronization; this method does not fence.
    void recover(const Core::Field<2>& psi, const Core::Field<2>& chi,
        const Core::Field<3>& w, const Core::Field<3>& omega1, const Core::Field<3>& omega2,
        const Core::Field<1>& spacing, Core::Field<3>& output1, Core::Field<3>& output2,
        int bottom_level, int top_level) const;

private:
    void validate_horizontal_field(const Core::Field<2>& field, const char* role) const;
    void validate_volume(const Core::Field<3>& field, int nz, const char* role) const;

    Core::Geometry::HorizontalDomainLayout layout_;
    Operators::HorizontalWindReconstructionDeviceView reconstruction_;
    Operators::HorizontalVorticityDeviceView vorticity_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_HORIZONTAL_WIND_COLUMN_RECOVERY_HPP
