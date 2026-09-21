#ifndef VVM_DYNAMICS_HORIZONTAL_WIND_TOPOLOGY_CONSTRAINT_HPP
#define VVM_DYNAMICS_HORIZONTAL_WIND_TOPOLOGY_CONSTRAINT_HPP

#include <memory>

#include "core/Grid.hpp"
#include "core/State.hpp"

namespace VVM::Dynamics {

// Host-side lifecycle for horizontal harmonic/global degrees of freedom.
//
// The local wind diagnostic is intentionally unaware of the topology that
// creates these null modes. A topology-specific implementation may capture a
// target before recovery and/or constrain the recovered wind afterwards.
//
// This interface owns no numerical wind-recovery stencil. Implementations may
// perform global reductions and topology-specific corrections, but must leave
// geometry/local differential operators outside this responsibility boundary.
class HorizontalWindTopologyConstraint {
public:
    virtual ~HorizontalWindTopologyConstraint() = default;

    virtual void before_recovery(bool initial) = 0;
    virtual void after_recovery(bool initial) = 0;
};

// Current RLL topology implementation:
//
// q1 periodic, q2 periodic:
//   preserve the two non-contractible cycle integrals. The target belongs to
//   the incoming physical wind, so it is captured before the first recovery.
//
// q1 periodic, q2 bounded:
//   preserve the south-wall q1 circulation. The target belongs to the first
//   diagnosed channel wind, so it is captured after the first recovery.
//
// The implementation remains RLL-specific because these cycles/walls are
// topology-specific. WindSolver itself only sees the lifecycle interface.
std::unique_ptr<HorizontalWindTopologyConstraint> make_regular_lat_lon_circulation_constraint(
    const Core::Grid& grid, Core::State& state);

} // namespace VVM::Dynamics

#endif
