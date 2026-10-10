#ifndef VVM_DYNAMICS_HORIZONTAL_WIND_TOPOLOGY_CONSTRAINT_HPP
#define VVM_DYNAMICS_HORIZONTAL_WIND_TOPOLOGY_CONSTRAINT_HPP

#include <memory>

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Field.hpp"
#include "core/haloexchange/HaloExchanger.hpp"

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
    // Preserve the direct rotational basis after covariant column recovery.
    virtual void correct_canonical_recovered_wind() = 0;
    // Seed from the incoming wind, or restore from a checkpoint's saved basis.
    virtual void seed_from_physical_wind() = 0;
    virtual void seed_from_contravariant_wind() = 0;
    virtual void prepare_physical_mean_constraint(Core::HaloExchanger& halo) = 0;
    virtual void nudge_physical_top_mean(
        const Core::Field<0>& target_u, const Core::Field<0>& target_v, Real fraction) = 0;
};

class HorizontalEllipticSolver;

// Periodic cycles or a bounded-q2 wall circulation, using cached harmonic
// covectors for the supplied metric. The harmonic solves run once before
// diagnostic graph capture; timestep corrections remain entirely on device.
std::unique_ptr<HorizontalWindTopologyConstraint> make_generalized_circulation_constraint(
    const Core::Grid& grid,
    Core::State& state,
    Core::Field<3>& covariant_q1_wind,
    Core::Field<3>& covariant_q2_wind,
    Core::HaloExchanger& halo,
    HorizontalEllipticSolver& solver,
    int iterations);

// Compatibility entry for charts with analytic separable harmonic covectors.
std::unique_ptr<HorizontalWindTopologyConstraint> make_regular_lat_lon_circulation_constraint(
    const Core::Grid& grid,
    Core::State& state,
    Core::Field<3>& covariant_q1_wind,
    Core::Field<3>& covariant_q2_wind);

} // namespace VVM::Dynamics

#endif
