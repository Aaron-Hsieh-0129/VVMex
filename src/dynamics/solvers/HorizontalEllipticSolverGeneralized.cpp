#include "dynamics/solvers/HorizontalEllipticSolver.hpp"

namespace VVM::Dynamics {
namespace {

using Plane = Core::Field<2>::ViewType;
using Operator = Operators::GeneralizedHorizontalEllipticDeviceView;
using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

// Capture a single device handle, not all geometry views by value.
//
// The same launch type serves single, paired and channel solves and is
// prepared here, in the translation unit that also launches it during
// CUDA graph capture.
struct RelaxationFunctor {
    Kokkos::View<const Operator> operation;

    Plane rhs_z;
    Plane previous_z;
    Plane current_z;

    Plane rhs_t;
    Plane previous_t;
    Plane current_t;

    Real shift = real(0.0);
    Real q2_plus_value = real(0.0);
    int q2_plus_row = -1;

    bool update_z = false;
    bool update_t = false;
    bool evaluate = true;

    KOKKOS_INLINE_FUNCTION void
    operator()(int j, int i) const {
        if (!evaluate) {
            return;
        }

        if (update_z) {
            current_z(j, i) = j == q2_plus_row
                                  ? q2_plus_value
                                  : operation().relaxed_at_z(previous_z, rhs_z(j, i), shift, j, i);
        }

        if (update_t) {
            current_t(j, i) = operation().relaxed_at_t(previous_t, rhs_t(j, i), shift, j, i);
        }
    }
};

void
launch(const RelaxationFunctor& functor, int h, int ny, int nx) {
    Kokkos::parallel_for("RelaxGeneralizedHorizontalElliptic",
        Kokkos::Experimental::require(Policy({h, h}, {ny - h, nx - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        functor);
}

} // namespace

void
HorizontalEllipticSolver::prepare_generalized_execution() const {
    RelaxationFunctor preparation{};
    preparation.evaluate = false;

    launch(preparation, 0, 1, 1);
    Kokkos::fence();
}

void
HorizontalEllipticSolver::relax_generalized_single(bool at_z,
    const Core::Field<2>& right_hand_side,
    const Core::Field<2>& previous,
    Core::Field<2>& current,
    const Options& options) const {
    RelaxationFunctor functor{};

    functor.operation = generalized_;
    functor.shift = options.diagonal_shift;

    if (at_z) {
        functor.update_z = true;
        functor.rhs_z = right_hand_side.get_device_data();
        functor.previous_z = previous.get_device_data();
        functor.current_z = current.get_mutable_device_data();
    }
    else {
        functor.update_t = true;
        functor.rhs_t = right_hand_side.get_device_data();
        functor.previous_t = previous.get_device_data();
        functor.current_t = current.get_mutable_device_data();
    }

    launch(functor,
        grid_.get_halo_cells(),
        grid_.get_local_total_points_y(),
        grid_.get_local_total_points_x());
}

void
HorizontalEllipticSolver::relax_generalized_pair(const Core::Field<2>& right_hand_side_at_z,
    const Core::Field<2>& previous_at_z,
    Core::Field<2>& current_at_z,
    const Core::Field<2>& right_hand_side_at_t,
    const Core::Field<2>& previous_at_t,
    Core::Field<2>& current_at_t,
    const Options& options,
    const bool constrain_q2_plus_wall) const {
    RelaxationFunctor functor{};

    functor.operation = generalized_;
    functor.shift = options.diagonal_shift;
    functor.update_z = true;
    functor.update_t = true;

    functor.rhs_z = right_hand_side_at_z.get_device_data();
    functor.previous_z = previous_at_z.get_device_data();
    functor.current_z = current_at_z.get_mutable_device_data();

    functor.rhs_t = right_hand_side_at_t.get_device_data();
    functor.previous_t = previous_at_t.get_device_data();
    functor.current_t = current_at_t.get_mutable_device_data();

    if (constrain_q2_plus_wall &&
        grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1) {

        functor.q2_plus_row = grid_.get_local_total_points_y() - grid_.get_halo_cells() - 1;
        functor.q2_plus_value = options.psi_q2_plus;
    }

    launch(functor,
        grid_.get_halo_cells(),
        grid_.get_local_total_points_y(),
        grid_.get_local_total_points_x());
}

} // namespace VVM::Dynamics
