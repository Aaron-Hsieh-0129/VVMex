#include "dynamics/solvers/HorizontalEllipticSolver.hpp"

#include "core/geometry/GeometryKind.hpp"

#include <stdexcept>
#include <utility>

namespace VVM::Dynamics {

void
HorizontalEllipticSolver::solve_bounded_q2_free_slip_at_z_and_t(
    const Core::Field<2>& right_hand_side_at_z,
    Core::Field<2>& solution_at_z,
    const Core::Field<2>& right_hand_side_at_t,
    Core::Field<2>& solution_at_t,
    const Options& options) {

    validate_solve_arguments(right_hand_side_at_z, solution_at_z, options);

    validate_solve_arguments(right_hand_side_at_t, solution_at_t, options);

    if (&solution_at_z == &solution_at_t || &right_hand_side_at_z == &solution_at_t ||
        &right_hand_side_at_t == &solution_at_z) {

        throw std::invalid_argument("HorizontalEllipticSolver requires distinct "
                                    "Z and T solution fields, and neither "
                                    "right-hand side may alias a solution field.");
    }

    if (!options.refresh_initial_halos) {
        throw std::invalid_argument("The bounded-q2 free-slip potential solve "
                                    "requires initial halo refresh.");
    }

    // Cartesian deliberately retains its exact historical elliptic path.
    //
    // This specialization uses GeneralizedHorizontalElliptic and therefore
    // requires a geometry for which generalized_ has been constructed.
    if (grid_.geometry().kind() == Core::Geometry::GeometryKind::Cartesian) {

        throw std::invalid_argument("The bounded-q2 generalized potential solve "
                                    "does not use the Cartesian exact solver path.");
    }

    const auto& horizontal = grid_.horizontal_specification();

    if (horizontal.topology.q1 != Core::HorizontalEdgeTopology::Periodic ||
        horizontal.topology.q2 != Core::HorizontalEdgeTopology::Bounded) {

        throw std::invalid_argument("The bounded-q2 free-slip potential solve "
                                    "requires periodic q1 and bounded q2 topology.");
    }

    Core::Boundary::HorizontalBoundaryStencils boundary(grid_);

    Core::Field<2>* current_at_z = &solution_at_z;

    Core::Field<2>* previous_at_z = &scratch_at_z_;

    Core::Field<2>* current_at_t = &solution_at_t;

    Core::Field<2>* previous_at_t = &scratch_at_t_;

    const auto refresh = [&](Core::Field<2>& z, Core::Field<2>& t) {
        halo_exchanger_.exchange_multiple_halos({&z, &t}, 1);

        boundary.fill_positive_face_q2_dirichlet_halos(z,
            options.psi_q2_minus,
            options.psi_q2_plus);

        boundary.fill_centered_q2_neumann_halos(t);
    };

    refresh(solution_at_z, solution_at_t);

    for (int iteration = 0; iteration < options.iterations; ++iteration) {

        std::swap(current_at_z, previous_at_z);

        std::swap(current_at_t, previous_at_t);

        relax_generalized_pair(right_hand_side_at_z,
            *previous_at_z,
            *current_at_z,
            right_hand_side_at_t,
            *previous_at_t,
            *current_at_t,
            options,
            true);

        refresh(*current_at_z, *current_at_t);
    }

    if (current_at_z != &solution_at_z) {

        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
            solution_at_z.get_mutable_device_data(),
            current_at_z->get_device_data());
    }

    if (current_at_t != &solution_at_t) {

        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
            solution_at_t.get_mutable_device_data(),
            current_at_t->get_device_data());
    }
}

} // namespace VVM::Dynamics
