#include "dynamics/solvers/HorizontalEllipticSolver.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace VVM {
namespace Dynamics {

HorizontalEllipticSolver::HorizontalEllipticSolver(const Core::Grid& grid, Core::HaloExchanger& halo_exchanger)
    : grid_(grid),
      halo_exchanger_(halo_exchanger),
      laplace_beltrami_(Operators::make_horizontal_laplace_beltrami_device_view(grid.geometry())),
      scratch_at_z_("horizontal_elliptic_scratch_at_z", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      scratch_at_t_("horizontal_elliptic_scratch_at_t", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}) {

    const auto& horizontal = grid_.horizontal_specification();

    if (grid_.get_halo_cells() < 1) {
        throw std::invalid_argument("HorizontalEllipticSolver requires at least one horizontal halo cell.");
    }

    if (horizontal.nx > 1 && horizontal.topology.q1 == Core::HorizontalEdgeTopology::Bounded) {
        throw std::invalid_argument("HorizontalEllipticSolver does not yet support bounded q1 topology.");
    }

    if (horizontal.ny > 1 && horizontal.topology.q2 == Core::HorizontalEdgeTopology::Bounded) {
        bounded_q2_stencils_ = std::make_unique<Core::Boundary::HorizontalBoundaryStencils>(grid_);
    }
}

void HorizontalEllipticSolver::validate_field_extents(const Core::Field<2>& field, const char* role) const {
    const auto& data = field.get_device_data();
    const int expected_ny = grid_.get_local_total_points_y();
    const int expected_nx = grid_.get_local_total_points_x();

    if (static_cast<int>(data.extent(0)) != expected_ny ||
        static_cast<int>(data.extent(1)) != expected_nx) {

        throw std::invalid_argument(
            std::string("HorizontalEllipticSolver ") + role +
            " field '" + field.get_name() +
            "' does not match the Grid horizontal extents.");
    }
}

void HorizontalEllipticSolver::validate_solve_arguments(
    const Core::Field<2>& right_hand_side,
    const Core::Field<2>& solution,
    const Options& options) const {

    validate_field_extents(right_hand_side, "right-hand-side");
    validate_field_extents(solution, "solution");

    if (&right_hand_side == &solution) {
        throw std::invalid_argument("HorizontalEllipticSolver requires distinct right-hand-side and solution fields.");
    }

    if (options.iterations <= 0) {
        throw std::invalid_argument("HorizontalEllipticSolver requires options.iterations > 0.");
    }

    if (!std::isfinite(options.diagonal_shift) ||
        options.diagonal_shift < VVM::real(0.0)) {

        throw std::invalid_argument("HorizontalEllipticSolver requires a finite nonnegative diagonal shift.");
    }
}

void HorizontalEllipticSolver::make_extrapolated_guess(
    const Core::Field<2>& current, const Core::Field<2>& previous, Core::Field<2>& guess) const {

    validate_field_extents(current, "current-solution");
    validate_field_extents(previous, "previous-solution");
    validate_field_extents(guess, "initial-guess");

    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const auto current_data = current.get_device_data();
    const auto previous_data = previous.get_device_data();
    auto guess_data = guess.get_mutable_device_data();

    Kokkos::parallel_for("MakeHorizontalEllipticExtrapolatedGuess",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            guess_data(j, i) = VVM::real(2.0) * current_data(j, i) - previous_data(j, i);
        }
    );
}

void HorizontalEllipticSolver::refresh_solution_halos(Core::Field<2>& field) {
    halo_exchanger_.exchange_halos(field, 1);

    if (bounded_q2_stencils_) {
        bounded_q2_stencils_->fill_constant_q2_halos(field);
    }
}

void HorizontalEllipticSolver::refresh_solution_halos(Core::Field<2>& first, Core::Field<2>& second) {
    halo_exchanger_.exchange_multiple_halos({&first, &second}, 1);

    if (bounded_q2_stencils_) {
        bounded_q2_stencils_->fill_constant_q2_halos(first);
        bounded_q2_stencils_->fill_constant_q2_halos(second);
    }
}

void HorizontalEllipticSolver::solve_at_t(
    const Core::Field<2>& right_hand_side, Core::Field<2>& solution, const Options& options) {

    validate_solve_arguments(right_hand_side, solution, options);

    const int halo = grid_.get_halo_cells();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const auto right_hand_side_data = right_hand_side.get_device_data();
    const auto laplace_beltrami = laplace_beltrami_;
    const VVM::Real diagonal_shift = options.diagonal_shift;

    Core::Field<2>* current = &solution;
    Core::Field<2>* previous = &scratch_at_t_;

    refresh_solution_halos(solution);

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        std::swap(current, previous);

        const auto previous_data = previous->get_device_data();
        auto current_data = current->get_mutable_device_data();

        Kokkos::parallel_for("RelaxHorizontalEllipticAtT",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({halo, halo}, {ny - halo, nx - halo}),
            KOKKOS_LAMBDA(const int j, const int i) {
                const VVM::Real operator_value = laplace_beltrami.calculate_jacobian_weighted_at_t(previous_data, j, i);
                const VVM::Real diagonal = laplace_beltrami.jacobian_weighted_diagonal_at_t(j, i);

                const VVM::Real weighted_right_hand_side = laplace_beltrami.divergence.t.sqrt_g(j, i) * right_hand_side_data(j, i);

                current_data(j, i) = previous_data(j, i) + (operator_value - weighted_right_hand_side) / (diagonal_shift - diagonal);
            });

        refresh_solution_halos(*current);
    }

    if (current != &solution) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), solution.get_mutable_device_data(), current->get_device_data());
    }
}

void HorizontalEllipticSolver::solve_at_z(
    const Core::Field<2>& right_hand_side,
    Core::Field<2>& solution,
    const Options& options) {

    validate_solve_arguments(right_hand_side, solution, options);

    const int halo = grid_.get_halo_cells();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const auto right_hand_side_data = right_hand_side.get_device_data();
    const auto laplace_beltrami = laplace_beltrami_;
    const VVM::Real diagonal_shift = options.diagonal_shift;

    Core::Field<2>* current = &solution;
    Core::Field<2>* previous = &scratch_at_z_;

    refresh_solution_halos(solution);

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        std::swap(current, previous);

        const auto previous_data = previous->get_device_data();
        auto current_data = current->get_mutable_device_data();

        Kokkos::parallel_for("RelaxHorizontalEllipticAtZ",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({halo, halo}, {ny - halo, nx - halo}),
            KOKKOS_LAMBDA(const int j, const int i) {
                const VVM::Real operator_value = laplace_beltrami.calculate_jacobian_weighted_at_z(previous_data, j, i);
                const VVM::Real diagonal = laplace_beltrami.jacobian_weighted_diagonal_at_z(j, i);
                const VVM::Real weighted_right_hand_side = laplace_beltrami.divergence.z.sqrt_g(j, i) * right_hand_side_data(j, i);

                current_data(j, i) = previous_data(j, i) + (operator_value - weighted_right_hand_side) / (diagonal_shift - diagonal);
            });

        refresh_solution_halos(*current);
    }

    if (current != &solution) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), solution.get_mutable_device_data(), current->get_device_data());
    }
}

void HorizontalEllipticSolver::solve_at_z_and_t(
    const Core::Field<2>& right_hand_side_at_z, Core::Field<2>& solution_at_z,
    const Core::Field<2>& right_hand_side_at_t, Core::Field<2>& solution_at_t,
    const Options& options) {

    validate_solve_arguments(right_hand_side_at_z, solution_at_z, options);
    validate_solve_arguments(right_hand_side_at_t, solution_at_t, options);

    if (&solution_at_z == &solution_at_t ||
        &right_hand_side_at_z == &solution_at_t ||
        &right_hand_side_at_t == &solution_at_z) {

        throw std::invalid_argument(
            "HorizontalEllipticSolver requires distinct Z and T "
            "solution fields, and neither right-hand side may alias "
            "a solution field.");
    }

    const int halo = grid_.get_halo_cells();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    const auto right_hand_side_at_z_data = right_hand_side_at_z.get_device_data();
    const auto right_hand_side_at_t_data = right_hand_side_at_t.get_device_data();

    const auto laplace_beltrami = laplace_beltrami_;

    const VVM::Real diagonal_shift = options.diagonal_shift;

    Core::Field<2>* current_at_z = &solution_at_z;
    Core::Field<2>* previous_at_z = &scratch_at_z_;

    Core::Field<2>* current_at_t = &solution_at_t;
    Core::Field<2>* previous_at_t = &scratch_at_t_;

    refresh_solution_halos(solution_at_z, solution_at_t);

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        std::swap(current_at_z, previous_at_z);
        std::swap(current_at_t, previous_at_t);

        const auto previous_at_z_data = previous_at_z->get_device_data();
        const auto previous_at_t_data = previous_at_t->get_device_data();

        auto current_at_z_data = current_at_z->get_mutable_device_data();
        auto current_at_t_data = current_at_t->get_mutable_device_data();

        Kokkos::parallel_for("RelaxHorizontalEllipticAtZAndT",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({halo, halo}, {ny - halo, nx - halo}),
            KOKKOS_LAMBDA(const int j, const int i) {
                const VVM::Real operator_at_z = laplace_beltrami.calculate_jacobian_weighted_at_z(previous_at_z_data, j, i);
                const VVM::Real diagonal_at_z = laplace_beltrami.jacobian_weighted_diagonal_at_z(j, i);

                const VVM::Real weighted_right_hand_side_at_z = laplace_beltrami.divergence.z.sqrt_g(j, i) * right_hand_side_at_z_data(j, i);

                current_at_z_data(j, i) = previous_at_z_data(j, i) + (operator_at_z - weighted_right_hand_side_at_z) / (diagonal_shift - diagonal_at_z);

                const VVM::Real operator_at_t = laplace_beltrami.calculate_jacobian_weighted_at_t(previous_at_t_data, j, i);

                const VVM::Real diagonal_at_t = laplace_beltrami.jacobian_weighted_diagonal_at_t(j, i);
                const VVM::Real weighted_right_hand_side_at_t = laplace_beltrami.divergence.t.sqrt_g(j, i) * right_hand_side_at_t_data(j, i);

                current_at_t_data(j, i) = previous_at_t_data(j, i) + (operator_at_t - weighted_right_hand_side_at_t) / (diagonal_shift - diagonal_at_t);
            }
        );

        refresh_solution_halos(*current_at_z, *current_at_t);
    }

    if (current_at_z != &solution_at_z) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), solution_at_z.get_mutable_device_data(), current_at_z->get_device_data());
    }

    if (current_at_t != &solution_at_t) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), solution_at_t.get_mutable_device_data(), current_at_t->get_device_data());
    }
}

} // namespace Dynamics
} // namespace VVM
