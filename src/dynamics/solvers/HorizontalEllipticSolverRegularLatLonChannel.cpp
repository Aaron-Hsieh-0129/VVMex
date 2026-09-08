#include "dynamics/solvers/HorizontalEllipticSolver.hpp"

#include <stdexcept>
#include <utility>

namespace VVM {
namespace Dynamics {

void HorizontalEllipticSolver::solve_regular_lat_lon_channel_at_z_and_t(
    const Core::Field<2>& right_hand_side_at_z,
    Core::Field<2>& solution_at_z,
    const Core::Field<2>& right_hand_side_at_t,
    Core::Field<2>& solution_at_t,
    const Options& options) {

    validate_solve_arguments(
        right_hand_side_at_z,
        solution_at_z,
        options);

    validate_solve_arguments(
        right_hand_side_at_t,
        solution_at_t,
        options);

    if (&solution_at_z == &solution_at_t ||
        &right_hand_side_at_z == &solution_at_t ||
        &right_hand_side_at_t == &solution_at_z) {

        throw std::invalid_argument(
            "HorizontalEllipticSolver requires distinct Z and T solution fields, "
            "and neither right-hand side may alias a solution field.");
    }

    if (!options.refresh_initial_halos) {
        throw std::invalid_argument(
            "The RLL channel potential solve requires initial halo refresh.");
    }

    if (grid_.geometry().kind() !=
        Core::Geometry::GeometryKind::RegularLatLon) {

        throw std::invalid_argument(
            "The RLL channel potential solve requires regular latitude-longitude geometry.");
    }

    const auto& horizontal =
        grid_.horizontal_specification();

    if (horizontal.topology.q1 !=
            Core::HorizontalEdgeTopology::Periodic ||
        horizontal.topology.q2 !=
            Core::HorizontalEdgeTopology::Bounded) {

        throw std::invalid_argument(
            "The RLL channel potential solve requires periodic q1 and bounded q2.");
    }

    const int halo =
        grid_.get_halo_cells();

    const int ny =
        grid_.get_local_total_points_y();

    const int nx =
        grid_.get_local_total_points_x();

    const bool owns_north_wall =
        grid_.get_local_physical_end_y() ==
        grid_.get_global_points_y() - 1;

    const int north_wall_j =
        ny - halo - 1;

    const auto right_hand_side_at_z_data =
        right_hand_side_at_z.get_device_data();

    const auto right_hand_side_at_t_data =
        right_hand_side_at_t.get_device_data();

    const auto metrics =
        regular_lat_lon_metrics_;

    const VVM::Real diagonal_shift =
        options.diagonal_shift;

    Core::Boundary::HorizontalBoundaryStencils boundary(
        grid_);

    Core::Field<2>* current_at_z =
        &solution_at_z;

    Core::Field<2>* previous_at_z =
        &scratch_at_z_;

    Core::Field<2>* current_at_t =
        &solution_at_t;

    Core::Field<2>* previous_at_t =
        &scratch_at_t_;

    const auto refresh =
        [&](Core::Field<2>& z,
            Core::Field<2>& t) {

            halo_exchanger_.exchange_multiple_halos(
                {&z, &t},
                1);

            boundary
                .fill_positive_face_q2_dirichlet_halos(
                    z, options.channel_psi_south, options.channel_psi_north);

            boundary
                .fill_centered_q2_neumann_halos(
                    t);
        };

    refresh(
        solution_at_z,
        solution_at_t);

    for (int iteration = 0;
         iteration < options.iterations;
         ++iteration) {

        std::swap(
            current_at_z,
            previous_at_z);

        std::swap(
            current_at_t,
            previous_at_t);

        const auto previous_at_z_data =
            previous_at_z->get_device_data();

        const auto previous_at_t_data =
            previous_at_t->get_device_data();

        auto current_at_z_data =
            current_at_z->get_mutable_device_data();

        auto current_at_t_data =
            current_at_t->get_mutable_device_data();

        const auto policy =
            Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {halo, halo},
                    {ny - halo, nx - halo}),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);

        Kokkos::parallel_for(
            "RelaxHorizontalEllipticRLLChannelAtZAndT",
            policy,
            KOKKOS_LAMBDA(
                const int j,
                const int i) {

                if (owns_north_wall &&
                    j == north_wall_j) {

                    current_at_z_data(j, i) =
                        VVM::real(0.0);
                } else {
                    const VVM::Real gradient_z_q1_plus =
                        metrics.inv_sqrt_g_at_v(j) *
                        metrics.sqrt_g_g_contra_11_at_v(j) *
                        (
                            previous_at_z_data(j, i + 1) -
                            previous_at_z_data(j, i)
                        ) /
                        metrics.dq1;

                    const VVM::Real gradient_z_q1_minus =
                        metrics.inv_sqrt_g_at_v(j) *
                        metrics.sqrt_g_g_contra_11_at_v(j) *
                        (
                            previous_at_z_data(j, i) -
                            previous_at_z_data(j, i - 1)
                        ) /
                        metrics.dq1;

                    const VVM::Real gradient_z_q2_plus =
                        metrics.inv_sqrt_g_at_u(j + 1) *
                        metrics.sqrt_g_g_contra_22_at_u(j + 1) *
                        (
                            previous_at_z_data(j + 1, i) -
                            previous_at_z_data(j, i)
                        ) /
                        metrics.dq2;

                    const VVM::Real gradient_z_q2_minus =
                        metrics.inv_sqrt_g_at_u(j) *
                        metrics.sqrt_g_g_contra_22_at_u(j) *
                        (
                            previous_at_z_data(j, i) -
                            previous_at_z_data(j - 1, i)
                        ) /
                        metrics.dq2;

                    const VVM::Real operator_at_z =
                        (
                            metrics.sqrt_g_at_v(j) *
                                gradient_z_q1_plus -
                            metrics.sqrt_g_at_v(j) *
                                gradient_z_q1_minus
                        ) /
                            metrics.dq1 +
                        (
                            metrics.sqrt_g_at_u(j + 1) *
                                gradient_z_q2_plus -
                            metrics.sqrt_g_at_u(j) *
                                gradient_z_q2_minus
                        ) /
                            metrics.dq2;

                    const VVM::Real diagonal_at_z =
                        -(
                            (
                                metrics.sqrt_g_g_contra_11_at_v(j) +
                                metrics.sqrt_g_g_contra_11_at_v(j)
                            ) *
                                metrics.inverse_dq1_squared +
                            (
                                metrics.sqrt_g_g_contra_22_at_u(j + 1) +
                                metrics.sqrt_g_g_contra_22_at_u(j)
                            ) *
                                metrics.inverse_dq2_squared
                        );

                    current_at_z_data(j, i) =
                        previous_at_z_data(j, i) +
                        (
                            operator_at_z -
                            metrics.sqrt_g_at_z(j) *
                                right_hand_side_at_z_data(j, i)
                        ) /
                        (
                            diagonal_shift -
                            diagonal_at_z
                        );
                }

                const VVM::Real gradient_t_q1_plus =
                    metrics.inv_sqrt_g_at_u(j) *
                    metrics.sqrt_g_g_contra_11_at_u(j) *
                    (
                        previous_at_t_data(j, i + 1) -
                        previous_at_t_data(j, i)
                    ) /
                    metrics.dq1;

                const VVM::Real gradient_t_q1_minus =
                    metrics.inv_sqrt_g_at_u(j) *
                    metrics.sqrt_g_g_contra_11_at_u(j) *
                    (
                        previous_at_t_data(j, i) -
                        previous_at_t_data(j, i - 1)
                    ) /
                    metrics.dq1;

                const VVM::Real gradient_t_q2_plus =
                    metrics.inv_sqrt_g_at_v(j) *
                    metrics.sqrt_g_g_contra_22_at_v(j) *
                    (
                        previous_at_t_data(j + 1, i) -
                        previous_at_t_data(j, i)
                    ) /
                    metrics.dq2;

                const VVM::Real gradient_t_q2_minus =
                    metrics.inv_sqrt_g_at_v(j - 1) *
                    metrics.sqrt_g_g_contra_22_at_v(j - 1) *
                    (
                        previous_at_t_data(j, i) -
                        previous_at_t_data(j - 1, i)
                    ) /
                    metrics.dq2;

                const VVM::Real operator_at_t =
                    (
                        metrics.sqrt_g_at_u(j) *
                            gradient_t_q1_plus -
                        metrics.sqrt_g_at_u(j) *
                            gradient_t_q1_minus
                    ) /
                        metrics.dq1 +
                    (
                        metrics.sqrt_g_at_v(j) *
                            gradient_t_q2_plus -
                        metrics.sqrt_g_at_v(j - 1) *
                            gradient_t_q2_minus
                    ) /
                        metrics.dq2;

                const VVM::Real diagonal_at_t =
                    -(
                        (
                            metrics.sqrt_g_g_contra_11_at_u(j) +
                            metrics.sqrt_g_g_contra_11_at_u(j)
                        ) *
                            metrics.inverse_dq1_squared +
                        (
                            metrics.sqrt_g_g_contra_22_at_v(j) +
                            metrics.sqrt_g_g_contra_22_at_v(j - 1)
                        ) *
                            metrics.inverse_dq2_squared
                    );

                current_at_t_data(j, i) =
                    previous_at_t_data(j, i) +
                    (
                        operator_at_t -
                        metrics.sqrt_g_at_t(j) *
                            right_hand_side_at_t_data(j, i)
                    ) /
                    (
                        diagonal_shift -
                        diagonal_at_t
                    );
            });

        refresh(
            *current_at_z,
            *current_at_t);
    }

    if (current_at_z !=
        &solution_at_z) {

        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace(),
            solution_at_z.get_mutable_device_data(),
            current_at_z->get_device_data());
    }

    if (current_at_t !=
        &solution_at_t) {

        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace(),
            solution_at_t.get_mutable_device_data(),
            current_at_t->get_device_data());
    }
}

} // namespace Dynamics
} // namespace VVM
