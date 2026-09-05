#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <utility>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Core::HorizontalEdgeTopology;
using VVM::Core::Boundary::HorizontalBoundaryStencils;

struct ErrorNorms {
    double solution_linf = 0.0;
    double relative_residual_linf = 0.0;
    double solution_mean = 0.0;
    bool finite = true;
};

int mpi_rank = 0;
int failures = 0;

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "Rank %d FAIL: %s\n", mpi_rank, message);
}

void initialize_manufactured_problem(
    const Grid& grid,
    Field<2>& exact_solution,
    Field<2>& right_hand_side) {

    const int h = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    const int local_start_y = grid.get_local_physical_start_y();
    const int local_start_x = grid.get_local_physical_start_x();
    const int global_ny = grid.get_global_points_y();
    const int global_nx = grid.get_global_points_x();

    const VVM::Real pi = VVM::real(std::acos(-1.0));
    const VVM::Real inverse_dx_squared =
        VVM::real(1.0) / (grid.get_dx() * grid.get_dx());
    const VVM::Real inverse_dy_squared =
        VVM::real(1.0) / (grid.get_dy() * grid.get_dy());

    const VVM::Real sin_x =
        VVM::real(std::sin(std::acos(-1.0) / static_cast<double>(global_nx)));
    const VVM::Real sin_y =
        VVM::real(std::sin(std::acos(-1.0) / (2.0 * static_cast<double>(global_ny))));

    const VVM::Real eigenvalue_x =
        -VVM::real(4.0) * inverse_dx_squared * sin_x * sin_x;
    const VVM::Real eigenvalue_y =
        -VVM::real(4.0) * inverse_dy_squared * sin_y * sin_y;
    const VVM::Real laplacian_eigenvalue = eigenvalue_x + eigenvalue_y;

    auto exact = exact_solution.get_mutable_device_data();
    auto rhs = right_hand_side.get_mutable_device_data();

    Kokkos::parallel_for(
        "InitializeBoundedQ2PoissonProblem",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {h, h},
            {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const int global_i = local_start_x + i - h;
            const int global_j = local_start_y + j - h;

            const VVM::Real q1_phase =
                VVM::real(2.0) * pi *
                (static_cast<VVM::Real>(global_i) + VVM::real(0.5)) /
                static_cast<VVM::Real>(global_nx);

            const VVM::Real q2_phase =
                pi *
                (static_cast<VVM::Real>(global_j) + VVM::real(0.5)) /
                static_cast<VVM::Real>(global_ny);

            const VVM::Real value =
                Kokkos::cos(q1_phase) * Kokkos::cos(q2_phase);

            exact(j, i) = value;
            rhs(j, i) = laplacian_eigenvalue * value;
        });

    Kokkos::fence();
}

ErrorNorms calculate_error_norms(
    const Grid& grid,
    const Field<2>& solution,
    const Field<2>& exact_solution,
    const Field<2>& right_hand_side) {

    const int h = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    const double inverse_dx_squared =
        1.0 / (static_cast<double>(grid.get_dx()) * static_cast<double>(grid.get_dx()));
    const double inverse_dy_squared =
        1.0 / (static_cast<double>(grid.get_dy()) * static_cast<double>(grid.get_dy()));

    const auto solution_host = solution.get_host_data();
    const auto exact_host = exact_solution.get_host_data();
    const auto rhs_host = right_hand_side.get_host_data();

    double local_solution_linf = 0.0;
    double local_residual_linf = 0.0;
    double local_rhs_linf = 0.0;
    double local_solution_sum = 0.0;
    long long local_point_count = 0;
    int local_finite = 1;

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            const double value = static_cast<double>(solution_host(j, i));
            const double exact = static_cast<double>(exact_host(j, i));
            const double rhs = static_cast<double>(rhs_host(j, i));

            const double laplacian =
                inverse_dx_squared *
                    (static_cast<double>(solution_host(j, i - 1)) -
                     2.0 * value +
                     static_cast<double>(solution_host(j, i + 1))) +
                inverse_dy_squared *
                    (static_cast<double>(solution_host(j - 1, i)) -
                     2.0 * value +
                     static_cast<double>(solution_host(j + 1, i)));

            local_solution_linf =
                std::max(local_solution_linf, std::abs(value - exact));
            local_residual_linf =
                std::max(local_residual_linf, std::abs(laplacian - rhs));
            local_rhs_linf =
                std::max(local_rhs_linf, std::abs(rhs));

            local_solution_sum += value;
            ++local_point_count;

            if (!std::isfinite(value) || !std::isfinite(laplacian)) {
                local_finite = 0;
            }
        }
    }

    double global_solution_linf = 0.0;
    double global_residual_linf = 0.0;
    double global_rhs_linf = 0.0;
    double global_solution_sum = 0.0;
    long long global_point_count = 0;
    int global_finite = 0;

    MPI_Allreduce(
        &local_solution_linf,
        &global_solution_linf,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD);

    MPI_Allreduce(
        &local_residual_linf,
        &global_residual_linf,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD);

    MPI_Allreduce(
        &local_rhs_linf,
        &global_rhs_linf,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD);

    MPI_Allreduce(
        &local_solution_sum,
        &global_solution_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD);

    MPI_Allreduce(
        &local_point_count,
        &global_point_count,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        MPI_COMM_WORLD);

    MPI_Allreduce(
        &local_finite,
        &global_finite,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD);

    const double residual_scale =
        std::max(global_rhs_linf, std::numeric_limits<double>::min());

    ErrorNorms result;
    result.solution_linf = global_solution_linf;
    result.relative_residual_linf = global_residual_linf / residual_scale;
    result.solution_mean =
        global_solution_sum / static_cast<double>(global_point_count);
    result.finite = global_finite != 0;

    return result;
}

ErrorNorms solve_manufactured_problem(
    const Grid& grid,
    HaloExchanger& halo_exchanger) {

    const int h = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Field<2> exact_solution("bounded_q2_exact", {ny, nx});
    Field<2> right_hand_side("bounded_q2_rhs", {ny, nx});
    Field<2> solution_a("bounded_q2_solution_a", {ny, nx});
    Field<2> solution_b("bounded_q2_solution_b", {ny, nx});

    initialize_manufactured_problem(
        grid,
        exact_solution,
        right_hand_side);

    solution_a.set_to_zero();
    solution_b.set_to_zero();
    Kokkos::fence();

    HorizontalBoundaryStencils boundary_stencils(grid);

    Field<2>* current = &solution_a;
    Field<2>* next = &solution_b;

    halo_exchanger.exchange_halos(*current, 1);
    boundary_stencils.fill_constant_q2_halos(*current);

    const VVM::Real inverse_dx_squared =
        VVM::real(1.0) / (grid.get_dx() * grid.get_dx());
    const VVM::Real inverse_dy_squared =
        VVM::real(1.0) / (grid.get_dy() * grid.get_dy());

    // This is the same stabilized Jacobi equation used by
    // WindSolver::relax_2d_batched(). WRXMU affects the convergence
    // rate but cancels from the converged Poisson equation.
    const VVM::Real relaxation_mass = VVM::real(2.5e-7);

    const VVM::Real inverse_diagonal =
        VVM::real(1.0) /
        (relaxation_mass +
         VVM::real(2.0) * inverse_dx_squared +
         VVM::real(2.0) * inverse_dy_squared);

    const auto rhs = right_hand_side.get_device_data();

    constexpr int iteration_count = 1500;

    for (int iteration = 0; iteration < iteration_count; ++iteration) {
        const auto previous = current->get_device_data();
        auto updated = next->get_mutable_device_data();

        Kokkos::parallel_for(
            "RelaxBoundedQ2Poisson",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {h, h},
                {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                updated(j, i) =
                    (
                        relaxation_mass * previous(j, i) +
                        inverse_dx_squared *
                            (previous(j, i - 1) + previous(j, i + 1)) +
                        inverse_dy_squared *
                            (previous(j - 1, i) + previous(j + 1, i)) -
                        rhs(j, i)
                    ) *
                    inverse_diagonal;
            });

        std::swap(current, next);

        // Internal and periodic edges are exchanged first. Physical q2
        // wall values are imposed afterward because MPI_PROC_NULL leaves
        // those halo rows untouched.
        halo_exchanger.exchange_halos(*current, 1);
        boundary_stencils.fill_constant_q2_halos(*current);
    }

    Kokkos::fence();

    return calculate_error_norms(
        grid,
        *current,
        exact_solution,
        right_hand_side);
}

void test_bounded_q2_poisson(
    const Grid& grid,
    HaloExchanger& halo_exchanger) {

    const auto& horizontal = grid.horizontal_specification();

    check(
        horizontal.topology.q1 == HorizontalEdgeTopology::Periodic,
        "Manufactured problem requires periodic q1 topology");

    check(
        horizontal.topology.q2 == HorizontalEdgeTopology::Bounded,
        "Manufactured problem requires bounded q2 topology");

    if (horizontal.topology.q1 != HorizontalEdgeTopology::Periodic ||
        horizontal.topology.q2 != HorizontalEdgeTopology::Bounded) {
        return;
    }

    const ErrorNorms norms =
        solve_manufactured_problem(grid, halo_exchanger);

    const bool double_precision =
        sizeof(VVM::Real) == sizeof(double);

    const double solution_tolerance =
        double_precision ? 1.0e-5 : 5.0e-3;

    const double residual_tolerance =
        double_precision ? 1.0e-5 : 5.0e-3;

    const double mean_tolerance =
        double_precision ? 1.0e-10 : 5.0e-5;

    if (mpi_rank == 0) {
        std::fprintf(
            stdout,
            "bounded q2 Poisson: error=%e residual=%e mean=%e\n",
            norms.solution_linf,
            norms.relative_residual_linf,
            norms.solution_mean);
    }

    check(
        norms.finite,
        "Poisson relaxation produced NaN or infinity");

    check(
        norms.solution_linf < solution_tolerance,
        "Manufactured solution error exceeds tolerance");

    check(
        norms.relative_residual_linf < residual_tolerance,
        "Relative Poisson residual exceeds tolerance");

    check(
        std::abs(norms.solution_mean) < mean_tolerance,
        "Poisson solution did not retain its zero-mean gauge");
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 2) {
        if (mpi_rank == 0) {
            std::fprintf(
                stderr,
                "usage: %s <config.json>\n",
                argv[0]);
        }

        MPI_Finalize();
        return 2;
    }

    Kokkos::initialize(
        Kokkos::InitializationSettings().set_device_id(0));

    {
        try {
            const VVM::Utils::ConfigurationManager config(argv[1]);
            const Grid grid(config, MPI_COMM_WORLD);

#if defined(ENABLE_NCCL)
            int mpi_size = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

            ncclUniqueId nccl_id;
            if (mpi_rank == 0) {
                ncclGetUniqueId(&nccl_id);
            }

            MPI_Bcast(
                &nccl_id,
                sizeof(nccl_id),
                MPI_BYTE,
                0,
                MPI_COMM_WORLD);

            ncclComm_t nccl_comm;
            ncclCommInitRank(
                &nccl_comm,
                mpi_size,
                nccl_id,
                mpi_rank);

            {
                cudaStream_t stream =
                    Kokkos::Cuda().cuda_stream();

                HaloExchanger halo_exchanger(
                    config,
                    grid,
                    nccl_comm,
                    stream);

                test_bounded_q2_poisson(
                    grid,
                    halo_exchanger);
            }

            ncclCommDestroy(nccl_comm);
#else
            HaloExchanger halo_exchanger(grid);

            test_bounded_q2_poisson(
                grid,
                halo_exchanger);
#endif
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(
                stderr,
                "Rank %d unexpected exception: %s\n",
                mpi_rank,
                error.what());
        }
    }

    Kokkos::finalize();

    int global_failures = 0;

    MPI_Allreduce(
        &failures,
        &global_failures,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        if (global_failures == 0) {
            std::fprintf(
                stdout,
                "test_bounded_q2_poisson_relaxation: PASS\n");
        } else {
            std::fprintf(
                stderr,
                "test_bounded_q2_poisson_relaxation: %d failure(s)\n",
                global_failures);
        }
    }

    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
