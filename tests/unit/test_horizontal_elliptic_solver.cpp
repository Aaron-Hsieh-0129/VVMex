#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include "core/geometry/GeometryKind.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>

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
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Dynamics::HorizontalEllipticSolver;
using VVM::Dynamics::Operators::ScalarStencilAtT;
using VVM::Dynamics::Operators::ScalarStencilAtZ;
using VVM::Dynamics::Operators::make_horizontal_laplace_beltrami_device_view;
using VVM::Utils::ConfigurationManager;

int failures = 0;
int mpi_rank = 0;

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(stderr, "Rank %d FAIL: %s\n", mpi_rank, message);
}

void refresh_halos(const Grid& grid, HaloExchanger& halo_exchanger, Field<2>& field) {
    halo_exchanger.exchange_halos(field, 1);

    const auto& horizontal = grid.horizontal_specification();
    if (horizontal.ny > 1 && horizontal.topology.q2 == HorizontalEdgeTopology::Bounded) {
        HorizontalBoundaryStencils boundary_stencils(grid);
        boundary_stencils.fill_constant_q2_halos(field);
    }
}

void initialize_fields(const Grid& grid, Field<2>& right_hand_side, Field<2>& solution_at_t, Field<2>& solution_at_z) {
    const int halo = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int global_start_j = grid.get_local_physical_start_y();
    const int global_start_i = grid.get_local_physical_start_x();

    auto rhs = right_hand_side.get_mutable_device_data();
    auto t = solution_at_t.get_mutable_device_data();
    auto z = solution_at_z.get_mutable_device_data();

    Kokkos::parallel_for(
        "InitializeHorizontalEllipticSolverTest",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real global_j = static_cast<VVM::Real>(global_start_j + j - halo);
            const VVM::Real global_i = static_cast<VVM::Real>(global_start_i + i - halo);

            rhs(j, i) = VVM::real(1.0e-12) * (VVM::real(0.75) + Kokkos::sin(VVM::real(0.17) * global_i) * Kokkos::cos(VVM::real(0.11) * global_j));
            t(j, i) = VVM::real(2.0) + VVM::real(0.03) * global_i - VVM::real(0.02) * global_j + VVM::real(0.001) * global_i * global_j;
            z(j, i) = t(j, i);
        });
}

void test_extrapolated_guess(const Grid& grid, HorizontalEllipticSolver& solver) {
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Field<2> current("extrapolation_current", {ny, nx});
    Field<2> previous("extrapolation_previous", {ny, nx});
    Field<2> guess("extrapolation_guess", {ny, nx});

    auto current_data = current.get_mutable_device_data();
    auto previous_data = previous.get_mutable_device_data();

    Kokkos::parallel_for(
        "InitializeHorizontalEllipticExtrapolationTest",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            current_data(j, i) = VVM::real(3.0) + VVM::real(0.2) * i - VVM::real(0.1) * j;
            previous_data(j, i) = VVM::real(1.0) - VVM::real(0.05) * i + VVM::real(0.3) * j;
        });

    solver.make_extrapolated_guess(current, previous, guess);

    const auto current_host = current.get_host_data();
    const auto previous_host = previous.get_host_data();
    const auto guess_host = guess.get_host_data();

    VVM::Real maximum_error = VVM::real(0.0);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const VVM::Real expected = VVM::real(2.0) * current_host(j, i) - previous_host(j, i);
            maximum_error = std::max(maximum_error, std::abs(guess_host(j, i) - expected));
        }
    }

    check(maximum_error == VVM::real(0.0), "The initial guess must equal 2 * current - previous in every cell");
}

void test_staggered_operator_diagonals(const Grid& grid) {
    const int halo = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int i = halo;
    const auto laplace_beltrami = make_horizontal_laplace_beltrami_device_view(grid.geometry());

    Kokkos::View<VVM::Real*> applied_at_t("horizontal_elliptic_diagonal_applied_t", ny);
    Kokkos::View<VVM::Real*> diagonal_at_t("horizontal_elliptic_diagonal_t", ny);
    Kokkos::View<VVM::Real*> applied_at_z("horizontal_elliptic_diagonal_applied_z", ny);
    Kokkos::View<VVM::Real*> diagonal_at_z("horizontal_elliptic_diagonal_z", ny);

    Kokkos::parallel_for(
        "CompareHorizontalEllipticStaggeredDiagonals",
        Kokkos::RangePolicy<>(halo, ny - halo),
        KOKKOS_LAMBDA(const int j) {
            ScalarStencilAtT impulse_at_t;
            ScalarStencilAtZ impulse_at_z;
            impulse_at_t.center = VVM::real(1.0);
            impulse_at_z.center = VVM::real(1.0);

            applied_at_t(j) = laplace_beltrami.calculate_jacobian_weighted_at_t(j, i, impulse_at_t);
            diagonal_at_t(j) = laplace_beltrami.jacobian_weighted_diagonal_at_t(j, i);
            applied_at_z(j) = laplace_beltrami.calculate_jacobian_weighted_at_z(j, i, impulse_at_z);
            diagonal_at_z(j) = laplace_beltrami.jacobian_weighted_diagonal_at_z(j, i);
        });

    const auto applied_t_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), applied_at_t);
    const auto diagonal_t_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonal_at_t);
    const auto applied_z_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), applied_at_z);
    const auto diagonal_z_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonal_at_z);

    const VVM::Real relative_tolerance = sizeof(VVM::Real) == sizeof(double) ? VVM::real(2.0e-13) : VVM::real(2.0e-5);
    for (int j = halo; j < ny - halo; ++j) {
        const VVM::Real t_scale = std::max(VVM::real(1.0), std::abs(diagonal_t_host(j)));
        const VVM::Real z_scale = std::max(VVM::real(1.0), std::abs(diagonal_z_host(j)));

        check(std::abs(applied_t_host(j) - diagonal_t_host(j)) <= relative_tolerance * t_scale, "The weighted T diagonal must equal the response to a T center impulse");
        check(std::abs(applied_z_host(j) - diagonal_z_host(j)) <= relative_tolerance * z_scale, "The weighted Z diagonal must equal the response to a Z center impulse");
    }
}

void test_one_iteration(const Grid& grid, HaloExchanger& halo_exchanger, HorizontalEllipticSolver& solver) {
    const int halo = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const VVM::Real diagonal_shift = VVM::real(2.5e-7);

    Field<2> right_hand_side("horizontal_elliptic_rhs", {ny, nx});
    Field<2> solution_at_t("horizontal_elliptic_solution_t", {ny, nx});
    Field<2> solution_at_z("horizontal_elliptic_solution_z", {ny, nx});

    initialize_fields(grid, right_hand_side, solution_at_t, solution_at_z);
    refresh_halos(grid, halo_exchanger, solution_at_t);
    refresh_halos(grid, halo_exchanger, solution_at_z);

    Kokkos::View<VVM::Real**> expected_at_t("horizontal_elliptic_expected_t", ny, nx);
    Kokkos::View<VVM::Real**> expected_at_z("horizontal_elliptic_expected_z", ny, nx);

    const auto rhs = right_hand_side.get_device_data();
    const auto initial_t = solution_at_t.get_device_data();
    const auto initial_z = solution_at_z.get_device_data();
    const auto t = grid.geometry().device_view(HorizontalLocation::T);
    const auto u = grid.geometry().device_view(HorizontalLocation::U);
    const auto v = grid.geometry().device_view(HorizontalLocation::V);
    const auto z = grid.geometry().device_view(HorizontalLocation::Z);
    const VVM::Real inverse_dq1_squared = VVM::real(1.0) / (grid.geometry().dq1() * grid.geometry().dq1());
    const VVM::Real inverse_dq2_squared = VVM::real(1.0) / (grid.geometry().dq2() * grid.geometry().dq2());

    Kokkos::parallel_for(
        "BuildHorizontalEllipticOneIterationReference",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({halo, halo}, {ny - halo, nx - halo}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const VVM::Real t_q1_west = u.sqrt_g_g_contra.a11(j, i - 1) * inverse_dq1_squared;
            const VVM::Real t_q1_east = u.sqrt_g_g_contra.a11(j, i) * inverse_dq1_squared;
            const VVM::Real t_q2_south = v.sqrt_g_g_contra.a22(j - 1, i) * inverse_dq2_squared;
            const VVM::Real t_q2_north = v.sqrt_g_g_contra.a22(j, i) * inverse_dq2_squared;
            const VVM::Real t_denominator = diagonal_shift + t_q1_west + t_q1_east + t_q2_south + t_q2_north;

            expected_at_t(j, i) = (
                diagonal_shift * initial_t(j, i) +
                t_q1_west * initial_t(j, i - 1) +
                t_q1_east * initial_t(j, i + 1) +
                t_q2_south * initial_t(j - 1, i) +
                t_q2_north * initial_t(j + 1, i) -
                t.sqrt_g(j, i) * rhs(j, i)) / t_denominator;

            const VVM::Real z_q1_west = v.sqrt_g_g_contra.a11(j, i) * inverse_dq1_squared;
            const VVM::Real z_q1_east = v.sqrt_g_g_contra.a11(j, i + 1) * inverse_dq1_squared;
            const VVM::Real z_q2_south = u.sqrt_g_g_contra.a22(j, i) * inverse_dq2_squared;
            const VVM::Real z_q2_north = u.sqrt_g_g_contra.a22(j + 1, i) * inverse_dq2_squared;
            const VVM::Real z_denominator = diagonal_shift + z_q1_west + z_q1_east + z_q2_south + z_q2_north;

            expected_at_z(j, i) = (
                diagonal_shift * initial_z(j, i) +
                z_q1_west * initial_z(j, i - 1) +
                z_q1_east * initial_z(j, i + 1) +
                z_q2_south * initial_z(j - 1, i) +
                z_q2_north * initial_z(j + 1, i) -
                z.sqrt_g(j, i) * rhs(j, i)) / z_denominator;
        });

    const HorizontalEllipticSolver::Options options{1, diagonal_shift};
    solver.solve_at_t(right_hand_side, solution_at_t, options);
    solver.solve_at_z(right_hand_side, solution_at_z, options);

    const auto actual_t_host = solution_at_t.get_host_data();
    const auto actual_z_host = solution_at_z.get_host_data();
    const auto expected_t_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected_at_t);
    const auto expected_z_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected_at_z);

    VVM::Real maximum_t_error = VVM::real(0.0);
    VVM::Real maximum_z_error = VVM::real(0.0);
    VVM::Real maximum_t_scale = VVM::real(0.0);
    VVM::Real maximum_z_scale = VVM::real(0.0);

    for (int j = halo; j < ny - halo; ++j) {
        for (int i = halo; i < nx - halo; ++i) {
            maximum_t_error = std::max(maximum_t_error, std::abs(actual_t_host(j, i) - expected_t_host(j, i)));
            maximum_z_error = std::max(maximum_z_error, std::abs(actual_z_host(j, i) - expected_z_host(j, i)));
            maximum_t_scale = std::max(maximum_t_scale, std::abs(expected_t_host(j, i)));
            maximum_z_scale = std::max(maximum_z_scale, std::abs(expected_z_host(j, i)));
        }
    }

    const VVM::Real relative_tolerance = sizeof(VVM::Real) == sizeof(double) ? VVM::real(2.0e-11) : VVM::real(2.0e-4);
    check(maximum_t_error <= relative_tolerance * std::max(VVM::real(1.0), maximum_t_scale), "One T-point iteration must match the CVVM/VVMex shifted-Jacobi equation");
    check(maximum_z_error <= relative_tolerance * std::max(VVM::real(1.0), maximum_z_scale), "One Z-point iteration must match the CVVM/VVMex shifted-Jacobi equation");

    if (grid.geometry().kind() == GeometryKind::Cartesian) {
        VVM::Real maximum_tz_difference = VVM::real(0.0);
        for (int j = halo; j < ny - halo; ++j) {
            for (int i = halo; i < nx - halo; ++i) {
                maximum_tz_difference = std::max(maximum_tz_difference, std::abs(actual_t_host(j, i) - actual_z_host(j, i)));
            }
        }

        check(maximum_tz_difference <= relative_tolerance * std::max(VVM::real(1.0), maximum_t_scale), "Cartesian T and Z relaxation must reduce to the same legacy five-point update");
    }
}

void run_tests(Grid& grid, HaloExchanger& halo_exchanger) {
    HorizontalEllipticSolver solver(grid, halo_exchanger);
    test_extrapolated_guess(grid, solver);
    test_staggered_operator_diagonals(grid);
    test_one_iteration(grid, halo_exchanger, solver);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    Kokkos::initialize(argc, argv);

    {
        try {
            if (argc < 2) {
                throw std::invalid_argument("Usage: test_horizontal_elliptic_solver <configuration.json>");
            }

            const ConfigurationManager config(argv[1]);
            Grid grid(config);

#if defined(ENABLE_NCCL)
            int mpi_size = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

            ncclUniqueId nccl_id;
            if (mpi_rank == 0) {
                ncclGetUniqueId(&nccl_id);
            }

            MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

            ncclComm_t nccl_comm;
            ncclCommInitRank(&nccl_comm, mpi_size, nccl_id, mpi_rank);

            {
                cudaStream_t stream = Kokkos::Cuda().cuda_stream();
                HaloExchanger halo_exchanger(config, grid, nccl_comm, stream);
                run_tests(grid, halo_exchanger);
            }

            ncclCommDestroy(nccl_comm);
#else
            HaloExchanger halo_exchanger(grid);
            run_tests(grid, halo_exchanger);
#endif
        } catch (const std::exception& error) {
            ++failures;
            std::fprintf(stderr, "Rank %d unexpected exception: %s\n", mpi_rank, error.what());
        }
    }

    Kokkos::finalize();

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        if (global_failures == 0) {
            std::fprintf(stdout, "test_horizontal_elliptic_solver: PASS\n");
        } else {
            std::fprintf(stderr, "test_horizontal_elliptic_solver: %d failure(s)\n", global_failures);
        }
    }

    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}

