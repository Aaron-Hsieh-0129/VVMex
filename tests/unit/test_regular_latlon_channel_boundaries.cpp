#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/vvm_types.hpp"
#include "dynamics/operators/HorizontalLaplaceBeltrami.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Core::Boundary::HorizontalBoundaryStencils;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Dynamics::HorizontalEllipticSolver;
using VVM::Dynamics::Operators::make_horizontal_laplace_beltrami_device_view;
using VVM::Utils::ConfigurationManager;

int mpi_rank = 0;
int failures = 0;

void check(
    const bool condition,
    const char* message) {

    if (condition) {
        return;
    }

    ++failures;

    std::fprintf(
        stderr,
        "Rank %d FAIL: %s\n",
        mpi_rank,
        message);
}

[[noreturn]] void fatal(
    const char* message) {

    std::fprintf(
        stderr,
        "Rank %d fatal: %s\n",
        mpi_rank,
        message);

    MPI_Abort(
        MPI_COMM_WORLD,
        2);

    std::abort();
}

#if defined(ENABLE_NCCL)
void nccl_check(
    const ncclResult_t result) {

    if (result != ncclSuccess) {
        fatal(
            ncclGetErrorString(
                result));
    }
}
#endif

bool close_enough(
    const Real actual,
    const Real expected) {

    const Real scale =
        std::max(
            real(1.0),
            std::max(
                Kokkos::abs(actual),
                Kokkos::abs(expected)));

    return
        Kokkos::abs(
            actual -
            expected) <=
        real(512.0) *
            std::numeric_limits<Real>::epsilon() *
            scale;
}

void initialize_boundary_fields(
    const Grid& grid,
    Field<2>& centered,
    Field<2>& positive_face,
    Field<3>& u,
    Field<3>& v) {

    const int ny =
        grid.get_local_total_points_y();

    const int nx =
        grid.get_local_total_points_x();

    const int nz =
        static_cast<int>(
            u.get_device_data().extent(0));

    const int h =
        grid.get_halo_cells();

    const int start_y =
        grid.get_local_physical_start_y();

    const int start_x =
        grid.get_local_physical_start_x();

    auto centered_data =
        centered.get_mutable_device_data();

    auto face_data =
        positive_face.get_mutable_device_data();

    auto u_data =
        u.get_mutable_device_data();

    auto v_data =
        v.get_mutable_device_data();

    const auto h1_at_u =
        grid.geometry()
            .device_view(
                HorizontalLocation::U)
            .contravariant_to_physical.a11;

    Kokkos::parallel_for(
        "InitializeRLLChannelBoundaryFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(
            const int j,
            const int i) {

            const Real global_j =
                static_cast<Real>(
                    start_y +
                    j -
                    h);

            const Real global_i =
                static_cast<Real>(
                    start_x +
                    i -
                    h);

            centered_data(j, i) =
                real(10.0) +
                real(0.5) *
                    global_j +
                real(0.01) *
                    global_i;

            face_data(j, i) =
                real(20.0) -
                real(0.25) *
                    global_j +
                real(0.02) *
                    global_i;
        });

    Kokkos::parallel_for(
        "InitializeRLLChannelWindFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {nz, ny, nx}),
        KOKKOS_LAMBDA(
            const int k,
            const int j,
            const int i) {

            const Real covariant_u =
                real(3.0) +
                real(0.2) *
                    static_cast<Real>(k) +
                real(0.01) *
                    static_cast<Real>(
                        start_x +
                        i -
                        h);

            u_data(k, j, i) =
                covariant_u /
                h1_at_u(j, i);

            v_data(k, j, i) =
                real(1.0) +
                real(0.1) *
                    static_cast<Real>(k) +
                real(0.03) *
                    static_cast<Real>(
                        start_y +
                        j -
                        h);
        });
}

void test_field_specific_boundaries(
    const Grid& grid,
    HaloExchanger& halo) {

    const int h =
        grid.get_halo_cells();

    const int ny =
        grid.get_local_total_points_y();

    const int nx =
        grid.get_local_total_points_x();

    const int nz = 3;

    Field<2> centered(
        "channel_centered",
        {ny, nx});

    Field<2> positive_face(
        "channel_positive_face",
        {ny, nx});

    Field<3> u(
        "channel_physical_u",
        {nz, ny, nx});

    Field<3> v(
        "channel_physical_v",
        {nz, ny, nx});

    initialize_boundary_fields(
        grid,
        centered,
        positive_face,
        u,
        v);

    halo.exchange_multiple_halos(
        std::vector<Field<2>*>{
            &centered,
            &positive_face
        });

    halo.exchange_multiple_halos(
        std::vector<Field<3>*>{
            &u,
            &v
        });

    HorizontalBoundaryStencils boundary(
        grid);

    boundary
        .fill_centered_q2_neumann_halos(
            centered);

    boundary
        .fill_positive_face_q2_homogeneous_dirichlet_halos(
            positive_face);

    boundary
        .fill_regular_lat_lon_free_slip_physical_wind_halos(
            u,
            v);

    Kokkos::fence();

    const auto centered_host =
        centered.get_host_data();

    const auto face_host =
        positive_face.get_host_data();

    const auto u_host =
        u.get_host_data();

    const auto v_host =
        v.get_host_data();

    const auto h1_device =
        grid.geometry()
            .device_view(
                HorizontalLocation::U)
            .contravariant_to_physical.a11
            .one_dimensional;

    const auto h1_host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            h1_device);

    const bool owns_south =
        grid.get_local_physical_start_y() == 0;

    const bool owns_north =
        grid.get_local_physical_end_y() ==
        grid.get_global_points_y() - 1;

    if (owns_south) {
        const int wall_j =
            h - 1;

        for (int i = h;
             i < nx-h;
             ++i) {

            for (int distance = 0;
                 distance < h;
                 ++distance) {

                check(
                    centered_host(
                        wall_j -
                            distance,
                        i) ==
                    centered_host(
                        h +
                            distance,
                        i),
                    "South centered field must use even reflection.");

                if (distance == 0) {
                    check(
                        face_host(
                            wall_j,
                            i) ==
                        real(0.0),
                        "South positive-face wall value must be zero.");
                } else {
                    check(
                        face_host(
                            wall_j -
                                distance,
                            i) ==
                        -face_host(
                            wall_j +
                                distance,
                            i),
                        "South positive-face exterior value must use odd reflection.");
                }
            }
        }

        for (int k = 0;
             k < nz;
             ++k) {

            for (int i = h;
                 i < nx-h;
                 ++i) {

                check(
                    v_host(
                        k,
                        wall_j,
                        i) ==
                    real(0.0),
                    "South physical normal wind must vanish at the wall.");

                for (int distance = 0;
                     distance < h;
                     ++distance) {

                    const int exterior_j =
                        wall_j -
                        distance;

                    const int interior_j =
                        h +
                        distance;

                    check(
                        close_enough(
                            h1_host(exterior_j) *
                                u_host(
                                    k,
                                    exterior_j,
                                    i),
                            h1_host(interior_j) *
                                u_host(
                                    k,
                                    interior_j,
                                    i)),
                        "South covariant tangential wind must use even reflection.");

                    if (distance > 0) {
                        check(
                            v_host(
                                k,
                                exterior_j,
                                i) ==
                            -v_host(
                                k,
                                wall_j +
                                    distance,
                                i),
                            "South normal wind must use odd reflection.");
                    }
                }
            }
        }
    }

    if (owns_north) {
        const int wall_j =
            ny -
            h -
            1;

        const int first_halo_j =
            wall_j +
            1;

        for (int i = h;
             i < nx-h;
             ++i) {

            for (int distance = 0;
                 distance < h;
                 ++distance) {

                check(
                    centered_host(
                        first_halo_j +
                            distance,
                        i) ==
                    centered_host(
                        wall_j -
                            distance,
                        i),
                    "North centered field must use even reflection.");
            }

            check(
                face_host(
                    wall_j,
                    i) ==
                real(0.0),
                "North positive-face wall value must be zero.");

            for (int distance = 1;
                 distance <= h;
                 ++distance) {

                check(
                    face_host(
                        wall_j +
                            distance,
                        i) ==
                    -face_host(
                        wall_j -
                            distance,
                        i),
                    "North positive-face exterior value must use odd reflection.");
            }
        }

        for (int k = 0;
             k < nz;
             ++k) {

            for (int i = h;
                 i < nx-h;
                 ++i) {

                check(
                    v_host(
                        k,
                        wall_j,
                        i) ==
                    real(0.0),
                    "North physical normal wind must vanish at the wall.");

                for (int distance = 0;
                     distance < h;
                     ++distance) {

                    const int exterior_j =
                        first_halo_j +
                        distance;

                    const int interior_j =
                        wall_j -
                        distance;

                    check(
                        close_enough(
                            h1_host(exterior_j) *
                                u_host(
                                    k,
                                    exterior_j,
                                    i),
                            h1_host(interior_j) *
                                u_host(
                                    k,
                                    interior_j,
                                    i)),
                        "North covariant tangential wind must use even reflection.");
                }

                for (int distance = 1;
                     distance <= h;
                     ++distance) {

                    check(
                        v_host(
                            k,
                            wall_j +
                                distance,
                            i) ==
                        -v_host(
                            k,
                            wall_j -
                                distance,
                            i),
                        "North normal wind must use odd reflection.");
                }
            }
        }
    }
}

void initialize_solver_fields(
    const Grid& grid,
    Field<2>& rhs_z,
    Field<2>& rhs_t,
    Field<2>& initial_z,
    Field<2>& initial_t) {

    const int ny =
        grid.get_local_total_points_y();

    const int nx =
        grid.get_local_total_points_x();

    const int h =
        grid.get_halo_cells();

    const int start_y =
        grid.get_local_physical_start_y();

    const int start_x =
        grid.get_local_physical_start_x();

    auto rz =
        rhs_z.get_mutable_device_data();

    auto rt =
        rhs_t.get_mutable_device_data();

    auto z =
        initial_z.get_mutable_device_data();

    auto t =
        initial_t.get_mutable_device_data();

    Kokkos::parallel_for(
        "InitializeRLLChannelSolverFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(
            const int j,
            const int i) {

            const Real global_j =
                static_cast<Real>(
                    start_y +
                    j -
                    h);

            const Real global_i =
                static_cast<Real>(
                    start_x +
                    i -
                    h);

            rz(j, i) =
                real(1.0e-6) *
                (
                    real(0.5) +
                    Kokkos::sin(
                        real(0.23) *
                        global_i)
                );

            rt(j, i) =
                real(1.0e-6) *
                (
                    real(0.25) +
                    Kokkos::cos(
                        real(0.19) *
                        global_j)
                );

            z(j, i) =
                real(2.0) +
                real(0.07) *
                    global_i -
                real(0.03) *
                    global_j;

            t(j, i) =
                real(-1.0) +
                real(0.04) *
                    global_i +
                real(0.02) *
                    global_j;
        });
}

void test_fixed_iteration_channel_rows(
    const Grid& grid,
    HaloExchanger& halo) {

    const int h =
        grid.get_halo_cells();

    const int ny =
        grid.get_local_total_points_y();

    const int nx =
        grid.get_local_total_points_x();

    Field<2> rhs_z(
        "channel_rhs_z",
        {ny, nx});

    Field<2> rhs_t(
        "channel_rhs_t",
        {ny, nx});

    Field<2> prior_z(
        "channel_prior_z",
        {ny, nx});

    Field<2> prior_t(
        "channel_prior_t",
        {ny, nx});

    Field<2> expected_z(
        "channel_expected_z",
        {ny, nx});

    Field<2> expected_t(
        "channel_expected_t",
        {ny, nx});

    Field<2> solved_z(
        "channel_solved_z",
        {ny, nx});

    Field<2> solved_t(
        "channel_solved_t",
        {ny, nx});

    initialize_solver_fields(
        grid,
        rhs_z,
        rhs_t,
        prior_z,
        prior_t);

    Kokkos::deep_copy(
        solved_z.get_mutable_device_data(),
        prior_z.get_device_data());

    Kokkos::deep_copy(
        solved_t.get_mutable_device_data(),
        prior_t.get_device_data());

    halo.exchange_multiple_halos(
        {&prior_z, &prior_t},
        1);

    HorizontalBoundaryStencils boundary(
        grid);

    boundary
        .fill_positive_face_q2_homogeneous_dirichlet_halos(
            prior_z);

    boundary
        .fill_centered_q2_neumann_halos(
            prior_t);

    Kokkos::deep_copy(
        expected_z.get_mutable_device_data(),
        prior_z.get_device_data());

    Kokkos::deep_copy(
        expected_t.get_mutable_device_data(),
        prior_t.get_device_data());

    const auto laplace =
        make_horizontal_laplace_beltrami_device_view(
            grid.geometry());

    const auto previous_z =
        prior_z.get_device_data();

    const auto previous_t =
        prior_t.get_device_data();

    const auto rz =
        rhs_z.get_device_data();

    const auto rt =
        rhs_t.get_device_data();

    auto ez =
        expected_z.get_mutable_device_data();

    auto et =
        expected_t.get_mutable_device_data();

    const Real shift =
        real(0.25);

    const bool owns_north =
        grid.get_local_physical_end_y() ==
        grid.get_global_points_y() - 1;

    const int north_wall_j =
        ny -
        h -
        1;

    Kokkos::parallel_for(
        "ReferenceRLLChannelPotentialIteration",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {h, h},
            {ny - h, nx - h}),
        KOKKOS_LAMBDA(
            const int j,
            const int i) {

            if (owns_north &&
                j == north_wall_j) {

                ez(j, i) =
                    real(0.0);
            } else {
                const Real operator_z =
                    laplace
                        .calculate_jacobian_weighted_at_z(
                            previous_z,
                            j,
                            i);

                const Real diagonal_z =
                    laplace
                        .jacobian_weighted_diagonal_at_z(
                            j,
                            i);

                ez(j, i) =
                    previous_z(j, i) +
                    (
                        operator_z -
                        laplace.divergence.z.sqrt_g(j, i) *
                            rz(j, i)
                    ) /
                    (
                        shift -
                        diagonal_z
                    );
            }

            const Real operator_t =
                laplace
                    .calculate_jacobian_weighted_at_t(
                        previous_t,
                        j,
                        i);

            const Real diagonal_t =
                laplace
                    .jacobian_weighted_diagonal_at_t(
                        j,
                        i);

            et(j, i) =
                previous_t(j, i) +
                (
                    operator_t -
                    laplace.divergence.t.sqrt_g(j, i) *
                        rt(j, i)
                ) /
                (
                    shift -
                    diagonal_t
                );
        });

    halo.exchange_multiple_halos(
        {&expected_z, &expected_t},
        1);

    boundary
        .fill_positive_face_q2_homogeneous_dirichlet_halos(
            expected_z);

    boundary
        .fill_centered_q2_neumann_halos(
            expected_t);

    HorizontalEllipticSolver solver(
        grid,
        halo);

    HorizontalEllipticSolver::Options options;

    options.iterations = 1;
    options.diagonal_shift = shift;
    options.refresh_initial_halos = true;

    solver
        .solve_regular_lat_lon_channel_at_z_and_t(
            rhs_z,
            solved_z,
            rhs_t,
            solved_t,
            options);

    Kokkos::fence();

    const auto expected_z_host =
        expected_z.get_host_data();

    const auto expected_t_host =
        expected_t.get_host_data();

    const auto solved_z_host =
        solved_z.get_host_data();

    const auto solved_t_host =
        solved_t.get_host_data();

    for (int j = h;
         j < ny - h;
         ++j) {

        for (int i = h;
             i < nx - h;
             ++i) {

            check(
                close_enough(
                    solved_z_host(j, i),
                    expected_z_host(j, i)),
                "RLL channel Z row differs from the fixed-iteration reference.");

            check(
                close_enough(
                    solved_t_host(j, i),
                    expected_t_host(j, i)),
                "RLL channel T row differs from the fixed-iteration reference.");
        }
    }

    if (grid.get_local_physical_start_y() == 0) {
        const int wall_j =
            h - 1;

        for (int i = h;
             i < nx - h;
             ++i) {

            check(
                solved_z_host(
                    wall_j,
                    i) ==
                real(0.0),
                "Solved psi must be zero on the south wall.");

            check(
                solved_t_host(
                    wall_j,
                    i) ==
                solved_t_host(
                    h,
                    i),
                "Solved chi must have zero south normal difference.");
        }
    }

    if (grid.get_local_physical_end_y() ==
        grid.get_global_points_y() - 1) {

        const int wall_j =
            ny -
            h -
            1;

        for (int i = h;
             i < nx-h;
             ++i) {

            check(
                solved_z_host(
                    wall_j,
                    i) ==
                real(0.0),
                "Solved psi must be zero on the north wall.");

            check(
                solved_t_host(
                    wall_j + 1,
                    i) ==
                solved_t_host(
                    wall_j,
                    i),
                "Solved chi must have zero north normal difference.");
        }
    }
}

int run(
    const Grid& grid,
    HaloExchanger& halo) {

    const auto& horizontal =
        grid.horizontal_specification();

    check(
        grid.geometry().kind() ==
            VVM::Core::Geometry::GeometryKind::RegularLatLon,
        "Test requires regular latitude-longitude geometry.");

    check(
        horizontal.topology.q1 ==
            VVM::Core::HorizontalEdgeTopology::Periodic,
        "Test requires periodic q1.");

    check(
        horizontal.topology.q2 ==
            VVM::Core::HorizontalEdgeTopology::Bounded,
        "Test requires bounded q2.");

    if (failures == 0) {
        test_field_specific_boundaries(
            grid,
            halo);

        test_fixed_iteration_channel_rows(
            grid,
            halo);
    }

    int global_failures = 0;

    MPI_Allreduce(
        &failures,
        &global_failures,
        1,
        MPI_INT,
        MPI_SUM,
        grid.get_comm());

    if (mpi_rank == 0) {
        if (global_failures == 0) {
            std::printf(
                "test_regular_latlon_channel_boundaries: PASS\n");
        } else {
            std::fprintf(
                stderr,
                "test_regular_latlon_channel_boundaries: %d failure(s)\n",
                global_failures);
        }
    }

    return
        global_failures;
}

} // namespace

int main(
    int argc,
    char** argv) {

    MPI_Init(
        &argc,
        &argv);

    int mpi_size = 0;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &mpi_rank);

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &mpi_size);

    if (argc != 2 ||
        (
            mpi_size != 1 &&
            mpi_size != 2 &&
            mpi_size != 4
        )) {

        fatal(
            "Provide one RLL configuration and use 1, 2, or 4 ranks.");
    }

    int result = 0;

    try {
        Kokkos::initialize(
            argc,
            argv);

        {
            ConfigurationManager config(
                argv[1]);

            Grid grid(
                config);

#if defined(ENABLE_NCCL)
            ncclUniqueId id;

            if (mpi_rank == 0) {
                nccl_check(
                    ncclGetUniqueId(
                        &id));
            }

            MPI_Bcast(
                &id,
                static_cast<int>(
                    sizeof(id)),
                MPI_BYTE,
                0,
                grid.get_comm());

            ncclComm_t communicator;

            nccl_check(
                ncclCommInitRank(
                    &communicator,
                    mpi_size,
                    id,
                    mpi_rank));

            {
                HaloExchanger halo(
                    config,
                    grid,
                    communicator,
                    Kokkos::Cuda()
                        .cuda_stream());

                result =
                    run(
                        grid,
                        halo);

                Kokkos::fence();
            }

            nccl_check(
                ncclCommDestroy(
                    communicator));
#else
            HaloExchanger halo(
                grid);

            result =
                run(
                    grid,
                    halo);
#endif
        }

        Kokkos::finalize();
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "Rank %d unexpected exception: %s\n",
            mpi_rank,
            error.what());

        result = 1;

        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }

    MPI_Finalize();

    return
        result == 0
            ? 0
            : 1;
}
