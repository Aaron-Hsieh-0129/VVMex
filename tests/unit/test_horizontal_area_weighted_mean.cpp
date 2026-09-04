#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <exception>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

int rank_id = 0;
int failures = 0;

constexpr VVM::Real tolerance =
    sizeof(VVM::Real) == sizeof(double)
        ? VVM::Real(1.0e-12)
        : VVM::Real(2.0e-5);

void check_close(
    const char* name,
    const VVM::Real actual,
    const VVM::Real expected) {

    const VVM::Real scale =
        std::abs(expected) > VVM::real(1.0)
            ? std::abs(expected)
            : VVM::real(1.0);

    const bool passed =
        std::abs(actual - expected) <=
        tolerance * scale;

    if (!passed) {
        ++failures;
    }

    if (rank_id == 0) {
        std::fprintf(
            stdout,
            "[%s] %-52s actual=%.17g expected=%.17g\n",
            passed ? "PASS" : "FAIL",
            name,
            static_cast<double>(actual),
            static_cast<double>(expected));
    }
}

void check_true(
    const char* name,
    const bool condition) {

    if (!condition) {
        ++failures;
    }

    if (rank_id == 0) {
        std::fprintf(
            stdout,
            "[%s] %s\n",
            condition ? "PASS" : "FAIL",
            name);
    }
}

VVM::Real read_scalar(
    const VVM::Core::ScalarView& value) {

    VVM::Real host_value = VVM::real(0.0);
    Kokkos::deep_copy(host_value, value);

    return host_value;
}

void fill_affine_2d(
    const VVM::Core::Grid& grid,
    VVM::Core::Field<2>& field,
    const VVM::Real halo_value) {

    auto values =
        field.get_mutable_device_data();

    const int h =
        grid.get_halo_cells();

    const int ny =
        grid.get_local_total_points_y();

    const int nx =
        grid.get_local_total_points_x();

    const int global_start_j =
        grid.get_local_physical_start_y();

    const int global_start_i =
        grid.get_local_physical_start_x();

    Kokkos::parallel_for(
        "FillAreaMeanAffine2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(
            const int j,
            const int i) {

            const bool physical =
                j >= h &&
                j < ny - h &&
                i >= h &&
                i < nx - h;

            if (physical) {
                const int global_j =
                    global_start_j + j - h;

                const int global_i =
                    global_start_i + i - h;

                values(j, i) =
                    VVM::real(3.0) +
                    VVM::real(2.0) *
                        VVM::real(global_i) -
                    VVM::real(0.5) *
                        VVM::real(global_j);
            }
            else {
                values(j, i) =
                    halo_value;
            }
        });

    Kokkos::fence();
}

void fill_constant_2d(
    const VVM::Core::Grid& grid,
    VVM::Core::Field<2>& field,
    const VVM::Real physical_value,
    const VVM::Real halo_value) {

    auto values =
        field.get_mutable_device_data();

    const int h =
        grid.get_halo_cells();

    const int ny =
        grid.get_local_total_points_y();

    const int nx =
        grid.get_local_total_points_x();

    Kokkos::parallel_for(
        "FillAreaMeanConstant2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(
            const int j,
            const int i) {

            const bool physical =
                j >= h &&
                j < ny - h &&
                i >= h &&
                i < nx - h;

            values(j, i) =
                physical
                    ? physical_value
                    : halo_value;
        });

    Kokkos::fence();
}

void fill_3d(
    const VVM::Core::Grid& grid,
    VVM::Core::Field<3>& field,
    const VVM::Real halo_value) {

    auto values =
        field.get_mutable_device_data();

    const int h =
        grid.get_halo_cells();

    const int nz =
        grid.get_local_total_points_z();

    const int ny =
        grid.get_local_total_points_y();

    const int nx =
        grid.get_local_total_points_x();

    const int global_start_j =
        grid.get_local_physical_start_y();

    const int global_start_i =
        grid.get_local_physical_start_x();

    const int global_nx =
        grid.get_global_points_x();

    Kokkos::parallel_for(
        "FillAreaMean3D",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {nz, ny, nx}),
        KOKKOS_LAMBDA(
            const int k,
            const int j,
            const int i) {

            const bool physical =
                k >= h &&
                k < nz - h &&
                j >= h &&
                j < ny - h &&
                i >= h &&
                i < nx - h;

            if (physical) {
                const int global_j =
                    global_start_j + j - h;

                const int global_i =
                    global_start_i + i - h;

                const VVM::Real horizontal_index =
                    VVM::real(global_j) *
                        VVM::real(global_nx) +
                    VVM::real(global_i);

                values(k, j, i) =
                    VVM::real(100.0) *
                        VVM::real(k) +
                    horizontal_index;
            }
            else {
                values(k, j, i) =
                    halo_value;
            }
        });

    Kokkos::fence();
}

} // namespace

int main(
    int argc,
    char** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank_id);

    if (argc < 2) {
        if (rank_id == 0) {
            std::fprintf(
                stderr,
                "usage: %s <config.json>\n",
                argv[0]);
        }

        MPI_Finalize();
        return 2;
    }

    Kokkos::initialize(
        Kokkos::InitializationSettings()
            .set_device_id(0));

    int exit_code = 0;

    {
        VVM::Utils::ConfigurationManager
            config(argv[1]);

        VVM::Core::Grid
            grid(config, MPI_COMM_WORLD);

        VVM::Core::Parameters
            parameters(config, grid);

#if defined(ENABLE_NCCL)
        int mpi_size = 1;

        MPI_Comm_size(
            MPI_COMM_WORLD,
            &mpi_size);

        ncclUniqueId id;

        if (rank_id == 0) {
            ncclGetUniqueId(&id);
        }

        MPI_Bcast(
            &id,
            sizeof(id),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD);

        ncclComm_t nccl_comm;

        ncclCommInitRank(
            &nccl_comm,
            mpi_size,
            id,
            rank_id);

        cudaStream_t stream =
            Kokkos::Cuda().cuda_stream();

        VVM::Core::State state(
            config,
            parameters,
            grid,
            nccl_comm,
            stream);
#else
        VVM::Core::State state(
            config,
            parameters,
            grid);
#endif

        VVM::Core::ScalarView
            point_mean("point_mean");

        VVM::Core::ScalarView
            area_mean("area_mean");

        const int global_nx =
            grid.get_global_points_x();

        const int global_ny =
            grid.get_global_points_y();

        const VVM::Real mean_i =
            VVM::real(global_nx - 1) /
            VVM::real(2.0);

        const VVM::Real mean_j =
            VVM::real(global_ny - 1) /
            VVM::real(2.0);

        const VVM::Real affine_expected =
            VVM::real(3.0) +
            VVM::real(2.0) * mean_i -
            VVM::real(0.5) * mean_j;

        auto& centered_2d =
            state.get_field<2>("Tg");

        fill_affine_2d(
            grid,
            centered_2d,
            VVM::real(1.0e9));

        state.calculate_horizontal_mean(
            centered_2d,
            point_mean);

        state.calculate_horizontal_area_weighted_mean(
            centered_2d,
            area_mean);

        check_close(
            "Cartesian centered area mean matches analytic mean",
            read_scalar(area_mean),
            affine_expected);

        check_close(
            "Cartesian centered area mean matches point mean",
            read_scalar(area_mean),
            read_scalar(point_mean));

        fill_affine_2d(
            grid,
            centered_2d,
            VVM::real(-7.0e9));

        state.calculate_horizontal_area_weighted_mean(
            centered_2d,
            area_mean);

        check_close(
            "area mean excludes horizontal halos",
            read_scalar(area_mean),
            affine_expected);

        auto& staggered_2d =
            state.get_field<2>("utop");

        const VVM::Real constant =
            VVM::real(7.25);

        fill_constant_2d(
            grid,
            staggered_2d,
            constant,
            VVM::real(9.0e8));

        state.calculate_horizontal_area_weighted_mean(
            staggered_2d,
            area_mean);

        check_close(
            "field metadata selects U-location geometry",
            read_scalar(area_mean),
            constant);

        auto& field_3d =
            state.get_field<3>("th");

        fill_3d(
            grid,
            field_3d,
            VVM::real(1.0e9));

        const VVM::Real point_count =
            VVM::real(global_nx) *
            VVM::real(global_ny);

        const VVM::Real horizontal_index_mean =
            (point_count - VVM::real(1.0)) /
            VVM::real(2.0);

        const int h =
            grid.get_halo_cells();

        const int nz =
            grid.get_local_total_points_z();

        const int k_mid =
            h + (nz - 2 * h) / 2;

        const int k_top =
            nz - h - 1;

        state.calculate_horizontal_area_weighted_mean(
            field_3d,
            area_mean,
            k_mid);

        check_close(
            "3D area mean at explicit level",
            read_scalar(area_mean),
            VVM::real(100.0) *
                VVM::real(k_mid) +
                horizontal_index_mean);

        state.calculate_horizontal_area_weighted_mean(
            field_3d,
            area_mean);

        check_close(
            "3D default level is highest physical level",
            read_scalar(area_mean),
            VVM::real(100.0) *
                VVM::real(k_top) +
                horizontal_index_mean);

        bool lower_halo_threw = false;

        try {
            state.calculate_horizontal_area_weighted_mean(
                field_3d,
                area_mean,
                h - 1);
        }
        catch (const std::exception&) {
            lower_halo_threw = true;
        }

        check_true(
            "lower-halo k_level throws",
            lower_halo_threw);

        bool upper_halo_threw = false;

        try {
            state.calculate_horizontal_area_weighted_mean(
                field_3d,
                area_mean,
                nz - h);
        }
        catch (const std::exception&) {
            upper_halo_threw = true;
        }

        check_true(
            "upper-halo k_level throws",
            upper_halo_threw);

        VVM::Core::Field<2> unspecified(
            "unspecified_staggering",
            std::array<int, 2>{
                grid.get_local_total_points_y(),
                grid.get_local_total_points_x()});

        bool unspecified_threw = false;

        try {
            state.calculate_horizontal_area_weighted_mean(
                unspecified,
                area_mean);
        }
        catch (const std::exception&) {
            unspecified_threw = true;
        }

        check_true(
            "unspecified horizontal staggering throws",
            unspecified_threw);

        int global_failures = 0;

        MPI_Allreduce(
            &failures,
            &global_failures,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD);

        exit_code =
            global_failures == 0
                ? 0
                : 1;

        if (rank_id == 0) {
            std::fprintf(
                stdout,
                "%s: %d failure(s)\n",
                global_failures == 0
                    ? "OK"
                    : "FAILED",
                global_failures);
        }

#if defined(ENABLE_NCCL)
        ncclCommDestroy(nccl_comm);
#endif
    }

    Kokkos::finalize();
    MPI_Finalize();

    return exit_code;
}
