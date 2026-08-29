// HaloExchanger, the part of it both communication backends must agree on
// (src/core/HaloExchanger.hpp).
//
// The two backends are separate implementations of the same interface -- NCCL
// sends on the device stream, standard MPI posts Isend/Irecv -- so nothing but a
// test keeps them in step. This file states the contract in terms of values:
//
//   * after a default exchange (depth == -1) every halo cell of a doubly
//     periodic field holds the value of the global cell it wraps onto,
//     including the corners and including the OUTER halo layer;
//   * a depth-limited exchange (depth == 1) touches the innermost layer only
//     and leaves the outer layer as it was.
//
// The field is filled with a value that is unique per global cell, so a halo
// cell that was filled from the wrong place, or not filled at all, is caught
// exactly where it happens rather than as a slow blow-up ten minutes into a run.
//
// <cstdio> rather than <iostream>: see test_horizontal_mean.cpp.

#include "core/Grid.hpp"
#include "core/HaloExchanger.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "utils/ConfigurationManager.hpp"
#include "core/vvm_types.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdio>
#include <string>
#include <vector>

#if defined(ENABLE_NCCL)
#include <nccl.h>
#include <cuda_runtime.h>
#endif

namespace {

int g_rank = 0;
int g_failures = 0;

void report(const char* name, int bad_cells) {
    if (bad_cells != 0) ++g_failures;
    if (g_rank == 0) {
        std::fprintf(stdout, "[%s] %-52s bad cells=%d\n",
                     bad_cells == 0 ? "PASS" : "FAIL", name, bad_cells);
    }
}

// Unique per global cell, and cheap to recompute inside a kernel.
KOKKOS_INLINE_FUNCTION
VVM::Real expected_value(int k, int gj, int gi) {
    return VVM::real(k) * VVM::real(1000000.0) +
           VVM::real(gj) * VVM::real(1000.0) +
           VVM::real(gi);
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    if (argc < 2) {
        if (g_rank == 0) std::fprintf(stderr, "usage: %s <config.json>\n", argv[0]);
        MPI_Finalize();
        return 2;
    }

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
    int exit_code = 0;
    {
        VVM::Utils::ConfigurationManager config(argv[1]);
        VVM::Core::Grid grid(config, MPI_COMM_WORLD);
        VVM::Core::Parameters parameters(config, grid);

#if defined(ENABLE_NCCL)
        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        ncclUniqueId id;
        if (g_rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclComm_t nccl_comm;
        ncclCommInitRank(&nccl_comm, size, id, g_rank);
        cudaStream_t stream = Kokkos::Cuda().cuda_stream();
        VVM::Core::State state(config, parameters, grid, nccl_comm, stream);
        VVM::Core::HaloExchanger halo(config, grid, nccl_comm, stream);
        const char* backend = "NCCL";
#else
        VVM::Core::State state(config, parameters, grid);
        VVM::Core::HaloExchanger halo(grid);
        const char* backend = "MPI";
#endif

        const int h = grid.get_halo_cells();
        const int nz = grid.get_local_total_points_z();
        const int ny = grid.get_local_total_points_y();
        const int nx = grid.get_local_total_points_x();
        const int j0 = grid.get_local_physical_start_y();
        const int i0 = grid.get_local_physical_start_x();
        const int gny = grid.get_global_points_y();
        const int gnx = grid.get_global_points_x();

        // Grid ownership must be derived from the rank in cart_comm_, not the
        // original communicator rank: MPI_Cart_create may reorder them.
        int cart_rank = MPI_PROC_NULL;
        int cart_dims[2] = {0, 0};
        int cart_periods[2] = {0, 0};
        int cart_coords[2] = {0, 0};
        MPI_Comm_rank(grid.get_cart_comm(), &cart_rank);
        MPI_Cart_get(grid.get_cart_comm(), 2, cart_dims, cart_periods, cart_coords);
        MPI_Cart_coords(grid.get_cart_comm(), cart_rank, 2, cart_coords);

        const auto expected_start = [](int global_size, int process_count, int coord) {
            const int base = global_size / process_count;
            const int remainder = global_size % process_count;
            return coord * base + (coord < remainder ? coord : remainder);
        };
        int bad_ownership = 0;
        if (gny > 1 && j0 != expected_start(gny, cart_dims[0], cart_coords[0])) {
            ++bad_ownership;
        }
        if (gnx > 1 && i0 != expected_start(gnx, cart_dims[1], cart_coords[1])) {
            ++bad_ownership;
        }
        report("grid ownership follows Cartesian rank coordinates", bad_ownership);

        int mpi_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        if (g_rank == 0) {
            std::fprintf(stdout,
                         "halo-exchange test: backend=%s ranks=%d grid=%dx%d halo=%d\n",
                         backend, mpi_size, gnx, gny, h);
        }

        auto& field = state.get_field<3>("th");
        auto view = field.get_mutable_device_data();

        // Physical cells get their value, halo cells get a marker.
        const VVM::Real unfilled = VVM::real(-1.0);
        auto fill_physical_only = [&]() {
            Kokkos::parallel_for("fill",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    const bool physical = (j >= h && j < ny - h && i >= h && i < nx - h);
                    view(k, j, i) = physical
                        ? expected_value(k, j0 + (j - h), i0 + (i - h))
                        : unfilled;
                });
            Kokkos::fence();
        };

        // Count halo cells that do not hold the periodic wrap of their position.
        auto count_wrong = [&](int depth) {
            int bad = 0;
            Kokkos::parallel_reduce("check",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, 0, 0}, {nz - h, ny, nx}),
                KOKKOS_LAMBDA(const int k, const int j, const int i, int& acc) {
                    const bool inner = (j >= h - depth && j < ny - h + depth &&
                                        i >= h - depth && i < nx - h + depth);
                    if (!inner) return;  // outside the exchanged depth
                    int gj = (j0 + (j - h)) % gny; if (gj < 0) gj += gny;
                    int gi = (i0 + (i - h)) % gnx; if (gi < 0) gi += gnx;
                    if (view(k, j, i) != expected_value(k, gj, gi)) ++acc;
                }, bad);
            Kokkos::fence();
            return bad;
        };

        // --- default depth: the whole halo, corners included ------------------
        fill_physical_only();
        halo.exchange_halos(field);
        report("default exchange fills the whole halo", count_wrong(h));

        // The outer layer specifically -- this is the one a depth-of-1 fallback
        // silently leaves stale.
        {
            int bad_outer = 0;
            Kokkos::parallel_reduce("check_outer",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, 0, 0}, {nz - h, ny, nx}),
                KOKKOS_LAMBDA(const int k, const int j, const int i, int& acc) {
                    const bool outer_layer = (j < h - 1 || j >= ny - h + 1 ||
                                              i < h - 1 || i >= nx - h + 1);
                    if (!outer_layer) return;
                    int gj = (j0 + (j - h)) % gny; if (gj < 0) gj += gny;
                    int gi = (i0 + (i - h)) % gnx; if (gi < 0) gi += gnx;
                    if (view(k, j, i) != expected_value(k, gj, gi)) ++acc;
                }, bad_outer);
            Kokkos::fence();
            report("default exchange fills the outer halo layer", bad_outer);
        }

        // --- depth 1: innermost layer only ------------------------------------
        if (h > 1) {
            fill_physical_only();
            halo.exchange_halos(field, 1);
            report("depth-1 exchange fills the innermost layer", count_wrong(1));

            int touched_outer = 0;
            Kokkos::parallel_reduce("check_outer_untouched",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, 0, 0}, {nz - h, ny, nx}),
                KOKKOS_LAMBDA(const int k, const int j, const int i, int& acc) {
                    const bool outer_layer = (j < h - 1 || j >= ny - h + 1 ||
                                              i < h - 1 || i >= nx - h + 1);
                    if (outer_layer && view(k, j, i) != unfilled) ++acc;
                }, touched_outer);
            Kokkos::fence();
            report("depth-1 exchange leaves the outer layer alone", touched_outer);
        }

        // --- the name-based batched exchange (Model, Initializer, WindSolver) ---
        // Different code from the single-field path in both backends: one packed
        // buffer for all the named fields.
        {
            fill_physical_only();
            halo.exchange_multiple_halos({"th"}, state);
            report("named batched exchange fills the whole halo", count_wrong(h));
        }

        // --- 2D fields, including the batched form the wind solver relaxes with ---
        {
            auto& field2d = state.get_field<2>("utop");
            auto view2d = field2d.get_mutable_device_data();

            auto fill2d = [&]() {
                Kokkos::parallel_for("fill2d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
                    KOKKOS_LAMBDA(const int j, const int i) {
                        const bool physical = (j >= h && j < ny - h && i >= h && i < nx - h);
                        view2d(j, i) = physical
                            ? expected_value(0, j0 + (j - h), i0 + (i - h))
                            : unfilled;
                    });
                Kokkos::fence();
            };
            auto wrong2d = [&](int depth) {
                int bad = 0;
                Kokkos::parallel_reduce("check2d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
                    KOKKOS_LAMBDA(const int j, const int i, int& acc) {
                        const bool inner = (j >= h - depth && j < ny - h + depth &&
                                            i >= h - depth && i < nx - h + depth);
                        if (!inner) return;
                        int gj = (j0 + (j - h)) % gny; if (gj < 0) gj += gny;
                        int gi = (i0 + (i - h)) % gnx; if (gi < 0) gi += gnx;
                        if (view2d(j, i) != expected_value(0, gj, gi)) ++acc;
                    }, bad);
                Kokkos::fence();
                return bad;
            };

            fill2d();
            halo.exchange_halos(field2d);
            report("2D default exchange fills the whole halo", wrong2d(h));

            // What relax_2d_batched() does every iteration.
            fill2d();
            std::vector<VVM::Core::Field<2>*> batch{&field2d};
            halo.exchange_multiple_halos(batch, 1);
            report("2D batched depth-1 exchange fills the inner layer", wrong2d(1));
        }

        int global_failures = 0;
        MPI_Allreduce(&g_failures, &global_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        exit_code = global_failures == 0 ? 0 : 1;
        if (g_rank == 0) {
            std::fprintf(stdout, "%s: %d failure(s)\n",
                         global_failures == 0 ? "OK" : "FAILED", global_failures);
        }

#if defined(ENABLE_NCCL)
        ncclCommDestroy(nccl_comm);
#endif
    }
    Kokkos::finalize();
    MPI_Finalize();
    return exit_code;
}
