// State::calculate_horizontal_mean(), the one API both communication backends
// share (src/core/State.hpp).
//
// The same source builds against either backend: with ENABLE_NCCL=ON the global
// sum is an NCCL all-reduce on the device, with ENABLE_NCCL=OFF it is a host
// MPI_Allreduce. The expected values below are backend-independent, so running
// this binary from both build trees is also the cross-backend comparison -- the
// printed means are exact decimal digits of the same numbers.
//
// What is pinned here:
//   * 2D mean on one rank, and on several ranks (the local sums must be summed
//     globally and normalized by gnx*gny, not by the local point count);
//   * 3D mean at an explicit physical level;
//   * 3D mean with the default k_level == -1, which must resolve to nz - h - 1;
//   * halo cells are excluded -- they are filled with junk that would move the
//     mean by orders of magnitude, and rewriting the junk must not change it;
//   * an explicit level inside the halo throws instead of quietly yielding 0.
//
// Field values are the global linear index of the cell, so the exact mean is
// known in closed form and is independent of how the domain is decomposed.
//
// Reporting goes through <cstdio> rather than <iostream>: the NVHPC build
// compiles against the system GCC 13 headers while GCC 11's libstdc++ loads at
// run time, and that mismatch faults inside the ostream sentry.

#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "utils/ConfigurationManager.hpp"
#include "core/vvm_types.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(ENABLE_NCCL)
#include <nccl.h>
#include <cuda_runtime.h>
#endif

namespace {

int g_rank = 0;
int g_failures = 0;

// Relative tolerance: the reductions are exact in double, but the same test is
// meaningful in a single-precision build, where a 32^2 sum of ~10^3 values is
// already at the edge of the mantissa.
constexpr VVM::Real kTolerance =
    sizeof(VVM::Real) == sizeof(double) ? VVM::Real(1e-12) : VVM::Real(1e-5);

void report(const char* name, VVM::Real got, VVM::Real want) {
    const VVM::Real scale = std::fabs(want) > VVM::Real(1) ? std::fabs(want) : VVM::Real(1);
    const bool ok = std::fabs(got - want) <= kTolerance * scale;
    if (!ok) ++g_failures;
    if (g_rank == 0) {
        std::fprintf(stdout, "[%s] %-46s got=%.17g want=%.17g\n",
                     ok ? "PASS" : "FAIL", name,
                     static_cast<double>(got), static_cast<double>(want));
    }
}

void report_bool(const char* name, bool ok) {
    if (!ok) ++g_failures;
    if (g_rank == 0) {
        std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", name);
    }
}

VVM::Real read_scalar(const VVM::Core::ScalarView& v) {
    VVM::Real host_value = VVM::real(0.0);
    Kokkos::deep_copy(host_value, v);
    return host_value;
}

// Physical cells hold their global horizontal index; halo cells hold junk.
void fill_2d(const VVM::Core::Grid& grid, VVM::Core::Field<2>& field, VVM::Real halo_junk) {
    auto view = field.get_mutable_device_data();
    const int h = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int j0 = grid.get_local_physical_start_y();
    const int i0 = grid.get_local_physical_start_x();
    const int gnx = grid.get_global_points_x();

    Kokkos::parallel_for("fill_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const bool physical = (j >= h && j < ny - h && i >= h && i < nx - h);
            if (physical) {
                const int gj = j0 + (j - h);
                const int gi = i0 + (i - h);
                view(j, i) = VVM::real(gj) * VVM::real(gnx) + VVM::real(gi);
            } else {
                view(j, i) = halo_junk;
            }
        });
    Kokkos::fence();
}

// As above, with the level index folded in so that each level has its own mean.
void fill_3d(const VVM::Core::Grid& grid, VVM::Core::Field<3>& field, VVM::Real halo_junk) {
    auto view = field.get_mutable_device_data();
    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int j0 = grid.get_local_physical_start_y();
    const int i0 = grid.get_local_physical_start_x();
    const int gnx = grid.get_global_points_x();

    Kokkos::parallel_for("fill_3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const bool physical = (k >= h && k < nz - h &&
                                   j >= h && j < ny - h &&
                                   i >= h && i < nx - h);
            if (physical) {
                const int gj = j0 + (j - h);
                const int gi = i0 + (i - h);
                view(k, j, i) = VVM::real(k) * VVM::real(100.0) +
                                VVM::real(gj) * VVM::real(gnx) + VVM::real(gi);
            } else {
                view(k, j, i) = halo_junk;
            }
        });
    Kokkos::fence();
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    if (argc < 2) {
        if (g_rank == 0) {
            std::fprintf(stderr, "usage: %s <config.json>\n", argv[0]);
        }
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
        const char* backend = "NCCL";
#else
        VVM::Core::State state(config, parameters, grid);
        const char* backend = "MPI";
#endif

        const int h = grid.get_halo_cells();
        const int nz = grid.get_local_total_points_z();
        const int gnx = grid.get_global_points_x();
        const int gny = grid.get_global_points_y();
        const VVM::Real npoints = VVM::real(gnx) * VVM::real(gny);
        // Values 0 .. N-1, each exactly once over the global domain.
        const VVM::Real index_mean = (npoints - VVM::real(1.0)) / VVM::real(2.0);

        int mpi_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        if (g_rank == 0) {
            std::fprintf(stdout,
                         "horizontal-mean test: backend=%s ranks=%d grid=%dx%d halo=%d nz_local=%d\n",
                         backend, mpi_size, gnx, gny, h, nz);
        }

        VVM::Core::ScalarView mean("mean");

        // --- 2D, on however many ranks this run was launched with -----------
        auto& field_2d = state.get_field<2>("utop");
        fill_2d(grid, field_2d, VVM::real(1.0e9));
        state.calculate_horizontal_mean(field_2d, mean);
        report("2D horizontal mean", read_scalar(mean), index_mean);

        // Halo exclusion: rewriting the halo must not move the mean.
        fill_2d(grid, field_2d, VVM::real(-7.0e9));
        state.calculate_horizontal_mean(field_2d, mean);
        report("2D mean is unchanged by halo contents", read_scalar(mean), index_mean);

        // --- 3D, explicit physical level ------------------------------------
        auto& field_3d = state.get_field<3>("th");
        fill_3d(grid, field_3d, VVM::real(1.0e9));

        const int k_mid = h + (nz - 2 * h) / 2;
        state.calculate_horizontal_mean(field_3d, mean, k_mid);
        report("3D mean at an explicit level", read_scalar(mean),
               VVM::real(k_mid) * VVM::real(100.0) + index_mean);

        // Lowest and highest physical levels are both valid.
        state.calculate_horizontal_mean(field_3d, mean, h);
        report("3D mean at the lowest physical level", read_scalar(mean),
               VVM::real(h) * VVM::real(100.0) + index_mean);

        // --- 3D, default level ------------------------------------------------
        const int k_top = nz - h - 1;
        state.calculate_horizontal_mean(field_3d, mean);
        const VVM::Real default_mean = read_scalar(mean);
        report("3D mean with default k_level", default_mean,
               VVM::real(k_top) * VVM::real(100.0) + index_mean);

        state.calculate_horizontal_mean(field_3d, mean, k_top);
        report("default k_level equals nz - h - 1", default_mean, read_scalar(mean));

        // Halo exclusion in 3D, including the vertical halo.
        fill_3d(grid, field_3d, VVM::real(-7.0e9));
        state.calculate_horizontal_mean(field_3d, mean);
        report("3D mean is unchanged by halo contents", read_scalar(mean), default_mean);

        // --- reduction order, made visible -----------------------------------
        // Integer-valued cells sum exactly, so they cannot show how the global
        // sum was ordered. These values cannot be summed exactly, so the printed
        // digits depend on the order the per-rank partial sums are combined --
        // NCCL's ring versus whatever MPI_Allreduce picks. Compared across
        // backends by eye (and by diffing this line), not asserted bit-for-bit:
        // neither library specifies its order.
        {
            auto view = field_2d.get_mutable_device_data();
            const int nx_t = grid.get_local_total_points_x();
            const int ny_t = grid.get_local_total_points_y();
            const int i0 = grid.get_local_physical_start_x();
            const int j0 = grid.get_local_physical_start_y();
            Kokkos::parallel_for("fill_fractional",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny_t, nx_t}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    const bool physical = (j >= h && j < ny_t - h && i >= h && i < nx_t - h);
                    if (physical) {
                        const int gj = j0 + (j - h);
                        const int gi = i0 + (i - h);
                        view(j, i) = VVM::real(1.0) /
                                     (VVM::real(gj) * VVM::real(gnx) + VVM::real(gi) + VVM::real(1.0));
                    } else {
                        view(j, i) = VVM::real(0.0);
                    }
                });
            Kokkos::fence();

            state.calculate_horizontal_mean(field_2d, mean);
            const VVM::Real fractional = read_scalar(mean);
            if (g_rank == 0) {
                std::fprintf(stdout, "[INFO] inexact-sum mean = %.17g  (%s, %d ranks)\n",
                             static_cast<double>(fractional), backend, mpi_size);
            }

            // Same idea, but with the summands spanning many orders of magnitude,
            // which is where two different reduction orders diverge soonest.
            Kokkos::parallel_for("fill_wide_magnitude",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny_t, nx_t}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    const bool physical = (j >= h && j < ny_t - h && i >= h && i < nx_t - h);
                    if (physical) {
                        const int gj = j0 + (j - h);
                        const int gi = i0 + (i - h);
                        const VVM::Real n = VVM::real(gj) * VVM::real(gnx) + VVM::real(gi);
                        // Alternating sign and an exponent sweep: catastrophic
                        // cancellation, so the ordering shows in the top bits.
                        const VVM::Real sign = ((gi + gj) % 2 == 0) ? VVM::real(1.0) : VVM::real(-1.0);
                        view(j, i) = sign * Kokkos::exp(VVM::real(1.0e-5) * n);
                    } else {
                        view(j, i) = VVM::real(0.0);
                    }
                });
            Kokkos::fence();
            state.calculate_horizontal_mean(field_2d, mean);
            if (g_rank == 0) {
                std::fprintf(stdout, "[INFO] wide-magnitude mean = %.17g  (%s, %d ranks)\n",
                             static_cast<double>(read_scalar(mean)), backend, mpi_size);
            }

            // Structured patterns can agree by symmetry. Pseudo-random values are
            // the honest test of whether the two libraries combine the per-rank
            // partial sums in the same order.
            Kokkos::parallel_for("fill_pseudo_random",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny_t, nx_t}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    const bool physical = (j >= h && j < ny_t - h && i >= h && i < nx_t - h);
                    if (physical) {
                        const int gj = j0 + (j - h);
                        const int gi = i0 + (i - h);
                        uint64_t x = static_cast<uint64_t>(gj) * 2654435761u +
                                     static_cast<uint64_t>(gi) * 40503u + 1u;
                        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
                        // Uniform in [-1, 1), no structure for an ordering to exploit.
                        view(j, i) = VVM::real(2.0) *
                                     (VVM::real(x >> 11) / VVM::real(9007199254740992.0)) -
                                     VVM::real(1.0);
                    } else {
                        view(j, i) = VVM::real(0.0);
                    }
                });
            Kokkos::fence();
            state.calculate_horizontal_mean(field_2d, mean);
            if (g_rank == 0) {
                std::fprintf(stdout, "[INFO] pseudo-random mean = %.17g  (%s, %d ranks)\n",
                             static_cast<double>(read_scalar(mean)), backend, mpi_size);
            }

            // Same values through the 3D path: a different view rank, so a
            // different reduction kernel instantiation.
            auto view3 = field_3d.get_mutable_device_data();
            const int nz_t = grid.get_local_total_points_z();
            Kokkos::parallel_for("fill3_pseudo_random",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz_t, ny_t, nx_t}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    const bool physical = (k >= h && k < nz_t - h &&
                                           j >= h && j < ny_t - h && i >= h && i < nx_t - h);
                    if (physical) {
                        const int gj = j0 + (j - h);
                        const int gi = i0 + (i - h);
                        uint64_t x = static_cast<uint64_t>(gj) * 2654435761u +
                                     static_cast<uint64_t>(gi) * 40503u + 1u;
                        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
                        view3(k, j, i) = VVM::real(2.0) *
                                         (VVM::real(x >> 11) / VVM::real(9007199254740992.0)) -
                                         VVM::real(1.0);
                    } else {
                        view3(k, j, i) = VVM::real(0.0);
                    }
                });
            Kokkos::fence();
            state.calculate_horizontal_mean(field_3d, mean);
            if (g_rank == 0) {
                std::fprintf(stdout, "[INFO] pseudo-random 3D mean = %.17g  (%s, %d ranks)\n",
                             static_cast<double>(read_scalar(mean)), backend, mpi_size);
            }

            // A constant field, which is what the initialiser actually averages
            // (v starts as the V(k) profile). The value is not representable, so
            // summing N copies of it accumulates rounding, and the total depends
            // on the order the local reduction combines them.
            const VVM::Real constant = VVM::real(-2.8652);
            Kokkos::parallel_for("fill3_constant",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz_t, ny_t, nx_t}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    const bool physical = (k >= h && k < nz_t - h &&
                                           j >= h && j < ny_t - h && i >= h && i < nx_t - h);
                    view3(k, j, i) = physical ? constant : VVM::real(0.0);
                });
            Kokkos::fence();
            state.calculate_horizontal_mean(field_3d, mean);
            if (g_rank == 0) {
                std::fprintf(stdout, "[INFO] constant-field 3D mean = %.17g  (%s, %d ranks)\n",
                             static_cast<double>(read_scalar(mean)), backend, mpi_size);
            }
            // Harmonic-series sum over the domain; only the last bits are at stake.
            VVM::Real harmonic = VVM::real(0.0);
            for (int n = static_cast<int>(npoints); n >= 1; --n) {
                harmonic += VVM::real(1.0) / VVM::real(n);
            }
            report("inexact-sum mean matches a serial sum",
                   fractional, harmonic / npoints);
        }

        // --- invalid explicit levels fail loudly ------------------------------
        bool threw_below = false;
        try {
            state.calculate_horizontal_mean(field_3d, mean, h - 1);
        } catch (const std::exception&) {
            threw_below = true;
        }
        report_bool("k_level in the lower halo throws", threw_below);

        bool threw_above = false;
        try {
            state.calculate_horizontal_mean(field_3d, mean, nz - h);
        } catch (const std::exception&) {
            threw_above = true;
        }
        report_bool("k_level in the upper halo throws", threw_above);

        // Every rank must agree, or a multi-rank failure could hide on rank 0.
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
