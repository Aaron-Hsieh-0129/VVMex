#include "dynamics/forcings/DeterministicPerturbation.hpp"

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <iostream>

using VVM::Dynamics::RandomForcingDetail::random_bits;
using VVM::Dynamics::RandomForcingDetail::signed_unit_random;

static_assert(random_bits(1129, 0, 2, 0, 0) == UINT64_C(0x2d6a29225eefb5ff));
static_assert(random_bits(1129, 7, 4, 383, 383) == UINT64_C(0xe461fe746d5a4598));
static_assert(random_bits(12345, 1, 10, 20, 30) == UINT64_C(0x96ad8774f38ee4d7));

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int result = 0;
    {
        constexpr int nz = 3;
        constexpr int ny = 384;
        constexpr int nx = 384;
        constexpr int seed = 1129;
        constexpr std::uint64_t step = 7;

        Kokkos::View<VVM::Real***> multidimensional("random_multidimensional", nz, ny, nx);
        Kokkos::View<VVM::Real***> flattened("random_flattened", nz, ny, nx);

        Kokkos::parallel_for("deterministic_random_md",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                multidimensional(k, j, i) = signed_unit_random(seed, step, k, j, i);
            });

        Kokkos::parallel_for("deterministic_random_flat",
            Kokkos::RangePolicy<>(0, nz * ny * nx),
            KOKKOS_LAMBDA(const int index) {
                const int k = index / (ny * nx);
                const int remainder = index % (ny * nx);
                const int j = remainder / nx;
                const int i = remainder % nx;
                flattened(k, j, i) = signed_unit_random(seed, step, k, j, i);
            });

        auto md_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), multidimensional);
        auto flat_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flattened);
        for (int k = 0; k < nz && result == 0; ++k) {
            for (int j = 0; j < ny && result == 0; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const VVM::Real host_value = signed_unit_random(seed, step, k, j, i);
                    if (md_host(k, j, i) != flat_host(k, j, i) ||
                        md_host(k, j, i) != host_value ||
                        md_host(k, j, i) < VVM::real(-1.0) ||
                        md_host(k, j, i) >= VVM::real(1.0)) {
                        std::cerr << "deterministic perturbation mismatch at "
                                  << k << ',' << j << ',' << i << '\n';
                        result = 1;
                        break;
                    }
                }
            }
        }
    }
    Kokkos::finalize();
    return result;
}
