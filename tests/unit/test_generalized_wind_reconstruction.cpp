#include "GeneralizedWindRecoveryTestGeometry.hpp"
#include "dynamics/operators/GeneralizedWindRecovery.hpp"
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <limits>

namespace {
using namespace WindRecoveryTest;
namespace O = VVM::Dynamics::Operators;

template <class Layout>
void
run(Real shear, bool varying) {
    Geometry g(shear, varying);
    const int nx = g.nx(), ny = g.ny(), h = g.h();
    const Real dq1 = g.dq1(), dq2 = g.dq2(), sentinel = real(-731);
    Kokkos::View<Real**, Layout> ps("psi", ny, nx), ch("chi", ny, nx);
    Kokkos::View<Real***, Layout> out("reconstructed_wind", 4, ny, nx);
    const auto op = O::make_generalized_horizontal_wind_reconstruction_device_view(g);
    const Real tol = real(512) * std::numeric_limits<Real>::epsilon();
    for (int mode = 0; mode < 4; ++mode) {
        for (int version : {1, 3}) {
            auto ph = Kokkos::create_mirror_view(ps), chh = Kokkos::create_mirror_view(ch);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    ph(j, i) = psi(g.p(3, i), g.q(3, j), version, mode);
                    chh(j, i) = chi(g.p(0, i), g.q(0, j), version, mode);
                }
            }
            Kokkos::deep_copy(ps, ph);
            Kokkos::deep_copy(ch, chh);
            Kokkos::deep_copy(out, sentinel);
            // Fill one halo around the physical domain to check div(rot) and
            // curl(grad) without fetching uninitialized reconstruction output.
            Kokkos::parallel_for("GeneralizedWindReconstructionTest",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {ny - 1, nx - 1}),
                KOKKOS_LAMBDA(int j, int i) {
                    out(0, j, i) = op.calculate_contravariant_q1_at_u(ps, ch, j, i);
                    out(1, j, i) = op.calculate_contravariant_q2_at_v(ps, ch, j, i);
                    out(2, j, i) = op.calculate_covariant_q1_at_u(ps, ch, j, i);
                    out(3, j, i) = op.calculate_covariant_q2_at_v(ps, ch, j, i);
                });
            const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (j == 0 || j == ny - 1 || i == 0 || i == nx - 1) {
                        for (int n = 0; n < 4; ++n) {
                            require(a(n, j, i) == sentinel, "output halo overwritten");
                        }
                        continue;
                    }
                    const auto e = reference(g, j, i, version, mode);
                    for (int n = 0; n < 4; ++n) {
                        require(std::isfinite(a(n, j, i)) &&
                                    std::abs(a(n, j, i) - e[n]) <=
                                        tol * std::max(real(1), std::abs(e[n])),
                            "native-metric reconstruction differs from the independent CVVM "
                            "oracle");
                    }
                }
            }
            for (int j = h; j < ny - h; ++j) {
                for (int i = h; i < nx - h; ++i) {
                    if (mode == 1) {
                        const Real d1 = (g.metric(1, j, i).J * a(0, j, i) -
                                            g.metric(1, j, i - 1).J * a(0, j, i - 1)) /
                                        dq1;
                        const Real d2 = (g.metric(2, j, i).J * a(1, j, i) -
                                            g.metric(2, j - 1, i).J * a(1, j - 1, i)) /
                                        dq2;
                        require(std::abs(d1 + d2) <=
                                    tol * std::max({real(1), std::abs(d1), std::abs(d2)}),
                            "Jacobian-weighted divergence of rotational wind is not zero");
                    }
                    if (mode == 2) {
                        const Real d1 = (a(3, j, i + 1) - a(3, j, i)) / dq1;
                        const Real d2 = (a(2, j + 1, i) - a(2, j, i)) / dq2;
                        require(std::abs(d1 - d2) <=
                                    tol * std::max({real(1), std::abs(d1), std::abs(d2)}),
                            "curl of the covariant chi gradient is not zero");
                    }
                }
            }
        }
    }
    std::cout << "PASS: reconstruction, shear=" << shear << ", varying=" << varying << '\n';
}
} // namespace

int
main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int result = 0;
    try {
        for (Real s : {real(.8), real(-.7)}) {
            for (bool varying : {false, true}) {
                run<Kokkos::LayoutLeft>(s, varying);
                run<Kokkos::LayoutRight>(s, varying);
            }
        }
        Geometry no_halo(real(.8), true, 0);
        bool rejected = false;
        try {
            (void)O::make_generalized_horizontal_wind_reconstruction_device_view(no_halo);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "missing halo was not rejected");
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        result = 1;
    }
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
