#include "GeneralizedWindRecoveryTestGeometry.hpp"
#include "dynamics/operators/GeneralizedHorizontalElliptic.hpp"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <utility>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {
using VVM::Real;
using VVM::real;
using WindRecoveryTest::Geometry;
using WindRecoveryTest::require;
using Operator = VVM::Dynamics::Operators::GeneralizedHorizontalEllipticDeviceView;
using VVM::Dynamics::Operators::make_generalized_horizontal_elliptic_device_view;

// Independent host reference: inverse/covariant metric, explicit native-face
// derivatives and four-point interpolation. No production reconstruction or
// elliptic method is used in this oracle.
template <class Scalar>
Real
reference_t(const Geometry& g, const Scalar& f, int j, int i) {
    const auto d1 = [&](int y, int x) {
        return (f(y, x + 1) - f(y, x)) / g.dq1();
    };
    const auto d2 = [&](int y, int x) {
        return (f(y + 1, x) - f(y, x)) / g.dq2();
    };
    const auto flux1 = [&](int y, int x) {
        Real cross = real(0);
        for (int dy : {-1, 0}) {
            for (int dx : {0, 1}) {
                cross += g.metric(2, y + dy, x + dx).c12 * d2(y + dy, x + dx);
            }
        }
        const auto m = g.metric(1, y, x);
        return m.J * (m.c11 * d1(y, x) + real(.25) * cross);
    };
    const auto flux2 = [&](int y, int x) {
        Real cross = real(0);
        for (int dy : {0, 1}) {
            for (int dx : {-1, 0}) {
                cross += g.metric(1, y + dy, x + dx).c12 * d1(y + dy, x + dx);
            }
        }
        const auto m = g.metric(2, y, x);
        return m.J * (m.c22 * d2(y, x) + real(.25) * cross);
    };
    return (flux1(j, i) - flux1(j, i - 1)) / g.dq1() + (flux2(j, i) - flux2(j - 1, i)) / g.dq2();
}

template <class Scalar>
Real
reference_z(const Geometry& g, const Scalar& f, int j, int i) {
    const auto r1 = [&](int y, int x) {
        return -(f(y, x) - f(y - 1, x)) / (g.metric(1, y, x).J * g.dq2());
    };
    const auto r2 = [&](int y, int x) {
        return (f(y, x) - f(y, x - 1)) / (g.metric(2, y, x).J * g.dq1());
    };
    const auto cov1 = [&](int y, int x) {
        Real cross = real(0);
        for (int dy : {-1, 0}) {
            for (int dx : {0, 1}) {
                cross += g.metric(2, y + dy, x + dx).g12 * r2(y + dy, x + dx);
            }
        }
        return g.metric(1, y, x).g11 * r1(y, x) + real(.25) * cross;
    };
    const auto cov2 = [&](int y, int x) {
        Real cross = real(0);
        for (int dy : {0, 1}) {
            for (int dx : {-1, 0}) {
                cross += g.metric(1, y + dy, x + dx).g12 * r1(y + dy, x + dx);
            }
        }
        return g.metric(2, y, x).g22 * r2(y, x) + real(.25) * cross;
    };
    return (cov2(j, i + 1) - cov2(j, i)) / g.dq1() - (cov1(j + 1, i) - cov1(j, i)) / g.dq2();
}

Real
potential(const Geometry& g, int loc, int j, int i, int version) {
    const Real p = g.p(loc, i), q = g.q(loc, j);
    return real(.3) * version +
           (real(1) + real(.125) * version) *
               (p * p + real(.5) * p * q + real(.75) * q * q + real(.125) * p - real(.375) * q);
}

template <class View>
std::vector<Real>
snapshot(const View& v) {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    std::vector<Real> a;
    for (std::size_t j = 0; j < h.extent(0); ++j) {
        for (std::size_t i = 0; i < h.extent(1); ++i) {
            a.push_back(h(j, i));
        }
    }
    return a;
}

template <class Layout>
struct Evaluate {
    Kokkos::View<const Operator> op;
    Kokkos::View<Real**, Layout> t, z;
    Kokkos::View<Real***, Layout> out;
    bool active = true;
    KOKKOS_INLINE_FUNCTION void
    operator()(int j, int i) const {
        if (!active) {
            return;
        }
        out(0, j, i) = op().calculate_jacobian_weighted_at_t(t, j, i);
        out(1, j, i) = op().calculate_jacobian_weighted_at_z(z, j, i);
        out(2, j, i) = op().jacobian_weighted_diagonal_at_t(j, i);
        out(3, j, i) = op().jacobian_weighted_diagonal_at_z(j, i);
    }
};

template <class Layout>
struct Relax {
    Kokkos::View<const Operator> op;
    Kokkos::View<Real**, Layout> old_t, old_z, new_t, new_z, rhs_t, rhs_z;
    Real shift = real(2);
    KOKKOS_INLINE_FUNCTION void
    operator()(int j, int i) const {
        new_t(j, i) = op().relaxed_at_t(old_t, rhs_t(j, i), shift, j, i);
        new_z(j, i) = op().relaxed_at_z(old_z, rhs_z(j, i), shift, j, i);
    }
};

#if defined(KOKKOS_ENABLE_CUDA)
void
cuda_check(cudaError_t code, const char* msg) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(code));
    }
}
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStream_t stream = Kokkos::Cuda().cuda_stream();
    ~Graph() {
        if (exec) {
            cudaGraphExecDestroy(exec);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
    template <class Launch>
    void
    capture(const Launch& launch) {
        Kokkos::fence();
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        try {
            launch();
        }
        catch (...) {
            cudaStreamEndCapture(stream, &graph);
            throw;
        }
        cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    }
    void
    run() {
        cuda_check(cudaGraphLaunch(exec, stream), "launch");
        cuda_check(cudaStreamSynchronize(stream), "synchronize");
    }
};
#endif

template <class Layout>
void
test(Real shear, bool varying, const char* layout_name) {
    using Plane = Kokkos::View<Real**, Layout>;
    Geometry g(shear, varying);
    const int h = g.h(), ny = g.ny(), nx = g.nx();
    const auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    const Real sentinel = real(-12345);
    const Real tol = real(1024) * std::numeric_limits<Real>::epsilon();
    Kokkos::View<Operator> operation("elliptic_operation");
    auto host_op = Kokkos::create_mirror_view(operation);
    host_op() = make_generalized_horizontal_elliptic_device_view(g);
    Kokkos::deep_copy(operation, host_op);
    Plane t("chi", ny, nx), z("psi", ny, nx);
    Kokkos::View<Real***, Layout> output("elliptic_output", 4, ny, nx);
    Evaluate<Layout> evaluator{operation, t, z, output};
    const auto launch = [&] {
        Kokkos::parallel_for("EvaluateGeneralizedElliptic", policy, evaluator);
    };

    // The actual functor launch type is warmed before graph capture.
    evaluator.active = false;
    launch();
    Kokkos::fence();
    evaluator.active = true;
#if defined(KOKKOS_ENABLE_CUDA)
    Graph graph;
    graph.capture(launch);
#endif
    for (int version : {0, 1, 3}) {
        auto ht = Kokkos::create_mirror_view(t), hz = Kokkos::create_mirror_view(z);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                ht(j, i) = version == 0 ? real(.75) : potential(g, 0, j, i, version);
                hz(j, i) = version == 0 ? real(-.625) : potential(g, 3, j, i, version);
            }
        }
        Kokkos::deep_copy(t, ht);
        Kokkos::deep_copy(z, hz);
        const auto t_before = snapshot(t), z_before = snapshot(z);
        Kokkos::deep_copy(output, sentinel);
        launch();
        auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
        std::vector<Real> direct;
        for (int n = 0; n < 4; ++n) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    direct.push_back(actual(n, j, i));
                }
            }
        }
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (j < h || j >= ny - h || i < h || i >= nx - h) {
                    for (int n = 0; n < 4; ++n) {
                        require(actual(n, j, i) == sentinel, "modified output halo");
                    }
                    continue;
                }
                const auto impulse = [&](int y, int x) {
                    return y == j && x == i ? real(1) : real(0);
                };
                const Real expected[4] = {reference_t(g, ht, j, i),
                    reference_z(g, hz, j, i),
                    reference_t(g, impulse, j, i),
                    reference_z(g, impulse, j, i)};
                for (int n = 0; n < 4; ++n) {
                    require(std::isfinite(actual(n, j, i)) &&
                                std::abs(actual(n, j, i) - expected[n]) <=
                                    tol * std::max(real(1), std::abs(expected[n])),
                        "native metric composition/diagonal mismatch");
                }
                require(actual(2, j, i) < real(0) && actual(3, j, i) < real(0),
                    "nonnegative diagonal in fixture");
                if (version == 0) {
                    require(actual(0, j, i) == real(0) && actual(1, j, i) == real(0),
                        "constant nullspace");
                }
                if (!varying && version != 0) {
                    const auto m = g.metric(0, j, i);
                    const Real analytic = m.J * (real(1) + real(.125) * version) *
                                          (real(2) * m.c11 + m.c12 + real(1.5) * m.c22);
                    for (int n = 0; n < 2; ++n) {
                        require(std::abs(actual(n, j, i) - analytic) <=
                                    tol * std::max(real(1), std::abs(analytic)),
                            "affine analytic Laplacian mismatch");
                    }
                }
            }
        }
#if defined(KOKKOS_ENABLE_CUDA)
        Kokkos::deep_copy(output, sentinel);
        graph.run();
        actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
        std::size_t p = 0;
        for (int n = 0; n < 4; ++n) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    require(actual(n, j, i) == direct[p++], "changed-input graph replay mismatch");
                }
            }
        }
#endif
        require(t_before == snapshot(t) && z_before == snapshot(z), "mutated potential inputs");
    }

    // Local Dirichlet manufactured solve. This does NOT exercise MPI halo
    // exchange, production channel BCs or global nullspace constraints.
    // The RHS is built with the independent host reference, not the solver.
    Plane rhs_t("rhs_t", ny, nx), rhs_z("rhs_z", ny, nx);
    Plane new_t("new_t", ny, nx), new_z("new_z", ny, nx);
    auto ht = Kokkos::create_mirror_view(t), hz = Kokkos::create_mirror_view(z);
    auto rt = Kokkos::create_mirror_view(rhs_t), rz = Kokkos::create_mirror_view(rhs_z);
    const auto exact_t = [&](int j, int i) {
        return potential(g, 0, j, i, 2);
    };
    const auto exact_z = [&](int j, int i) {
        return potential(g, 3, j, i, 2);
    };
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const bool inside = j >= h && j < ny - h && i >= h && i < nx - h;
            ht(j, i) = inside ? real(0) : exact_t(j, i);
            hz(j, i) = inside ? real(0) : exact_z(j, i);
            rt(j, i) = inside ? reference_t(g, exact_t, j, i) / g.metric(0, j, i).J : real(0);
            rz(j, i) = inside ? reference_z(g, exact_z, j, i) / g.metric(3, j, i).J : real(0);
        }
    }
    Kokkos::deep_copy(t, ht);
    Kokkos::deep_copy(z, hz);
    Kokkos::deep_copy(new_t, ht);
    Kokkos::deep_copy(new_z, hz);
    Kokkos::deep_copy(rhs_t, rt);
    Kokkos::deep_copy(rhs_z, rz);
    const auto saved_rhs_t = snapshot(rhs_t), saved_rhs_z = snapshot(rhs_z);
    for (int iter = 0; iter < 1200; ++iter) {
        Kokkos::parallel_for("ManufacturedEllipticRelaxation",
            policy,
            Relax<Layout>{operation, t, z, new_t, new_z, rhs_t, rhs_z});
        std::swap(t, new_t);
        std::swap(z, new_z);
    }
    ht = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t);
    hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
    const Real solution_tol = sizeof(Real) == sizeof(double) ? real(2e-10) : real(2e-4);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const bool inside = j >= h && j < ny - h && i >= h && i < nx - h;
            for (int n = 0; n < 2; ++n) {
                const Real actual = n == 0 ? ht(j, i) : hz(j, i),
                           expected = n == 0 ? exact_t(j, i) : exact_z(j, i);
                require(std::isfinite(actual) &&
                            std::abs(actual - expected) <=
                                (inside ? solution_tol * std::max(real(1), std::abs(expected))
                                        : real(0)),
                    "manufactured elliptic solution/halo mismatch");
            }
        }
    }
    require(saved_rhs_t == snapshot(rhs_t) && saved_rhs_z == snapshot(rhs_z), "mutated RHS");
    std::cout << "PASS elliptic " << layout_name << " shear=" << shear << " varying=" << varying
              << '\n';
}
} // namespace

int
main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int result = 0;
    try {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        require(size == 1, "this local operator test requires one MPI rank");
        for (Real shear : {real(-.8), real(.9)}) {
            for (bool varying : {false, true}) {
                test<Kokkos::LayoutLeft>(shear, varying, "LayoutLeft");
                test<Kokkos::LayoutRight>(shear, varying, "LayoutRight");
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        result = 1;
    }
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
