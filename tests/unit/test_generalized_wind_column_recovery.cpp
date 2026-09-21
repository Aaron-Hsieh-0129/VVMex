#include "GeneralizedWindRecoveryTestGeometry.hpp"
#include "core/Field.hpp"
#include "dynamics/solvers/HorizontalWindColumnRecovery.hpp"
#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {
using namespace WindRecoveryTest;
using VVM::Core::Field;
using VVM::Dynamics::HorizontalWindColumnRecovery;
using Values = std::vector<Real>;

template <std::size_t D>
Values
snapshot(const Field<D>& f) {
    const auto a = f.get_host_data();
    Values r;
    if constexpr (D == 0) {
        r.push_back(a());
    }
    if constexpr (D == 1) {
        for (std::size_t k = 0; k < a.extent(0); ++k) {
            r.push_back(a(k));
        }
    }
    if constexpr (D == 2) {
        for (std::size_t j = 0; j < a.extent(0); ++j) {
            for (std::size_t i = 0; i < a.extent(1); ++i) {
                r.push_back(a(j, i));
            }
        }
    }
    if constexpr (D == 3) {
        for (std::size_t k = 0; k < a.extent(0); ++k) {
            for (std::size_t j = 0; j < a.extent(1); ++j) {
                for (std::size_t i = 0; i < a.extent(2); ++i) {
                    r.push_back(a(k, j, i));
                }
            }
        }
    }
    return r;
}
bool
same(const Values& a, const Values& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Real)) == 0);
}

#if defined(KOKKOS_ENABLE_CUDA)
void
cuda_check(cudaError_t s, const char* what) {
    if (s != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
    }
}
struct Graph {
    cudaStream_t stream = Kokkos::Cuda().cuda_stream();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph() {
        cudaStreamSynchronize(stream);
        if (exec) {
            cudaGraphExecDestroy(exec);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
    template <class F>
    void
    capture(F f) {
        Kokkos::fence();
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        try {
            f();
        }
        catch (...) {
            cudaStreamEndCapture(stream, &graph);
            throw;
        }
        cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    }
    void
    run() const {
        cuda_check(cudaGraphLaunch(exec, stream), "launch");
        cuda_check(cudaStreamSynchronize(stream), "synchronize");
    }
};
#endif

void
run(Real shear, bool varying, bool vvm_sign) {
    Geometry g(shear, varying);
    const int h = g.h(), nx = g.nx(), ny = g.ny(), nz = 9, bottom = 2, top = 6;
    const Real sentinel = real(-731);
    HorizontalWindColumnRecovery recovery(g);
    Field<2> ps("psi", {ny, nx}), ch("chi", {ny, nx});
    Field<3> w("w", {nz, ny, nx}), om1("omega1", {nz, ny, nx}), om2("omega2", {nz, ny, nx});
    Field<3> eta("eta_con", {nz, ny, nx});
    Field<1> ds("spacing", {nz - 1});
    Field<0> increment("covariant_top_increment", {});
    Field<3> u("u_cov", {nz, ny, nx}), v("v_cov", {nz, ny, nx});
    Field<3> ur("u_cov_replay", {nz, ny, nx}), vr("v_cov_replay", {nz, ny, nx});
    std::array<Real, nz> height{};
    Real top_increment = real(0);
    const auto fill = [&](int version) {
        auto dsh = ds.get_host_data();
        height[0] = real(0);
        for (int k = 0; k < nz - 1; ++k) {
            dsh(k) = real(.6) + real(.07) * k + real(.025) * version;
            height[k + 1] = height[k] + dsh(k);
        }
        Kokkos::deep_copy(ds.get_mutable_device_data(), dsh);
        top_increment = real(.375) * version;
        Kokkos::deep_copy(increment.get_mutable_device_data(), top_increment);
        auto ph = ps.get_host_data(), chh = ch.get_host_data();
        auto wh = w.get_host_data(), a = om1.get_host_data(), b = om2.get_host_data(),
             e = eta.get_host_data();
        const auto ww = [&](int k, int j, int i) {
            return real(.07) * version * (height[top] - height[k]) *
                   (real(.2) * g.p(0, i) - real(.3) * g.q(0, j) + real(.1) * g.p(0, i) * g.q(0, j));
        };
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                ph(j, i) = psi(g.p(3, i), g.q(3, j), version, 3);
                chh(j, i) = chi(g.p(0, i), g.q(0, j), version, 3);
                for (int k = 0; k < nz; ++k) {
                    wh(k, j, i) = ww(k, j, i);
                }
                for (int k = 0; k < nz; ++k) {
                    const Real mid =
                        k < nz - 1 ? real(.5) * (height[k] + height[k + 1]) : height[k];
                    const Real s1 = real(.13) + real(.02) * g.p(1, i) - real(.03) * g.q(1, j) +
                                    real(.017) * mid;
                    const Real s2 = real(-.11) + real(.04) * g.p(2, i) + real(.01) * g.q(2, j) -
                                    real(.013) * mid;
                    // Analytic horizontal derivative of the bilinear w field at
                    // its target face. Unlike production, no array difference.
                    const Real dw1 = real(.07) * version * (height[top] - height[k]) *
                                     (real(.2) + real(.1) * g.q(1, j));
                    const Real dw2 = real(.07) * version * (height[top] - height[k]) *
                                     (real(-.3) + real(.1) * g.p(2, i));
                    a(k, j, i) = (dw2 - s2) / g.metric(2, j, i).J;
                    b(k, j, i) = (s1 - dw1) / g.metric(1, j, i).J;
                    e(k, j, i) = -b(k, j, i);
                }
            }
        }
        Kokkos::deep_copy(ps.get_mutable_device_data(), ph);
        Kokkos::deep_copy(ch.get_mutable_device_data(), chh);
        Kokkos::deep_copy(w.get_mutable_device_data(), wh);
        Kokkos::deep_copy(om1.get_mutable_device_data(), a);
        Kokkos::deep_copy(om2.get_mutable_device_data(), b);
        Kokkos::deep_copy(eta.get_mutable_device_data(), e);
    };
    const auto inputs = [&]() {
        Values r;
        const auto add = [&](const auto& f) {
            const auto a = snapshot(f);
            r.insert(r.end(), a.begin(), a.end());
        };
        add(ps);
        add(ch);
        add(w);
        add(om1);
        add(om2);
        add(eta);
        add(ds);
        add(increment);
        return r;
    };
    const auto evaluate = [&](Field<3>& a, Field<3>& b) {
        if (vvm_sign) {
            recovery.recover_from_vvm_contravariant_state(ps,
                ch,
                w,
                om1,
                eta,
                ds,
                increment,
                a,
                b,
                bottom,
                top);
        }
        else {
            recovery.recover(ps, ch, w, om1, om2, ds, a, b, bottom, top);
        }
    };
    const auto verify = [&](const Field<3>& uf, const Field<3>& vf, int version) {
        const auto a = snapshot(uf), b = snapshot(vf);
        const Real tol = real(1024) * std::numeric_limits<Real>::epsilon();
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const std::size_t idx = (static_cast<std::size_t>(k) * ny + j) * nx + i;
                    if (k < bottom || k > top || j < h || j >= ny - h || i < h || i >= nx - h) {
                        require(a[idx] == sentinel && b[idx] == sentinel,
                            "column wrote outside requested domain");
                        continue;
                    }
                    const auto top_value = reference(g, j, i, version, 3);
                    const Real dz = height[k] - height[top],
                               dz2 = height[k] * height[k] - height[top] * height[top];
                    const Real q1 =
                        top_value[2] + (vvm_sign ? top_increment : real(0)) +
                        (real(.13) + real(.02) * g.p(1, i) - real(.03) * g.q(1, j)) * dz +
                        real(.5) * real(.017) * dz2;
                    const Real q2 =
                        top_value[3] +
                        (real(-.11) + real(.04) * g.p(2, i) + real(.01) * g.q(2, j)) * dz -
                        real(.5) * real(.013) * dz2;
                    require(std::isfinite(a[idx]) && std::isfinite(b[idx]) &&
                                std::abs(a[idx] - q1) <= tol * std::max(real(1), std::abs(q1)) &&
                                std::abs(b[idx] - q2) <= tol * std::max(real(1), std::abs(q2)),
                        "covariant column differs from analytic vertical integral");
                }
            }
        }
    };
    fill(1);
    HorizontalWindColumnRecovery::prepare_execution();
#if defined(KOKKOS_ENABLE_CUDA)
    Graph graph;
    graph.capture([&]() {
        evaluate(ur, vr);
    });
#endif
    for (int version : {1, 2, 4}) {
        fill(version);
        for (auto* f : {&u, &v, &ur, &vr}) {
            Kokkos::deep_copy(f->get_mutable_device_data(), sentinel);
        }
        const auto before = inputs();
        evaluate(u, v);
        Kokkos::fence();
        verify(u, v, version);
        require(same(before, inputs()), "direct recovery modified an input");
#if defined(KOKKOS_ENABLE_CUDA)
        graph.run();
#else
        evaluate(ur, vr);
        Kokkos::fence();
#endif
        verify(ur, vr, version);
        require(same(before, inputs()), "replay recovery modified an input");
        require(same(snapshot(u), snapshot(ur)) && same(snapshot(v), snapshot(vr)),
            "direct/replay column differs");
    }
    bool rejected = false;
    try {
        recovery.recover(ps, ch, w, om1, om2, ds, u, u, bottom, top);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "aliased outputs were accepted");
    rejected = false;
    try {
        recovery.recover(ps, ch, w, om1, om2, ds, u, v, bottom, nz);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid column range was accepted");
    std::cout << "PASS: covariant column, shear=" << shear << ", varying=" << varying
              << ", VVM-sign=" << vvm_sign << '\n';
}
} // namespace

int
main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int result = 0;
    try {
        int ranks = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        require(ranks == 1, "test requires one MPI rank");
        for (Real s : {real(.8), real(-.7)}) {
            for (bool varying : {false, true}) {
                for (bool vvm_sign : {false, true}) {
                    run(s, varying, vvm_sign);
                }
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
