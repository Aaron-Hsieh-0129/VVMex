#include "core/Field.hpp"
#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/GeneralizedBuoyancy.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <limits>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::GeometryField2D;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometry;
using VVM::Core::Geometry::HorizontalGeometryDeviceView;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::Operators::GeneralizedBuoyancy;
using VVM::Dynamics::Operators::make_generalized_buoyancy_device_view;

int failures = 0;

void
check(const bool condition, const char* message) {

    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

bool
close(const Real actual, const Real expected, const Real tolerance) {

    return std::abs(actual - expected) <=
           tolerance * std::max(real(1.0), std::max(std::abs(actual), std::abs(expected)));
}

HorizontalDomainLayout
make_layout() {
    HorizontalDomainLayout layout;

    layout.global_nx = 24;
    layout.global_ny = 12;
    layout.local_physical_nx = 24;
    layout.local_physical_ny = 12;
    layout.global_start_i = 0;
    layout.global_start_j = 0;
    layout.halo = 2;
    layout.panel_id = -1;

    return layout;
}

#if defined(KOKKOS_ENABLE_CUDA)

void
require_cuda(const cudaError_t status, const char* operation) {

    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class TestGraph {
public:
    explicit TestGraph(const cudaStream_t stream) : stream_(stream) {}

    ~TestGraph() {
        if (capturing_) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(stream_, &abandoned);

            if (abandoned != nullptr) {
                cudaGraphDestroy(abandoned);
            }
        }

        cudaStreamSynchronize(stream_);

        if (executable_ != nullptr) {
            cudaGraphExecDestroy(executable_);
        }

        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
        }
    }

    void
    begin() {
        require_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
            "begin dry-buoyancy capture");

        capturing_ = true;
    }

    void
    finish() {
        const cudaError_t status = cudaStreamEndCapture(stream_, &graph_);

        capturing_ = false;

        require_cuda(status, "end dry-buoyancy capture");

        if (graph_ == nullptr) {
            throw std::runtime_error("Dry-buoyancy capture returned "
                                     "a null graph.");
        }

        require_cuda(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0),
            "instantiate dry-buoyancy graph");
    }

    void
    launch() {
        require_cuda(cudaGraphLaunch(executable_, stream_), "launch dry-buoyancy graph");

        require_cuda(cudaStreamSynchronize(stream_), "complete dry-buoyancy graph");
    }

private:
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    bool capturing_ = false;
};

#endif

template <typename Function>
void
expect_invalid(Function function, const char* message) {
    bool rejected = false;
    try {
        function();
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, message);
}

void
test_validation(const HorizontalDomainLayout& layout) {
    const CartesianGeometry cartesian(layout, real(1000.0), real(1000.0));
    const GeneralizedBuoyancy op(cartesian);
    auto no_halo = layout;
    no_halo.halo = 0;
    const CartesianGeometry narrow(no_halo, real(1000.0), real(1000.0));
    expect_invalid([&] {
        GeneralizedBuoyancy invalid(narrow);
    }, "Generalized buoyancy must reject a missing horizontal halo");

    const int nz = 8, ny = layout.local_total_ny(), nx = layout.local_total_nx();
    Field<3> th("validation_th", {nz, ny, nx}), out("validation_out", {nz, ny, nx});
    Field<3> qv("validation_qv", {nz, ny, nx}), qp("validation_qp", {nz, ny, nx});
    Field<3> mask("validation_mask", {nz, ny, nx}), wrong("wrong_shape", {nz, ny + 1, nx});
    Field<1> bar("validation_bar", {nz}), short_bar("short_bar", {6});
    Kokkos::View<Real> gravity("validation_gravity"), empty_gravity;
    Kokkos::deep_copy(gravity, real(9.81));
    Kokkos::deep_copy(bar.get_mutable_device_data(), real(300.0));

    expect_invalid([&] {
        op.add_xi_tendency(th, bar, gravity, th, 2, 6);
    }, "Buoyancy must reject input/output aliasing");
    expect_invalid([&] {
        op.add_xi_tendency(th, bar, gravity, out, -1, 6);
    }, "Buoyancy must reject negative start index");
    expect_invalid([&] {
        op.add_xi_tendency(th, bar, gravity, out, 2, 2);
    }, "Buoyancy must reject an empty vertical range");
    expect_invalid([&] {
        op.add_eta_tendency(th, bar, gravity, out, 2, nz);
    }, "Buoyancy must require the upper adjacent T level");
    expect_invalid([&] {
        op.add_xi_tendency(th, short_bar, gravity, out, 2, 6);
    }, "Buoyancy must reject insufficient thbar entries");
    expect_invalid([&] {
        op.add_eta_tendency(th, bar, empty_gravity, out, 2, 6);
    }, "Buoyancy must reject unallocated gravity");
    expect_invalid([&] {
        op.add_xi_tendency(th, bar, gravity, wrong, 2, 6);
    }, "Buoyancy must reject incorrect output extents");
    expect_invalid([&] {
        op.add_moist_tendency(th, bar, gravity, qv, qp, mask, qv, 2, 6, 2, true);
    }, "Moist buoyancy must reject aliasing a moisture input");
    expect_invalid([&] {
        op.add_moist_tendency(th, bar, gravity, wrong, qp, mask, out, 2, 6, 2, false);
    }, "Moist buoyancy must reject incorrect moisture extents");
}

void
test_analytic_and_replay(const RegularLatLonGeometry& geometry, const Real radius) {

    const auto layout = geometry.layout();

    const int h = layout.halo;
    const int ny = layout.local_total_ny();
    const int nx = layout.local_total_nx();
    const int nz = 8;
    const int k_begin = 2;
    const int k_end = 6;

    const Real sentinel = real(-4.5);
    const Real gravity_value = real(9.81);
    const Real longitude_slope = real(0.018);
    const Real latitude_slope = real(-0.027);

    Field<3> th("dry_buoyancy_th", {nz, ny, nx});
    Field<1> thbar("dry_buoyancy_thbar", {nz});

    Field<3> direct_xi("dry_buoyancy_direct_xi", {nz, ny, nx});
    Field<3> direct_eta("dry_buoyancy_direct_eta", {nz, ny, nx});
    Field<3> replay_xi("dry_buoyancy_replay_xi", {nz, ny, nx});
    Field<3> replay_eta("dry_buoyancy_replay_eta", {nz, ny, nx});

    Kokkos::View<Real> gravity("dry_buoyancy_gravity");

    Kokkos::deep_copy(gravity, gravity_value);

    auto th_data = th.get_mutable_device_data();
    auto thbar_data = thbar.get_mutable_device_data();

    const auto t_geometry = geometry.device_view(HorizontalLocation::T);

    Kokkos::parallel_for("InitializeDryBuoyancyThbar",
        Kokkos::RangePolicy<>(0, nz),
        KOKKOS_LAMBDA(
            const int k) { thbar_data(k) = real(295.0) + real(1.25) * static_cast<Real>(k); });

    Kokkos::parallel_for("InitializeDryBuoyancyTheta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real normalized_theta = real(1.0) + longitude_slope * t_geometry.longitude(j, i) +
                                          latitude_slope * t_geometry.latitude(j, i);

            th_data(k, j, i) = thbar_data(k) * normalized_theta;
        });

    Kokkos::deep_copy(direct_xi.get_mutable_device_data(), sentinel);
    Kokkos::deep_copy(direct_eta.get_mutable_device_data(), sentinel);
    Kokkos::deep_copy(replay_xi.get_mutable_device_data(), sentinel);
    Kokkos::deep_copy(replay_eta.get_mutable_device_data(), sentinel);

    Kokkos::fence();

    const GeneralizedBuoyancy buoyancy(geometry);
    const auto xi_weight =
        geometry.device_view(HorizontalLocation::V).contravariant_to_physical.a11;
    const auto eta_weight =
        geometry.device_view(HorizontalLocation::U).contravariant_to_physical.a22;

    GeneralizedBuoyancy::prepare_execution();

#if defined(KOKKOS_ENABLE_CUDA)
    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "The CUDA graph test requires CUDA as "
        "the default execution space.");

    TestGraph graph(Kokkos::Cuda().cuda_stream());

    graph.begin();

    buoyancy.add_xi_tendency(th, thbar, gravity, replay_xi, k_begin, k_end, xi_weight);

    buoyancy.add_eta_tendency(th, thbar, gravity, replay_eta, k_begin, k_end, eta_weight);

    graph.finish();
    graph.launch();
#else
    buoyancy.add_xi_tendency(th, thbar, gravity, replay_xi, k_begin, k_end, xi_weight);

    buoyancy.add_eta_tendency(th, thbar, gravity, replay_eta, k_begin, k_end, eta_weight);
#endif

    buoyancy.add_xi_tendency(th, thbar, gravity, direct_xi, k_begin, k_end, xi_weight);

    buoyancy.add_eta_tendency(th, thbar, gravity, direct_eta, k_begin, k_end, eta_weight);

    Kokkos::fence();

    const auto direct_xi_host = direct_xi.get_host_data();
    const auto direct_eta_host = direct_eta.get_host_data();
    const auto replay_xi_host = replay_xi.get_host_data();
    const auto replay_eta_host = replay_eta.get_host_data();

    const auto u_geometry = geometry.device_view(HorizontalLocation::U);

    Kokkos::View<Real**> expected_eta("dry_buoyancy_expected_eta", ny, nx);

    Kokkos::parallel_for("EvaluateAnalyticDryBuoyancyEta",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            expected_eta(j, i) =
                gravity_value * longitude_slope / (radius * Kokkos::cos(u_geometry.latitude(j, i)));
        });

    const auto expected_eta_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected_eta);

    const Real expected_xi_increment = gravity_value * latitude_slope / radius;

    const Real tolerance = sizeof(Real) == sizeof(double) ? real(2.0e-11) : real(2.0e-5);

    bool analytic_match = true;
    bool replay_match = true;
    bool untouched = true;
    bool finite = true;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const bool written =
                    k >= k_begin && k < k_end && j >= h && j < ny - h && i >= h && i < nx - h;

                if (written) {
                    analytic_match = analytic_match &&
                                     close(direct_xi_host(k, j, i),
                                         sentinel + expected_xi_increment,
                                         tolerance) &&
                                     close(direct_eta_host(k, j, i),
                                         sentinel + expected_eta_host(j, i),
                                         tolerance);

                    replay_match =
                        replay_match &&
                        close(replay_xi_host(k, j, i), direct_xi_host(k, j, i), tolerance) &&
                        close(replay_eta_host(k, j, i), direct_eta_host(k, j, i), tolerance);

                    finite = finite && std::isfinite(direct_xi_host(k, j, i)) &&
                             std::isfinite(direct_eta_host(k, j, i)) &&
                             std::isfinite(replay_xi_host(k, j, i)) &&
                             std::isfinite(replay_eta_host(k, j, i));
                }
                else {
                    untouched = untouched && direct_xi_host(k, j, i) == sentinel &&
                                direct_eta_host(k, j, i) == sentinel &&
                                replay_xi_host(k, j, i) == sentinel &&
                                replay_eta_host(k, j, i) == sentinel;
                }
            }
        }
    }

    check(analytic_match,
        "RLL dry buoyancy must match analytic "
        "physical eastward/northward gradients");

    check(replay_match,
        "RLL dry buoyancy graph replay must match "
        "ordinary execution");

    check(untouched,
        "RLL dry buoyancy must preserve halos and "
        "levels outside its write range");

    check(finite, "RLL dry buoyancy must produce finite values");
}

void
test_moist(const RegularLatLonGeometry& geometry, Real radius) {
    const auto layout = make_layout();
    const int h = layout.halo, nz = 10;
    const int ny = layout.local_total_ny(), nx = layout.local_total_nx();
    Field<3> th("moist_th", {nz, ny, nx}), qv("moist_qv", {nz, ny, nx});
    Field<3> qp("moist_qp", {nz, ny, nx}), mask("moist_mask", {nz, ny, nx});
    Field<3> out("moist_out", {nz, ny, nx});
    Field<3> replay("moist_replay", {nz, ny, nx});
    Field<1> bar("moist_bar", {nz});
    Kokkos::View<Real> gravity("moist_g");
    Kokkos::deep_copy(gravity, real(9.806));
    Kokkos::deep_copy(th.get_mutable_device_data(), real(300.));
    Kokkos::deep_copy(bar.get_mutable_device_data(), real(300.));
    auto vapor = qv.get_host_data(), condensate = qp.get_host_data();
    auto masks = mask.get_host_data();
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                vapor(k, j, i) = real(.01) + real(.0001) * (i + 2 * j);
                condensate(k, j, i) = real(.001) + real(.00001) * (2 * i + j);
                masks(k, j, i) = k == h && i == h && j == h ? real(0.) : real(1.);
            }
        }
    }
    Kokkos::deep_copy(qv.get_mutable_device_data(), vapor);
    Kokkos::deep_copy(qp.get_mutable_device_data(), condensate);
    Kokkos::deep_copy(mask.get_mutable_device_data(), masks);
    GeneralizedBuoyancy op(geometry);
    op.prepare_execution();
    const Real pi = std::acos(real(-1.));
    for (bool xi : {true, false}) {
        const auto weight =
            xi ? geometry.device_view(HorizontalLocation::V).contravariant_to_physical.a11
               : geometry.device_view(HorizontalLocation::U).contravariant_to_physical.a22;
        Kokkos::deep_copy(out.get_mutable_device_data(), real(7.));
        Kokkos::deep_copy(replay.get_mutable_device_data(), real(7.));
#if defined(KOKKOS_ENABLE_CUDA)
        TestGraph graph(Kokkos::Cuda().cuda_stream());
        graph.begin();
        op.add_moist_tendency(th, bar, gravity, qv, qp, mask, replay, h, nz - h - 1, h, xi, weight);
        graph.finish();
        graph.launch();
#else
        op.add_moist_tendency(th, bar, gravity, qv, qp, mask, replay, h, nz - h - 1, h, xi, weight);
#endif
        op.add_moist_tendency(th, bar, gravity, qv, qp, mask, out, h, nz - h - 1, h, xi, weight);
        Kokkos::fence();
        const auto result = out.get_host_data();
        const auto replay_result = replay.get_host_data();
        bool match = true;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    Real expected = real(7.);
                    if (k >= h && k < nz - h - 1 && j >= h && j < ny - h && i >= h && i < nx - h) {
                        const Real phi = -pi / real(6.) +
                                         (j - h + real(.5)) * (pi / real(3.) / layout.global_ny);
                        const Real distance =
                            xi ? radius * (pi / real(3.) / layout.global_ny)
                               : radius * std::cos(phi) * (real(2.) * pi / layout.global_nx);
                        const Real slope = xi ? real(.608) * real(.0002) - real(.00001)
                                              : real(.608) * real(.0001) - real(.00002);
                        expected += real(9.806) * slope / distance;
                        if (k == h && j == h && i == h) {
                            expected = real(0.);
                        }
                    }
                    match = match && close(result(k, j, i), expected, real(2.e-15));
                    match = match && result(k, j, i) == replay_result(k, j, i);
                }
            }
        }
        check(match, "Moist RLL buoyancy: physical gradients, signs, mask and untouched range");
    }
}

// Test-only nonorthogonal chart: x=2*q1+0.75*q2, y=3*q2, J=6.
// Reuse the Cartesian enum as a fixture label; no new production geometry
// or cubed-sphere implementation is introduced by this test.
class AffineGeometry final : public HorizontalGeometry {
public:
    explicit AffineGeometry(const HorizontalDomainLayout& layout)
        : coordinates_(layout, real(0.25), real(0.5)) {}
    GeometryKind
    kind() const noexcept override {
        return GeometryKind::Cartesian;
    }
    const char*
    name() const noexcept override {
        return "test_nonorthogonal_affine";
    }
    const HorizontalDomainLayout&
    layout() const noexcept override {
        return coordinates_.layout();
    }
    Real
    dq1() const noexcept override {
        return coordinates_.dq1();
    }
    Real
    dq2() const noexcept override {
        return coordinates_.dq2();
    }

protected:
    HorizontalGeometryDeviceView
    device_view_impl(HorizontalLocation location) const override {
        auto v = coordinates_.device_view(location);
        const auto c = [](Real x) {
            return GeometryField2D::constant_value(x);
        };
        v.sqrt_g = c(real(6.0));
        v.inv_sqrt_g = c(real(1.0) / real(6.0));
        v.g_cov = {c(real(4.0)), c(real(1.5)), c(real(9.5625))};
        v.sqrt_g_g_contra = {c(real(9.5625) / real(6.0)),
            c(real(-1.5) / real(6.0)),
            c(real(4.0) / real(6.0))};
        v.contravariant_to_physical = {c(real(2.0)), c(real(0.75)), c(real(0.0)), c(real(3.0))};
        v.physical_to_contravariant = {c(real(0.5)),
            c(real(-0.125)),
            c(real(0.0)),
            c(real(1.0) / real(3.0))};
        return v;
    }

private:
    CartesianGeometry coordinates_;
};

std::vector<Real>
snapshot_volume(const Field<3>& field) {
    const auto h = field.get_host_data();
    std::vector<Real> values;
    values.reserve(h.size());
    for (std::size_t k = 0; k < h.extent(0); ++k) {
        for (std::size_t j = 0; j < h.extent(1); ++j) {
            for (std::size_t i = 0; i < h.extent(2); ++i) {
                values.push_back(h(k, j, i));
            }
        }
    }
    return values;
}

// Canonical outputs can be O(1e-14) on an Earth-radius RLL grid. Start the
// written region at ZERO and compare relatively to the actual source, not
// to a large sentinel or max(1,abs(source)). This detects missing sources,
// wrong Jacobians and eta sign even in float precision.
template <typename Jacobian>
void
test_canonical_fields(const HorizontalGeometry& geometry, const char* label, Jacobian jacobian) {
    const auto layout = geometry.layout();
    const int h = layout.halo, nz = 8, begin = 1, end = nz - 1;
    const int ny = layout.local_total_ny(), nx = layout.local_total_nx();
    const Real g0 = real(8.0), sentinel = real(-4.5);
    const Real tolerance = real(512.0) * std::numeric_limits<Real>::epsilon();
    const Real dq1 = geometry.dq1(), dq2 = geometry.dq2();
    Field<3> th("canonical_th", {nz, ny, nx}), qv("canonical_qv", {nz, ny, nx});
    Field<3> qp("canonical_qp", {nz, ny, nx}), mask("canonical_mask", {nz, ny, nx});
    Field<3> dx("canonical_xi", {nz, ny, nx}), de("canonical_eta", {nz, ny, nx});
    Field<3> rx("replay_xi", {nz, ny, nx}), re("replay_eta", {nz, ny, nx});
    Field<1> bar("canonical_bar", {nz});
    Kokkos::View<Real> gravity("canonical_g");
    const GeneralizedBuoyancy op(geometry);
    GeneralizedBuoyancy::prepare_execution();

    for (int mode = 0; mode < 4; ++mode) {
        const bool moist = mode == 1 || mode == 2;
        const bool thermal = mode == 0 || mode == 2;
        const auto a = [&](int k) {
            return thermal ? real(0.125) * (real(1.0) + real(k) / real(16.0)) : real(0.0);
        };
        const auto b = [&](int k) {
            return thermal ? real(-0.0625) * (real(1.0) + real(k) / real(8.0)) : real(0.0);
        };
        const auto vapor_scale = [](int k) {
            return real(1.0) + real(k) / real(8.0);
        };
        const auto condensate_scale = [](int k) {
            return real(1.0) + real(k) / real(16.0);
        };
        const Real v1 = real(1.0) / real(4096.0), v2 = real(-1.0) / real(8192.0);
        const Real p1 = real(1.0) / real(16384.0), p2 = real(1.0) / real(32768.0);
        auto thh = th.get_host_data(), vh = qv.get_host_data(), ph = qp.get_host_data();
        auto mh = mask.get_host_data(), seed = dx.get_host_data();
        auto bh = bar.get_host_data();
        for (int k = 0; k < nz; ++k) {
            bh(k) = real(256.0) + real(16.0) * real(k);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    thh(k, j, i) = bh(k) * (real(2.0) + a(k) * real(i - h) + b(k) * real(j - h));
                    vh(k, j, i) =
                        real(0.03125) + vapor_scale(k) * (v1 * real(i - h) + v2 * real(j - h));
                    ph(k, j, i) = real(0.0078125) +
                                  condensate_scale(k) * (p1 * real(i - h) + p2 * real(j - h));
                    // Second zero lies ABOVE max_topo_idx and must not reset.
                    mh(k, j, i) =
                        j == h && i == h && (k == begin || k == begin + 1) ? real(0.0) : real(1.0);
                    const bool written =
                        k >= begin && k < end && j >= h && j < ny - h && i >= h && i < nx - h;
                    seed(k, j, i) = written ? real(0.0) : sentinel;
                }
            }
        }
        Kokkos::deep_copy(th.get_mutable_device_data(), thh);
        Kokkos::deep_copy(qv.get_mutable_device_data(), vh);
        Kokkos::deep_copy(qp.get_mutable_device_data(), ph);
        Kokkos::deep_copy(mask.get_mutable_device_data(), mh);
        Kokkos::deep_copy(bar.get_mutable_device_data(), bh);
        for (auto* field : {&dx, &de, &rx, &re}) {
            Kokkos::deep_copy(field->get_mutable_device_data(), seed);
        }
        Kokkos::deep_copy(gravity, g0);
        const auto initial_th = snapshot_volume(th), initial_qv = snapshot_volume(qv);
        const auto initial_qp = snapshot_volume(qp), initial_mask = snapshot_volume(mask);
        const auto bar_before = bar.get_host_data();
        std::vector<Real> initial_bar;
        for (int k = 0; k < nz; ++k) {
            initial_bar.push_back(bar_before(k));
        }

        const auto run = [&](Field<3>& xi, Field<3>& eta) {
            if (moist) {
                op.add_moist_tendency(th, bar, gravity, qv, qp, mask, xi, begin, end, begin, true);
                op.add_moist_tendency(th,
                    bar,
                    gravity,
                    qv,
                    qp,
                    mask,
                    eta,
                    begin,
                    end,
                    begin,
                    false);
            }
            else {
                op.add_xi_tendency(th, bar, gravity, xi, begin, end);
                op.add_eta_tendency(th, bar, gravity, eta, begin, end);
            }
        };
#if defined(KOKKOS_ENABLE_CUDA)
        TestGraph graph(Kokkos::Cuda().cuda_stream());
        graph.begin();
        run(rx, re);
        graph.finish();
#endif
        bool correct = true, replay_matches = true, inputs_unchanged = true;
        for (int pass = 0; pass < 2; ++pass) {
            // Replay must read the live device scalar, not a captured host value.
            Kokkos::deep_copy(gravity, pass == 0 ? g0 : real(2.0) * g0);
#if defined(KOKKOS_ENABLE_CUDA)
            graph.launch();
#else
            run(rx, re);
#endif
            run(dx, de);
            Kokkos::fence();
            const auto x = dx.get_host_data(), e = de.get_host_data();
            const auto xr = rx.get_host_data(), er = re.get_host_data();
            const Real accumulated_gravity = (pass == 0 ? real(1.0) : real(3.0)) * g0;
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        const bool written =
                            k >= begin && k < end && j >= h && j < ny - h && i >= h && i < nx - h;
                        const bool reset = moist && k == begin && j == h && i == h;
                        Real ex = sentinel, ee = sentinel;
                        if (written) {
                            Real d1 = real(0.5) * (a(k) + a(k + 1)),
                                 d2 = real(0.5) * (b(k) + b(k + 1));
                            if (moist) {
                                const Real vs = real(0.5) * (vapor_scale(k) + vapor_scale(k + 1));
                                const Real ps =
                                    real(0.5) * (condensate_scale(k) + condensate_scale(k + 1));
                                d1 += real(0.608) * vs * v1 - ps * p1;
                                d2 += real(0.608) * vs * v2 - ps * p2;
                            }
                            ex = reset ? real(0.0)
                                       : accumulated_gravity * d2 / (dq2 * jacobian(j, true));
                            ee = reset ? real(0.0)
                                       : accumulated_gravity * d1 / (dq1 * jacobian(j, false));
                        }
                        const auto matches = [&](Real actual, Real expected) {
                            if (!std::isfinite(actual)) {
                                return false;
                            }
                            return expected == real(0.0) || !written
                                       ? actual == expected
                                       : std::abs(actual - expected) <=
                                             tolerance * std::abs(expected);
                        };
                        correct = correct && matches(x(k, j, i), ex) && matches(e(k, j, i), ee);
                        replay_matches =
                            replay_matches && matches(xr(k, j, i), ex) && matches(er(k, j, i), ee);
                    }
                }
            }
        }
        const auto final_bar = bar.get_host_data();
        for (int k = 0; k < nz; ++k) {
            inputs_unchanged = inputs_unchanged && final_bar(k) == initial_bar[k];
        }
        inputs_unchanged = inputs_unchanged && snapshot_volume(th) == initial_th &&
                           snapshot_volume(qv) == initial_qv && snapshot_volume(qp) == initial_qp &&
                           snapshot_volume(mask) == initial_mask;
        std::printf("Canonical buoyancy %s mode=%d: %s\n",
            label,
            mode,
            correct && replay_matches && inputs_unchanged ? "PASS" : "FAIL");
        check(correct, "Canonical analytic source, sign, vertical average, accumulation and mask");
        check(replay_matches, "Canonical replay must use live inputs and preserve write range");
        check(inputs_unchanged, "Buoyancy must not mutate input fields");
    }
}

void
test_affine_tensor_sign(const AffineGeometry& geometry) {
    const auto operation = make_generalized_buoyancy_device_view(geometry);
    Kokkos::View<Real*> result("affine_tensor_sign", 2);
    // s=A*x+B*y. Curl(g*s*e_z)=(g*B,-g*A,0) in physical axes.
    const Real A = real(0.125), B = real(-0.25), g = real(8.0);
    const Real d1 = real(2.0) * A * geometry.dq1();
    const Real d2 = (real(0.75) * A + real(3.0) * B) * geometry.dq2();
    Kokkos::parallel_for("AffineBuoyancyTensorSign",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            const Real s1 = operation.calculate_omega1_at_v(0, 0, d2, d2, g);
            const Real s2 = operation.calculate_omega2_at_u(0, 0, d1, d1, g);
            result(0) = real(2.0) * s1 + real(0.75) * s2;
            result(1) = real(3.0) * s2;
        });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    const Real tol = real(32.0) * std::numeric_limits<Real>::epsilon();
    check(std::isfinite(host(0)) && std::isfinite(host(1)) && close(host(0), g * B, tol) &&
              close(host(1), -g * A, tol),
        "Nonorthogonal force curl must return true omega components before VVM eta sign");
}

} // namespace

int
main(int argc, char** argv) {

    Kokkos::initialize(argc, argv);

    {
        try {
            const auto layout = make_layout();

            const Real pi = std::acos(real(-1.0));
            const Real radius = real(6371220.0);

            const RegularLatLonGeometry geometry(layout,
                real(2.0) * pi / static_cast<Real>(layout.global_nx),
                (pi / real(3.0)) / static_cast<Real>(layout.global_ny),
                -pi,
                -pi / real(6.0),
                radius);

            test_validation(layout);
            test_analytic_and_replay(geometry, radius);
            test_moist(geometry, radius);
            const int h = layout.halo;
            const Real dy = geometry.dq2(), south = -pi / real(6.0);
            test_canonical_fields(geometry, "RLL", [&](int j, bool xi) {
                const Real phi = south + (real(j - h) + (xi ? real(1.0) : real(0.5))) * dy;
                return radius * radius * std::cos(phi);
            });
            const CartesianGeometry cartesian(layout, real(2.0), real(4.0));
            test_canonical_fields(cartesian, "Cartesian", [](int, bool) {
                return real(1.0);
            });
            const AffineGeometry affine(layout);
            test_canonical_fields(affine, "Nonorthogonal affine", [](int, bool) {
                return real(6.0);
            });
            test_affine_tensor_sign(affine);
        }
        catch (const std::exception& error) {
            ++failures;

            std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
        }
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::fprintf(stdout, "test_regular_latlon_dry_buoyancy: PASS\n");
    }
    else {
        std::fprintf(stderr,
            "test_regular_latlon_dry_buoyancy: "
            "%d failure(s)\n",
            failures);
    }

    return failures == 0 ? 0 : 1;
}
