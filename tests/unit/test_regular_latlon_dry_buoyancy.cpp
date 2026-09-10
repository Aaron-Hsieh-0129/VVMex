#include "core/Field.hpp"
#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/RegularLatLonDryBuoyancy.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::Operators::RegularLatLonDryBuoyancy;

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

void
test_validation(const HorizontalDomainLayout& layout) {

    const CartesianGeometry cartesian(layout, real(1000.0), real(1000.0));

    bool rejected = false;

    try {
        const RegularLatLonDryBuoyancy invalid(cartesian);

        (void)invalid;
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }

    check(rejected,
        "RLL dry buoyancy must reject "
        "Cartesian geometry");
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

    const RegularLatLonDryBuoyancy buoyancy(geometry);

    RegularLatLonDryBuoyancy::prepare_execution();

#if defined(KOKKOS_ENABLE_CUDA)
    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "The CUDA graph test requires CUDA as "
        "the default execution space.");

    TestGraph graph(Kokkos::Cuda().cuda_stream());

    graph.begin();

    buoyancy.add_xi_tendency(th, thbar, gravity, replay_xi, k_begin, k_end);

    buoyancy.add_eta_tendency(th, thbar, gravity, replay_eta, k_begin, k_end);

    graph.finish();
    graph.launch();
#else
    buoyancy.add_xi_tendency(th, thbar, gravity, replay_xi, k_begin, k_end);

    buoyancy.add_eta_tendency(th, thbar, gravity, replay_eta, k_begin, k_end);
#endif

    buoyancy.add_xi_tendency(th, thbar, gravity, direct_xi, k_begin, k_end);

    buoyancy.add_eta_tendency(th, thbar, gravity, direct_eta, k_begin, k_end);

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

void test_moist(const RegularLatLonGeometry& geometry, Real radius) {
    const auto layout = make_layout();
    const int h = layout.halo, nz = 10;
    const int ny = layout.local_total_ny(), nx = layout.local_total_nx();
    Field<3> th("moist_th", {nz,ny,nx}), qv("moist_qv", {nz,ny,nx});
    Field<3> qp("moist_qp", {nz,ny,nx}), mask("moist_mask", {nz,ny,nx});
    Field<3> out("moist_out", {nz,ny,nx});
    Field<3> replay("moist_replay", {nz,ny,nx});
    Field<1> bar("moist_bar", {nz});
    Kokkos::View<Real> gravity("moist_g");
    Kokkos::deep_copy(gravity, real(9.806));
    Kokkos::deep_copy(th.get_mutable_device_data(), real(300.));
    Kokkos::deep_copy(bar.get_mutable_device_data(), real(300.));
    auto vapor = qv.get_host_data(), condensate = qp.get_host_data();
    auto masks = mask.get_host_data();
    for (int k=0; k<nz; ++k) for (int j=0; j<ny; ++j) for (int i=0; i<nx; ++i) {
        vapor(k,j,i) = real(.01)+real(.0001)*(i+2*j);
        condensate(k,j,i) = real(.001)+real(.00001)*(2*i+j);
        masks(k,j,i) = k==h && i==h && j==h ? real(0.) : real(1.);
    }
    Kokkos::deep_copy(qv.get_mutable_device_data(), vapor);
    Kokkos::deep_copy(qp.get_mutable_device_data(), condensate);
    Kokkos::deep_copy(mask.get_mutable_device_data(), masks);
    RegularLatLonDryBuoyancy op(geometry);
    op.prepare_execution();
    const Real pi = std::acos(real(-1.));
    for (bool xi : {true,false}) {
        Kokkos::deep_copy(out.get_mutable_device_data(), real(7.));
        Kokkos::deep_copy(replay.get_mutable_device_data(), real(7.));
#if defined(KOKKOS_ENABLE_CUDA)
        TestGraph graph(Kokkos::Cuda().cuda_stream());
        graph.begin();
        op.add_moist_tendency(th,bar,gravity,qv,qp,mask,replay,h,nz-h-1,h,xi);
        graph.finish();
        graph.launch();
#else
        op.add_moist_tendency(th,bar,gravity,qv,qp,mask,replay,h,nz-h-1,h,xi);
#endif
        op.add_moist_tendency(th,bar,gravity,qv,qp,mask,out,h,nz-h-1,h,xi);
        Kokkos::fence();
        const auto result = out.get_host_data();
        const auto replay_result = replay.get_host_data();
        bool match = true;
        for (int k=0; k<nz; ++k) for (int j=0; j<ny; ++j) for (int i=0; i<nx; ++i) {
            Real expected = real(7.);
            if (k>=h && k<nz-h-1 && j>=h && j<ny-h && i>=h && i<nx-h) {
                const Real phi = -pi/real(6.)+(j-h+real(.5))*(pi/real(3.)/layout.global_ny);
                const Real distance = xi ? radius*(pi/real(3.)/layout.global_ny)
                    : radius*std::cos(phi)*(real(2.)*pi/layout.global_nx);
                const Real slope = xi ? real(.608)*real(.0002)-real(.00001)
                                      : real(.608)*real(.0001)-real(.00002);
                expected += real(9.806)*slope/distance;
                if (k==h && j==h && i==h) expected = real(0.);
            }
            match = match && close(result(k,j,i),expected,real(2.e-15));
            match = match && result(k,j,i) == replay_result(k,j,i);
        }
        check(match,"Moist RLL buoyancy: physical gradients, signs, mask and untouched range");
    }
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
