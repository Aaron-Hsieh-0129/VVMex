#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/solvers/VerticalWindDiagnostic.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using namespace VVM::Core::Geometry;

constexpr int nz = 6;
constexpr int bottom = 1;
constexpr int top = 4;
constexpr Real sentinel = Real(-12345);

struct Profile {
    Kokkos::View<Real**> values;
    int row;

    KOKKOS_INLINE_FUNCTION
    Real operator()(int k) const {
        return values(row, k);
    }
};

template<typename Volume>
void launch_kernel(VVM::Dynamics::VerticalWindDiagnosticDeviceView operation, Volume xi, Volume eta, Volume w, Volume zeta, Kokkos::View<Real**> profiles, Kokkos::View<Real****> result, int nx, int ny, int h, Real inverse_dz, Real shift) {
    const Profile rho{profiles, 0}, rho_up{profiles, 1}, flex_mid{profiles, 2}, flex_up{profiles, 3}, spacing{profiles, 4};
    const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h});

    Kokkos::parallel_for("VerticalWindDiagnosticBatch", Kokkos::Experimental::require(policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight), KOKKOS_LAMBDA(int j, int i) {
        for (int k = bottom; k <= top; ++k) {
            const Real rhs = operation.calculate_weighted_rhs_at_t(xi, eta, k, j, i);
            const auto row = operation.calculate_row_at_t(rho, rho_up, flex_mid, flex_up, inverse_dz, shift, k, j, i);

            result(0, k, j, i) = rhs;
            result(1, k, j, i) = operation.calculate_vorticity_divergence_at_z(xi, eta, k, j, i);
            result(2, k, j, i) = row.lower;
            result(3, k, j, i) = row.diagonal;
            result(4, k, j, i) = row.upper;
            result(5, k, j, i) = row.horizontal_neighbors(w, k, j, i);
            result(6, k, j, i) = row.line_rhs(w, rhs, k, j, i);
            result(7, k, j, i) = row.jacobi_value(w, rho_up, rhs, k, j, i);
        }

        operation.integrate_zeta_column(xi, eta, spacing, zeta, bottom, top, j, i, true);
    });
}

#if defined(KOKKOS_ENABLE_CUDA)
void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(code));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
#endif

template<typename Layout>
int run_case(const HorizontalGeometry& geometry, Real radius, Real south, const char* layout_name, bool stretched) {
    using Volume = Kokkos::View<Real***, Layout>;

    const bool spherical = geometry.kind() == GeometryKind::RegularLatLon;
    const auto layout = geometry.layout();
    const int nx = layout.local_total_nx(), ny = layout.local_total_ny(), h = layout.halo;
    const Real dx = geometry.dq1(), dy = geometry.dq2();
    const Real inverse_dz = real(0.01), shift = stretched ? real(0.25) : real(0.0);
    const auto operation = VVM::Dynamics::make_vertical_wind_diagnostic_device_view(geometry);

    Volume xi("xi", nz, ny, nx), eta("eta", nz, ny, nx), w("previous_w", nz, ny, nx), zeta("zeta", nz, ny, nx);
    Kokkos::View<Real**> profiles("profiles", 5, nz);
    Kokkos::View<Real****> result("rows_and_rhs", 8, nz, ny, nx);

    // Independent mirrors: source-preservation checks must not alias CPU inputs.
    auto hx = Kokkos::create_mirror(xi), he = Kokkos::create_mirror(eta), hw = Kokkos::create_mirror(w), hz = Kokkos::create_mirror(zeta);
    auto hp = Kokkos::create_mirror(profiles);

    const auto initialize = [&](int step) {
        const Real factor = step == 2 ? real(0.0) : Real(step + 1);

        for (int k = 0; k < nz; ++k) {
            hp(0, k) = stretched ? real(1.2) - real(0.06) * k : real(1.0);
            hp(1, k) = stretched ? real(1.17) - real(0.06) * k : real(1.0);
            hp(2, k) = stretched ? real(1.1) + real(0.04) * k : real(1.0);
            hp(3, k) = stretched ? real(0.9) + real(0.05) * k : real(1.0);
            hp(4, k) = real(1.0) / (inverse_dz * hp(3, k));

            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    hx(k, j, i) = factor * (real(0.01) * k + real(0.002) * i * i + real(0.003) * j);
                    he(k, j, i) = factor * (-real(0.015) * k + real(0.004) * i - real(0.0015) * j * j);
                    hw(k, j, i) = factor * (real(0.3) * k + real(0.02) * i * i - real(0.015) * j * j + real(0.001) * i * j);
                    hz(k, j, i) = k == top ? factor * real(0.02) * (i + j) : sentinel;
                }
            }
        }

        Kokkos::deep_copy(xi, hx);
        Kokkos::deep_copy(eta, he);
        Kokkos::deep_copy(w, hw);
        Kokkos::deep_copy(zeta, hz);
        Kokkos::deep_copy(profiles, hp);
        Kokkos::deep_copy(result, sentinel);
    };

    const auto launch = [&]() {
        launch_kernel(operation, xi, eta, w, zeta, profiles, result, nx, ny, h, inverse_dz, shift);
    };

    // Analytic metrics, independent of the geometry device arrays.
    const auto cos_center = [&](int j) {
        return spherical ? std::cos(south + (Real(j - h) + real(0.5)) * dy) : real(1.0);
    };

    const auto cos_face = [&](int j) {
        return spherical ? std::cos(south + Real(j - h + 1) * dy) : real(1.0);
    };

    const auto divergence = [&](int k, int j, int i) {
        return ((hx(k, j, i + 1) - hx(k, j, i)) / dx
               -(cos_center(j + 1) * he(k, j + 1, i) - cos_center(j) * he(k, j, i)) / dy) / (radius * cos_face(j));
    };

    const Real tolerance = sizeof(Real) == sizeof(double) ? real(2e-11) : real(3e-4);
    int failures = 0;

    initialize(0);
    launch();
    Kokkos::fence();

#if defined(KOKKOS_ENABLE_CUDA)
    // Explicit preparation above executes the same kernel outside capture.
    // It does not substitute for the changed-input graph checks below.
    Graph graph;
    const auto stream = Kokkos::Cuda().cuda_stream();

    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    cuda_check(cudaStreamEndCapture(stream, &graph.graph));
    cuda_check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
#endif

    for (int step = 0; step < 3; ++step) {
        initialize(step);
        launch();
        Kokkos::fence();

        auto direct = Kokkos::create_mirror(result);
        auto direct_zeta = Kokkos::create_mirror(zeta);
        Kokkos::deep_copy(direct, result);
        Kokkos::deep_copy(direct_zeta, zeta);

        Kokkos::deep_copy(result, sentinel);
        Kokkos::deep_copy(zeta, hz);

#if defined(KOKKOS_ENABLE_CUDA)
        cuda_check(cudaGraphLaunch(graph.executable, stream));
        cuda_check(cudaStreamSynchronize(stream));
#else
        launch();
        Kokkos::fence();
#endif

        auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
        auto actual_zeta = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), zeta);
        auto ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xi);
        auto ae = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eta);
        auto aw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        auto ap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), profiles);

        bool exact = true, sources = true, untouched = true, finite = true;
        Real error = real(0.0);

        const auto check = [&](Real value, Real expected, Real scale) {
            if (!std::isfinite(value) || !std::isfinite(expected)) finite = false;
            else error = std::max(error, std::abs(value - expected) / std::max(scale, std::abs(expected)));
        };

        for (int k = 0; k < nz; ++k) {
            for (int p = 0; p < 5; ++p) sources = sources && ap(p, k) == hp(p, k);

            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    sources = sources && ax(k, j, i) == hx(k, j, i) && ae(k, j, i) == he(k, j, i) && aw(k, j, i) == hw(k, j, i);
                    exact = exact && actual_zeta(k, j, i) == direct_zeta(k, j, i);

                    const bool interior = j >= h && j < ny - h && i >= h && i < nx - h;

                    for (int p = 0; p < 8; ++p) {
                        exact = exact && actual(p, k, j, i) == direct(p, k, j, i);
                        if (!interior || k < bottom || k > top) untouched = untouched && actual(p, k, j, i) == sentinel;
                    }

                    if (!interior || k < bottom) {
                        untouched = untouched && actual_zeta(k, j, i) == hz(k, j, i);
                        continue;
                    }

                    Real expected_zeta = hz(top, j, i);

                    if (k <= top) {
                        for (int level = top - 1; level >= k; --level) {
                            expected_zeta += hp(4, level) * divergence(level, j, i);
                        }
                    } else {
                        expected_zeta -= hp(4, top) * divergence(top, j, i);
                    }

                    check(actual_zeta(k, j, i), expected_zeta, real(1.0));

                    if (k == top) untouched = untouched && actual_zeta(k, j, i) == hz(k, j, i);
                    if (k > top) continue;

                    const Real ct = cos_center(j), cv = cos_face(j), cs = cos_face(j - 1);
                    const Real jacobian = radius * radius * ct;
                    const Real east = real(1.0) / (ct * dx * dx);
                    const Real north = cv / (dy * dy);
                    const Real south_coefficient = cs / (dy * dy);

                    const Real lower = -jacobian * hp(3, k) * hp(2, k) * inverse_dz * inverse_dz / hp(0, k);
                    const Real upper = -jacobian * hp(3, k) * hp(2, k + 1) * inverse_dz * inverse_dz / hp(0, k + 1);
                    const Real diagonal = (shift + real(2.0) * east + north + south_coefficient) / hp(1, k) - lower - upper;

                    const Real rhs = -radius * ((he(k, j, i) - he(k, j, i - 1)) / dx + (cv * hx(k, j, i) - cs * hx(k, j - 1, i)) / dy);
                    const Real neighbors = east * (hw(k, j, i + 1) + hw(k, j, i - 1)) + north * hw(k, j + 1, i) + south_coefficient * hw(k, j - 1, i);
                    const Real jacobi = (rhs + neighbors - lower * hp(1, k - 1) * hw(k - 1, j, i) - upper * hp(1, k + 1) * hw(k + 1, j, i)) / (diagonal * hp(1, k) - shift);

                    const Real expected[8] = {
                        rhs,
                        divergence(k, j, i),
                        lower,
                        diagonal,
                        upper,
                        neighbors,
                        shift * hw(k, j, i) + rhs + neighbors,
                        jacobi
                    };

                    for (int p = 0; p < 8; ++p) {
                        const Real scale = p == 0 ? radius : (p == 1 ? real(0.1) / radius : real(1.0));
                        check(actual(p, k, j, i), expected[p], scale);
                    }
                }
            }
        }

        const bool pass = exact && sources && untouched && finite && error <= tolerance;

        std::printf("%s R=%.0f %s stretched=%d step=%d exact=%d sources=%d untouched=%d error=%.3e %s\n",
            geometry.name(), double(radius), layout_name, int(stretched), step,
            int(exact), int(sources), int(untouched), double(error), pass ? "PASS" : "FAIL");

        failures += !pass;
    }

    Kokkos::fence();
    return failures;
}

int run() {
    HorizontalDomainLayout layout;
    layout.global_nx = layout.local_physical_nx = 8;
    layout.global_ny = layout.local_physical_ny = 8;
    layout.halo = 2;

    int failures = 0;

    const auto exercise = [&](const HorizontalGeometry& geometry, Real radius, Real south) {
        for (bool stretched : {false, true}) {
            failures += run_case<Kokkos::LayoutLeft>(geometry, radius, south, "LayoutLeft", stretched);
            failures += run_case<Kokkos::LayoutRight>(geometry, radius, south, "LayoutRight", stretched);
        }
    };

    CartesianGeometry cartesian(layout, real(2.0), real(3.0));
    exercise(cartesian, real(1.0), real(0.0));

    for (Real radius : {real(8.0), real(6371220.0)}) {
        RegularLatLonGeometry rll(layout, real(0.08), real(0.04), real(0.0), real(-0.3), radius);
        exercise(rll, radius, real(-0.3));
    }

    return failures;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int failures = 0;

    Kokkos::initialize(argc, argv);

    try {
        if (ranks != 1) throw std::runtime_error("This local stencil test requires one MPI rank.");
        failures = run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test_vertical_wind_diagnostic: %s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    Kokkos::finalize();
    MPI_Finalize();

    return failures == 0 ? 0 : 1;
}
