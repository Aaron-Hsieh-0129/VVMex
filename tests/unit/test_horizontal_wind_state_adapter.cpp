#include "core/Grid.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindStateAdapter.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::HorizontalEllipticSolver;
using VVM::Dynamics::HorizontalWindStateAdapter;

constexpr int nz = 6;
constexpr int top = 4;
const Real sentinel = real(-12345.0);

[[noreturn]] void fatal(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    std::fflush(stderr);

    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

template<std::size_t D>
void append(std::vector<Real>& values, const Field<D>& field) {
    const auto h = field.get_host_data();

    if constexpr (D == 0) {
        values.push_back(h());
    } else if constexpr (D == 1) {
        for (std::size_t i = 0; i < h.extent(0); ++i) values.push_back(h(i));
    } else if constexpr (D == 2) {
        for (std::size_t j = 0; j < h.extent(0); ++j) {
            for (std::size_t i = 0; i < h.extent(1); ++i) values.push_back(h(j, i));
        }
    } else if constexpr (D == 3) {
        for (std::size_t k = 0; k < h.extent(0); ++k) {
            for (std::size_t j = 0; j < h.extent(1); ++j) {
                for (std::size_t i = 0; i < h.extent(2); ++i) values.push_back(h(k, j, i));
            }
        }
    }
}

bool same(const std::vector<Real>& a, const std::vector<Real>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Real)) == 0);
}

struct Inputs {
    Field<2> rhs_psi, rhs_chi, psi, chi, previous_psi, previous_chi;
    Field<3> u, v, w;
    Field<1> flex, spacing;
    Field<0> circulation;

    Inputs(int ny, int nx)
        : rhs_psi("rhs_psi", {ny, nx}), rhs_chi("rhs_chi", {ny, nx}), psi("psi", {ny, nx}), chi("chi", {ny, nx}),
          previous_psi("previous_psi", {ny, nx}), previous_chi("previous_chi", {ny, nx}),
          u("source_u", {nz, ny, nx}), v("source_v", {nz, ny, nx}), w("w", {nz, ny, nx}),
          flex("flex", {nz - 1}), spacing("spacing", {nz - 1}), circulation("circulation", {}) {}

    void initialize(const Grid& grid, int replay) {
        const Real factor = replay == 3 ? real(0.0) : static_cast<Real>(replay + 1);

        auto fh = flex.get_host_data();
        for (int k = 0; k < nz - 1; ++k) {
            fh(k) = real(100.0) / (replay == 0 ? real(100.0) : real(40.0 * (k + 1) + 10.0 * replay));
        }

        Kokkos::deep_copy(flex.get_mutable_device_data(), fh);
        HorizontalWindStateAdapter::initialize_spacing(real(100.0), flex, spacing);

        const auto ds = spacing.get_host_data();
        std::array<Real, nz> z = {};
        for (int k = 1; k < nz; ++k) z[k] = z[k - 1] + ds(k - 1);

        const Real radius = grid.geometry().kind() == GeometryKind::RegularLatLon
            ? static_cast<const RegularLatLonGeometry&>(grid.geometry()).radius() : real(1.0);

        Kokkos::deep_copy(circulation.get_mutable_device_data(), factor * real(30.0) * radius);

        std::array<Field<2>*, 6> fields = {&rhs_psi, &rhs_chi, &psi, &chi, &previous_psi, &previous_chi};
        std::array<Field<2>::HostMirrorType, 6> h;

        for (int n = 0; n < 6; ++n) h[n] = fields[n]->get_host_data();

        auto uh = u.get_host_data();
        auto vh = v.get_host_data();
        auto wh = w.get_host_data();
        const Real pi2 = real(2.0) * std::acos(real(-1.0));

        for (int j = 0; j < grid.get_local_total_points_y(); ++j) {
            const Real y = pi2 * (grid.get_local_physical_start_y() + j - grid.get_halo_cells() + real(0.5)) / grid.get_global_points_y();

            for (int i = 0; i < grid.get_local_total_points_x(); ++i) {
                const Real x = pi2 * (grid.get_local_physical_start_x() + i - grid.get_halo_cells() + real(0.5)) / grid.get_global_points_x();

                h[0](j, i) = factor * real(1e-10) * std::sin(x);
                h[1](j, i) = factor * real(-6e-11) * std::cos(real(2.0) * x);
                h[2](j, i) = factor * real(1000.0) * std::sin(x) * std::cos(y);
                h[3](j, i) = factor * real(700.0) * std::cos(x) * std::sin(y);
                h[4](j, i) = real(0.8) * h[2](j, i);
                h[5](j, i) = real(0.6) * h[3](j, i);

                const Real a = factor * real(1e-3) * (real(1.0) + real(0.2) * std::cos(x));
                const Real b = factor * real(-8e-4) * (real(1.0) + real(0.1) * std::sin(y));

                for (int k = 0; k < nz; ++k) {
                    uh(k, j, i) = factor * (real(3.0) + real(0.2) * std::sin(x)) + a * z[k];
                    vh(k, j, i) = factor * (real(-2.0) + real(0.3) * std::cos(y)) + b * z[k];
                    wh(k, j, i) = factor * real(0.2) * (std::sin(x) + real(0.25) * std::cos(y)) * (real(1.0) + real(0.1) * k);
                }
            }
        }

        for (int n = 0; n < 6; ++n) Kokkos::deep_copy(fields[n]->get_mutable_device_data(), h[n]);

        Kokkos::deep_copy(u.get_mutable_device_data(), uh);
        Kokkos::deep_copy(v.get_mutable_device_data(), vh);
        Kokkos::deep_copy(w.get_mutable_device_data(), wh);
    }

    std::vector<Real> values() const {
        std::vector<Real> result;

        for (const auto* f : {&rhs_psi, &rhs_chi, &psi, &chi, &previous_psi, &previous_chi}) append(result, *f);
        for (const auto* f : {&u, &v, &w}) append(result, *f);

        append(result, flex);
        append(result, spacing);
        append(result, circulation);

        return result;
    }
};

struct Outputs {
    Field<2> psi, chi;
    Field<3> xi, eta, u, v;

    Outputs(int ny, int nx)
        : psi("out_psi", {ny, nx}), chi("out_chi", {ny, nx}), xi("xi", {nz, ny, nx}),
          eta("eta", {nz, ny, nx}), u("u", {nz, ny, nx}), v("v", {nz, ny, nx}) {}

    void reset() {
        Kokkos::deep_copy(psi.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(chi.get_mutable_device_data(), sentinel);

        for (auto* f : {&xi, &eta, &u, &v}) Kokkos::deep_copy(f->get_mutable_device_data(), sentinel);
    }

    std::vector<Real> values() const {
        std::vector<Real> result;

        append(result, psi);
        append(result, chi);

        for (const auto* f : {&xi, &eta, &u, &v}) append(result, *f);

        return result;
    }
};

// Prescribed test datum, not a proposed production circulation evolution law.
void add_circulation(const Grid& grid, const Field<0>& circulation, Field<3>& u) {
    const auto inverse_h1 = grid.geometry().device_view(VVM::Core::Geometry::HorizontalLocation::U).physical_to_contravariant.a11;
    const auto c = circulation.get_device_data();
    const auto ud = u.get_mutable_device_data();
    const int h = grid.get_halo_cells();

    Kokkos::parallel_for("PrescribeTestCirculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {grid.get_local_total_points_y() - h, grid.get_local_total_points_x() - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            ud(top, j, i) += c() * inverse_h1(j, i);
        });
}

bool check_math(const Grid& grid, const Inputs& input, const Outputs& output, int bottom, int replay) {
    const auto p = output.psi.get_host_data();
    const auto c = output.chi.get_host_data();
    const auto u = output.u.get_host_data();
    const auto v = output.v.get_host_data();
    const auto xi = output.xi.get_host_data();
    const auto eta = output.eta.get_host_data();
    const auto w = input.w.get_host_data();
    const auto ds = input.spacing.get_host_data();
    const Real circulation = input.circulation.get_host_data()();

    std::array<double, nz> z = {};
    for (int k = 1; k < nz; ++k) z[k] = z[k - 1] + ds(k - 1);

    const bool spherical = grid.geometry().kind() == GeometryKind::RegularLatLon;
    const auto* rll = spherical ? &static_cast<const RegularLatLonGeometry&>(grid.geometry()) : nullptr;
    const double radius = spherical ? rll->radius() : 1.0;
    const double dq1 = grid.geometry().dq1();
    const double dq2 = grid.geometry().dq2();
    const double pi2 = 2.0 * std::acos(-1.0);
    const double factor = replay == 3 ? 0.0 : replay + 1.0;
    const int h = grid.get_halo_cells();

    std::array<double, 4> errors = {}, scales = {};
    bool valid = true;

    const auto compare = [&](int n, double actual, double expected) {
        valid = valid && std::isfinite(actual) && std::isfinite(expected);
        errors[n] = std::max(errors[n], std::abs(actual - expected));
        scales[n] = std::max(scales[n], std::abs(expected));
    };

    for (int j = 0; j < grid.get_local_total_points_y(); ++j) {
        const int global_j = grid.get_local_physical_start_y() + j - h;
        const double y = pi2 * (global_j + 0.5) / grid.get_global_points_y();
        const double cos_u = spherical ? std::cos(rll->latitude_south_edge() + (global_j + 0.5) * dq2) : 1.0;
        const double cos_v = spherical ? std::cos(rll->latitude_south_edge() + (global_j + 1.0) * dq2) : 1.0;
        const double h1 = radius * cos_u;
        const double h2 = radius;

        for (int i = 0; i < grid.get_local_total_points_x(); ++i) {
            const double x = pi2 * (grid.get_local_physical_start_x() + i - h + 0.5) / grid.get_global_points_x();
            const double a = factor * 1e-3 * (1.0 + 0.2 * std::cos(x));
            const double b = factor * -8e-4 * (1.0 + 0.1 * std::sin(y));

            const bool physical = j >= h && j < grid.get_local_total_points_y() - h
                && i >= h && i < grid.get_local_total_points_x() - h;

            double utop = 0.0;
            double vtop = 0.0;

            if (physical) {
                utop = (-(p(j, i) - p(j - 1, i)) * cos_u / dq2 + (c(j, i + 1) - c(j, i)) / dq1 + circulation) / h1;
                vtop = ((p(j, i) - p(j, i - 1)) / (cos_v * dq1) + (c(j + 1, i) - c(j, i)) / dq2) / h2;
            }

            for (int k = 0; k < nz; ++k) {
                if (physical && k >= bottom && k <= top) {
                    compare(0, u(k, j, i), utop - a * (z[top] - z[k]));
                    compare(1, v(k, j, i), vtop - b * (z[top] - z[k]));
                } else {
                    valid = valid && u(k, j, i) == sentinel && v(k, j, i) == sentinel;
                }

                if (physical && k >= bottom && k < top) {
                    compare(2, xi(k, j, i), (w(k, j + 1, i) - w(k, j, i)) / (h2 * dq2) - b);
                    compare(3, eta(k, j, i), (w(k, j, i + 1) - w(k, j, i)) / (h1 * dq1) - a);
                } else {
                    valid = valid && xi(k, j, i) == sentinel && eta(k, j, i) == sentinel;
                }
            }
        }
    }

    const double tolerance = sizeof(Real) == sizeof(double) ? 5e-10 : 5e-4;

    for (int n = 0; n < 4; ++n) {
        valid = valid && errors[n] <= tolerance * std::max(1e-12, scales[n]);
    }

    return valid;
}

#if defined(ENABLE_NCCL)

void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) fatal(cudaGetErrorString(result));
}

void nccl_check(ncclResult_t result) {
    if (result != ncclSuccess) fatal(ncclGetErrorString(result));
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

int run_case(const Grid& grid, HaloExchanger& halo, int iterations, int bottom) {
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Inputs input(ny, nx);
    Outputs direct(ny, nx), replayed(ny, nx);
    HorizontalEllipticSolver first(grid, halo), second(grid, halo);
    HorizontalWindStateAdapter adapter(grid.geometry());

    HorizontalEllipticSolver::Options options;
    options.iterations = iterations;
    options.diagonal_shift = real(0.25);
    options.refresh_initial_halos = true;

#if defined(ENABLE_NCCL)
    Graph graph;
    const auto stream = Kokkos::Cuda().cuda_stream();
    const char* execution = "cuda_graph";
#else
    const char* execution = "direct_repeat";
#endif

    try {
        const auto execute = [&](HorizontalEllipticSolver& solver, Outputs& out) {
            if (bottom < top) {
                adapter.diagnose_vorticity(input.u, input.v, input.w, input.spacing, out.xi, out.eta, bottom, top - 1);
            }

            solver.make_extrapolated_guess(input.psi, input.previous_psi, out.psi);
            solver.make_extrapolated_guess(input.chi, input.previous_chi, out.chi);
            solver.solve_at_z_and_t(input.rhs_psi, out.psi, input.rhs_chi, out.chi, options);

            adapter.reconstruct_top(out.psi, out.chi, out.u, out.v, top);
            add_circulation(grid, input.circulation, out.u);
            adapter.integrate_from_top(input.w, out.xi, out.eta, input.spacing, out.u, out.v, bottom, top);
        };

        input.initialize(grid, 0);
        direct.reset();
        replayed.reset();
        HorizontalWindStateAdapter::prepare_execution();

        // Explicitly prepare the solver, transport, and test-kernel paths.
        execute(first, direct);
        Kokkos::fence();

        execute(second, replayed);
        Kokkos::fence();

        MPI_Barrier(grid.get_comm());

#if defined(ENABLE_NCCL)
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        execute(second, replayed);
        cuda_check(cudaStreamEndCapture(stream, &graph.graph));

        if (!graph.graph) fatal("Capture returned no graph.");

        cuda_check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
#endif

        int failures = 0;
        std::vector<Real> previous;

        for (int replay = 0; replay < 4; ++replay) {
            input.initialize(grid, replay);
            direct.reset();
            replayed.reset();
            Kokkos::fence();

            const auto before = input.values();

            MPI_Barrier(grid.get_comm());
            execute(first, direct);
            Kokkos::fence();

            bool preserved = same(before, input.values());

            MPI_Barrier(grid.get_comm());

#if defined(ENABLE_NCCL)
            cuda_check(cudaGraphLaunch(graph.executable, stream));
            cuda_check(cudaStreamSynchronize(stream));
#else
            execute(second, replayed);
            Kokkos::fence();
#endif

            preserved = preserved && same(before, input.values());

            const auto expected = direct.values();
            const auto actual = replayed.values();
            const bool finite = std::all_of(actual.begin(), actual.end(), [](Real value) {
                return std::isfinite(value);
            });

            std::array<int, 5> local = {
                same(expected, actual),
                preserved,
                check_math(grid, input, direct, bottom, replay) && check_math(grid, input, replayed, bottom, replay),
                replay == 0 || !same(previous, actual),
                finite
            };

            std::array<int, 5> global = {};
            MPI_Allreduce(local.data(), global.data(), 5, MPI_INT, MPI_MIN, grid.get_comm());

            const bool pass = std::all_of(global.begin(), global.end(), [](int value) {
                return value == 1;
            });

            if (grid.get_mpi_rank() == 0) {
                std::printf("ranks=%d execution=%s iterations=%d bottom=%d replay=%d exact=%d inputs=%d math_regions=%d updated=%d finite=%d %s\n",
                    grid.get_mpi_size(), execution, iterations, bottom, replay,
                    global[0], global[1], global[2], global[3], global[4], pass ? "PASS" : "FAIL");
            }

            failures += !pass;
            previous = actual;
        }

        Kokkos::fence();
        MPI_Barrier(grid.get_comm());

        return failures;
    } catch (const std::exception& error) {
        fatal(error.what());
    }
}

int run(const Grid& grid, HaloExchanger& halo) {
    int failures = 0;

    for (int iterations : {1, 4}) {
        for (int bottom : {0, 2, top}) {
            failures += run_case(grid, halo, iterations, bottom);
        }
    }

    return failures;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int failures = 0;

    try {
        Kokkos::initialize(argc, argv);

        if (argc != 2) fatal("Expected one configuration path.");
        if (ranks != 1 && ranks != 2 && ranks != 4) fatal("Use 1, 2, or 4 ranks.");

        {
            VVM::Utils::ConfigurationManager config(argv[1]);
            Grid grid(config);

#if defined(ENABLE_NCCL)
            ncclUniqueId id;
            if (rank == 0) nccl_check(ncclGetUniqueId(&id));

            MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, grid.get_comm());

            ncclComm_t comm = nullptr;
            nccl_check(ncclCommInitRank(&comm, ranks, id, rank));

            {
                HaloExchanger halo(config, grid, comm, Kokkos::Cuda().cuda_stream());

                try {
                    failures = run(grid, halo);
                } catch (const std::exception& error) {
                    fatal(error.what());
                }

                Kokkos::fence();
            }

            nccl_check(ncclCommDestroy(comm));
#else
            HaloExchanger halo(grid);
            failures = run(grid, halo);
#endif
        }

        Kokkos::finalize();
    } catch (const std::exception& error) {
        fatal(error.what());
    }

    int global = 0;
    MPI_Allreduce(&failures, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return global == 0 ? 0 : 1;
}
