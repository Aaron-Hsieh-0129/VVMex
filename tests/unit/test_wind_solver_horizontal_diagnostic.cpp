#include "dynamics/solvers/WindSolver.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(ENABLE_NCCL)
#include <nccl.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Dynamics::WindSolver;
using VVM::Dynamics::HorizontalEllipticSolver;

constexpr int nz = 6;
constexpr int top = 4;
constexpr int bottom = 1;
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
void append(std::vector<Real>& out, const Field<D>& field) {
    const auto d = field.get_host_data();

    if constexpr (D == 0) {
        out.push_back(d());
    } else if constexpr (D == 1) {
        for (std::size_t i = 0; i < d.extent(0); ++i) out.push_back(d(i));
    } else if constexpr (D == 2) {
        for (std::size_t j = 0; j < d.extent(0); ++j) {
            for (std::size_t i = 0; i < d.extent(1); ++i) out.push_back(d(j, i));
        }
    } else if constexpr (D == 3) {
        for (std::size_t k = 0; k < d.extent(0); ++k) {
            for (std::size_t j = 0; j < d.extent(1); ++j) {
                for (std::size_t i = 0; i < d.extent(2); ++i) out.push_back(d(k, j, i));
            }
        }
    }
}

bool same(const std::vector<Real>& a, const std::vector<Real>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Real)) == 0);
}

struct Sources {
    Field<3> zeta, w, xi, eta;
    Field<1> rho, rho_up, flex, spacing;
    Field<0> increment;

    Sources(int ny, int nx)
        : zeta("zeta", {nz, ny, nx}), w("w", {nz, ny, nx}), xi("xi", {nz, ny, nx}), eta("eta", {nz, ny, nx}),
          rho("rho", {nz}), rho_up("rho_up", {nz}), flex("flex", {nz}), spacing("spacing", {nz - 1}), increment("increment", {}) {}

    void initialize(const Grid& grid, int step) {
        const Real factor = step == 3 ? real(0.0) : static_cast<Real>(step + 1);

        auto zh = zeta.get_host_data();
        auto wh = w.get_host_data();
        auto xh = xi.get_host_data();
        auto eh = eta.get_host_data();
        const Real pi2 = real(2.0) * std::acos(real(-1.0));

        for (int j = 0; j < grid.get_local_total_points_y(); ++j) {
            for (int i = 0; i < grid.get_local_total_points_x(); ++i) {
                const Real x = pi2 * (grid.get_local_physical_start_x() + i - grid.get_halo_cells() + real(0.5)) / grid.get_global_points_x();

                for (int k = 0; k < nz; ++k) {
                    zh(k, j, i) = factor * real(1e-10) * std::sin(x);
                    wh(k, j, i) = k == top ? real(0.0) : factor * real(1e-5) * std::cos(x) * static_cast<Real>(k + 1);
                    xh(k, j, i) = factor * real(1e-4) * std::sin(x);
                    eh(k, j, i) = factor * real(-2e-4) * std::cos(x);
                }
            }
        }

        Kokkos::deep_copy(zeta.get_mutable_device_data(), zh);
        Kokkos::deep_copy(w.get_mutable_device_data(), wh);
        Kokkos::deep_copy(xi.get_mutable_device_data(), xh);
        Kokkos::deep_copy(eta.get_mutable_device_data(), eh);

        Kokkos::deep_copy(rho.get_mutable_device_data(), real(1.0));
        Kokkos::deep_copy(rho_up.get_mutable_device_data(), real(0.9));
        Kokkos::deep_copy(flex.get_mutable_device_data(), real(1.25));

        auto ds = spacing.get_host_data();
        for (int k = 0; k < nz - 1; ++k) {
            ds(k) = step == 0 ? real(100.0) : real(40.0 * (k + 1));
        }
        Kokkos::deep_copy(spacing.get_mutable_device_data(), ds);

        Real radius = real(1.0);
        if (grid.geometry().kind() == VVM::Core::Geometry::GeometryKind::RegularLatLon) {
            radius = static_cast<const VVM::Core::Geometry::RegularLatLonGeometry&>(grid.geometry()).radius();
        }

        Kokkos::deep_copy(increment.get_mutable_device_data(), factor * radius * real(30.0));
    }

    std::vector<Real> values() const {
        std::vector<Real> result;

        for (const auto* f : {&zeta, &w, &xi, &eta}) append(result, *f);
        for (const auto* f : {&rho, &rho_up, &flex, &spacing}) append(result, *f);
        append(result, increment);

        return result;
    }
};

struct StateFields {
    Field<2> psi, previous_psi, chi, previous_chi;
    Field<3> u, v;
    Field<2> rhs_psi, rhs_chi, solution_psi, solution_chi;

    StateFields(int ny, int nx)
        : psi("psi", {ny, nx}), previous_psi("previous_psi", {ny, nx}),
          chi("chi", {ny, nx}), previous_chi("previous_chi", {ny, nx}),
          u("u", {nz, ny, nx}), v("v", {nz, ny, nx}),
          rhs_psi("rhs_psi", {ny, nx}), rhs_chi("rhs_chi", {ny, nx}),
          solution_psi("solution_psi", {ny, nx}), solution_chi("solution_chi", {ny, nx}) {}

    void reset_history() {
        Kokkos::deep_copy(psi.get_mutable_device_data(), real(3.0));
        Kokkos::deep_copy(previous_psi.get_mutable_device_data(), real(1.0));
        Kokkos::deep_copy(chi.get_mutable_device_data(), real(-2.0));
        Kokkos::deep_copy(previous_chi.get_mutable_device_data(), real(-1.0));
    }

    void reset_outputs() {
        Kokkos::deep_copy(u.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(v.get_mutable_device_data(), sentinel);
    }

    WindSolver::HorizontalDiagnosticFields bind(const Sources& s) {
        return {
            psi, previous_psi, chi, previous_chi,
            s.zeta, s.w, s.xi, s.eta, u, v,
            s.rho, s.rho_up, s.flex, s.spacing, s.increment
        };
    }

    WindSolver::HorizontalDiagnosticWorkspace workspace() {
        return {rhs_psi, rhs_chi, solution_psi, solution_chi};
    }

    std::vector<Real> history() const {
        std::vector<Real> result;
        append(result, psi);
        append(result, chi);
        return result;
    }

    std::vector<Real> values() const {
        std::vector<Real> result;

        for (const auto* f : {&psi, &previous_psi, &chi, &previous_chi, &rhs_psi, &rhs_chi, &solution_psi, &solution_chi}) {
            append(result, *f);
        }

        append(result, u);
        append(result, v);
        return result;
    }
};

bool check(const Grid& grid, const Sources& source, const StateFields& state, const std::vector<Real>& old) {
    const auto p = state.psi.get_host_data();
    const auto pp = state.previous_psi.get_host_data();
    const auto c = state.chi.get_host_data();
    const auto cp = state.previous_chi.get_host_data();
    const auto rp = state.rhs_psi.get_host_data();
    const auto rc = state.rhs_chi.get_host_data();
    const auto zeta = source.zeta.get_host_data();
    const auto w = source.w.get_host_data();
    const auto xi = source.xi.get_host_data();
    const auto eta = source.eta.get_host_data();
    const auto ds = source.spacing.get_host_data();
    const auto u = state.u.get_host_data();
    const auto v = state.v.get_host_data();
    const double increment = source.increment.get_host_data()();

    const bool spherical = grid.geometry().kind() == VVM::Core::Geometry::GeometryKind::RegularLatLon;
    const auto* rll = spherical ? &static_cast<const VVM::Core::Geometry::RegularLatLonGeometry&>(grid.geometry()) : nullptr;
    const double radius = spherical ? rll->radius() : 1.0;
    const double dq1 = grid.geometry().dq1();
    const double dq2 = grid.geometry().dq2();
    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int h = grid.get_halo_cells();
    const std::size_t cells = static_cast<std::size_t>(ny) * nx;
    const double tolerance = sizeof(Real) == sizeof(double) ? 5e-10 : 5e-4;

    std::array<double, 3> errors = {}, scales = {};
    bool valid = true;

    const auto compare = [&](int n, double actual, double expected) {
        valid = valid && std::isfinite(actual) && std::isfinite(expected);
        errors[n] = std::max(errors[n], std::abs(actual - expected));
        scales[n] = std::max(scales[n], std::abs(expected));
    };

    for (int j = 0; j < ny; ++j) {
        const int gj = grid.get_local_physical_start_y() + j - h;
        const double cu = spherical ? std::cos(rll->latitude_south_edge() + (gj + 0.5) * dq2) : 1.0;
        const double cv = spherical ? std::cos(rll->latitude_south_edge() + (gj + 1.0) * dq2) : 1.0;

        for (int i = 0; i < nx; ++i) {
            const bool physical = j >= h && j < ny - h && i >= h && i < nx - h;

            valid = valid && rp(j, i) == zeta(top, j, i);
            compare(0, rc(j, i), 1.25 * static_cast<double>(real(0.9)) * w(top - 1, j, i) * static_cast<double>(real(0.01)));

            if (physical) {
                const std::size_t index = static_cast<std::size_t>(j) * nx + i;
                valid = valid && pp(j, i) == old[index] && cp(j, i) == old[cells + index];

                double eu = (-(p(j, i) - p(j - 1, i)) * cu / dq2 + (c(j, i + 1) - c(j, i)) / dq1 + increment) / (radius * cu);
                double ev = ((p(j, i) - p(j, i - 1)) / (cv * dq1) + (c(j + 1, i) - c(j, i)) / dq2) / radius;

                compare(1, u(top, j, i), eu);
                compare(2, v(top, j, i), ev);

                for (int k = top - 1; k >= bottom; --k) {
                    eu -= ((w(k, j, i + 1) - w(k, j, i)) / (dq1 * radius * cu) - eta(k, j, i)) * ds(k);
                    ev -= ((w(k, j + 1, i) - w(k, j, i)) / (dq2 * radius) - xi(k, j, i)) * ds(k);

                    compare(1, u(k, j, i), eu);
                    compare(2, v(k, j, i), ev);
                }
            }

            for (int k = 0; k < nz; ++k) {
                if (!physical || k < bottom || k > top) {
                    valid = valid && u(k, j, i) == sentinel && v(k, j, i) == sentinel;
                }
            }
        }
    }

    for (int n = 0; n < 3; ++n) {
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

int run_case(const Grid& grid, HaloExchanger& halo, int iterations) {
    Sources source(grid.get_local_total_points_y(), grid.get_local_total_points_x());
    StateFields direct(grid.get_local_total_points_y(), grid.get_local_total_points_x());
    StateFields replayed(grid.get_local_total_points_y(), grid.get_local_total_points_x());
    HorizontalEllipticSolver first(grid, halo), second(grid, halo);

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
        const auto execute = [&](HorizontalEllipticSolver& solver, StateFields& state) {
            WindSolver::diagnose_horizontal_wind(
                grid, halo, solver, state.bind(source), state.workspace(),
                options, real(0.01), bottom, top);
        };

        source.initialize(grid, 0);
        direct.reset_history();
        replayed.reset_history();

        WindSolver::prepare_horizontal_diagnostic_execution();

        // Explicit preparation of existing solver and communication kernels.
        execute(first, direct);
        Kokkos::fence();
        execute(second, replayed);
        Kokkos::fence();

        direct.reset_history();
        replayed.reset_history();
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

        for (int step = 0; step < 4; ++step) {
            source.initialize(grid, step);

            // Reset only output wind. Potential histories intentionally advance.
            direct.reset_outputs();
            replayed.reset_outputs();
            Kokkos::fence();

            const auto before = source.values();
            const auto old_direct = direct.history();
            const auto old_replayed = replayed.history();

            MPI_Barrier(grid.get_comm());
            execute(first, direct);
            Kokkos::fence();

            bool preserved = same(before, source.values());

            MPI_Barrier(grid.get_comm());

#if defined(ENABLE_NCCL)
            cuda_check(cudaGraphLaunch(graph.executable, stream));
            cuda_check(cudaStreamSynchronize(stream));
#else
            execute(second, replayed);
            Kokkos::fence();
#endif

            preserved = preserved && same(before, source.values());

            const auto a = direct.values();
            const auto b = replayed.values();

            std::array<int, 4> local = {
                same(a, b),
                preserved,
                check(grid, source, direct, old_direct) && check(grid, source, replayed, old_replayed),
                std::all_of(b.begin(), b.end(), [](Real value) { return std::isfinite(value); })
            };

            std::array<int, 4> global = {};
            MPI_Allreduce(local.data(), global.data(), 4, MPI_INT, MPI_MIN, grid.get_comm());

            const bool pass = std::all_of(global.begin(), global.end(), [](int value) {
                return value == 1;
            });

            failures += !pass;

            if (grid.get_mpi_rank() == 0) {
                std::printf("ranks=%d execution=%s iterations=%d step=%d exact=%d sources=%d rhs_history_wind=%d finite=%d %s\n",
                    grid.get_mpi_size(), execution, iterations, step,
                    global[0], global[1], global[2], global[3], pass ? "PASS" : "FAIL");
            }
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
        failures += run_case(grid, halo, iterations);
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

        if (argc != 2 || (ranks != 1 && ranks != 2 && ranks != 4)) {
            fatal("Provide a config and use 1, 2, or 4 ranks.");
        }

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
