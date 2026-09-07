#include "dynamics/solvers/VerticalEllipticSolver.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "utils/ConfigurationManager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>

namespace {
using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Dynamics::VerticalEllipticSolver;

[[noreturn]] void fatal(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::abort();
}

#if defined(ENABLE_NCCL)
void cuda_check(cudaError_t error) { if (error != cudaSuccess) fatal(cudaGetErrorString(error)); }
void nccl_check(ncclResult_t error) { if (error != ncclSuccess) fatal(ncclGetErrorString(error)); }
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
#endif

struct Reference {
    int nx, ny, nz, h;
    bool bounded, spherical;
    double dx, dy, radius = 1.0, south = 0.0;
    std::vector<double> w, history, rho, rho_up, mid, up;

    Reference(const Grid& grid, bool stretched)
        : nx(grid.get_global_points_x()), ny(grid.get_global_points_y()), nz(grid.get_local_total_points_z()), h(grid.get_halo_cells()),
          bounded(grid.horizontal_specification().topology.q2 == VVM::Core::HorizontalEdgeTopology::Bounded),
          spherical(grid.geometry().kind() == VVM::Core::Geometry::GeometryKind::RegularLatLon), dx(grid.geometry().dq1()), dy(grid.geometry().dq2()),
          w(nz * ny * nx, 0.0), history(w), rho(nz), rho_up(nz), mid(nz), up(nz) {
        if (spherical) {
            const auto& geometry = dynamic_cast<const VVM::Core::Geometry::RegularLatLonGeometry&>(grid.geometry());
            radius = geometry.radius();
            south = geometry.latitude_south_edge();
        }
        for (int k = 0; k < nz; ++k) {
            rho[k] = Real(stretched ? 1.4 - .02 * k : 1.0);
            rho_up[k] = Real(stretched ? 1.39 - .02 * k : 1.0);
            mid[k] = Real(stretched ? 1.1 + .02 * k : 1.0);
            up[k] = Real(stretched ? .9 + .03 * k : 1.0);
        }
    }

    int index(int k, int j, int i) const {
        i = (i % nx + nx) % nx;
        j = bounded ? std::max(0, std::min(ny - 1, j)) : (j % ny + ny) % ny;
        return (k * ny + j) * nx + i;
    }

    Real source(bool is_eta, int step, int k, int j, int i) const {
        const int n = index(k, j, i);
        i = n % nx;
        j = (n / nx) % ny;
        const double factor = step == 2 ? 0.0 : step + 1.0;
        const double angle = 2.0 * std::acos(-1.0) * i / nx;
        return Real(factor * (is_eta ? .02 * std::cos(angle) - .004 * j : .001 * k + .01 * std::sin(angle) + .003 * j));
    }

    // Independent dense elimination, not the production Thomas implementation.
    static std::vector<double> dense_solve(std::vector<double> a, std::vector<double> b) {
        const int n = int(b.size());
        for (int k = 0; k < n; ++k) {
            int pivot = k;
            for (int j = k + 1; j < n; ++j) if (std::abs(a[j * n + k]) > std::abs(a[pivot * n + k])) pivot = j;
            for (int i = k; i < n; ++i) std::swap(a[k * n + i], a[pivot * n + i]);
            std::swap(b[k], b[pivot]);
            if (!std::isfinite(a[k * n + k]) || a[k * n + k] == 0.0) fatal("Invalid dense reference pivot.");
            for (int j = k + 1; j < n; ++j) {
                const double factor = a[j * n + k] / a[k * n + k];
                for (int i = k; i < n; ++i) a[j * n + i] -= factor * a[k * n + i];
                b[j] -= factor * b[k];
            }
        }
        for (int k = n - 1; k >= 0; --k) {
            for (int i = k + 1; i < n; ++i) b[k] -= a[k * n + i] * b[i];
            b[k] /= a[k * n + k];
        }
        return b;
    }

    void advance(int step, int iterations, double shift) {
        const int last = nz - h - 2, levels = last - h + 1;
        auto current = w;
        for (int k = h; k <= last; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
            const int p = index(k, j, i);
            current[p] = 2.0 * w[p] - history[p];
        }
        history = w;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            auto next = current;
            for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
                const double ct = spherical ? std::cos(south + (j + .5) * dy) : 1.0;
                const double cn = spherical ? std::cos(south + (j + 1.0) * dy) : 1.0;
                const double cs = spherical ? std::cos(south + j * dy) : 1.0;
                const double J = radius * radius * ct;
                const double east = 1.0 / (ct * dx * dx), north = cn / (dy * dy), south_weight = cs / (dy * dy);
                std::vector<double> matrix(levels * levels, 0.0), rhs(levels);
                for (int k = h; k <= last; ++k) {
                    const int level = k - h;
                    const double lower = -J * up[k] * mid[k] * .0001 / rho[k];
                    const double upper = -J * up[k] * mid[k + 1] * .0001 / rho[k + 1];
                    matrix[level * levels + level] = (shift + 2.0 * east + north + south_weight) / rho_up[k] - lower - upper;
                    if (level > 0) matrix[level * levels + level - 1] = lower;
                    if (level + 1 < levels) matrix[level * levels + level + 1] = upper;
                    const double forcing = -radius * ((double(source(true, step, k, j, i)) - double(source(true, step, k, j, i - 1))) / dx
                        + (cn * double(source(false, step, k, j, i)) - cs * double(source(false, step, k, j - 1, i))) / dy);
                    rhs[level] = shift * current[index(k, j, i)] + forcing
                        + east * (current[index(k, j, i + 1)] + current[index(k, j, i - 1)])
                        + north * current[index(k, j + 1, i)] + south_weight * current[index(k, j - 1, i)];
                }
                const auto solution = dense_solve(matrix, rhs);
                for (int k = h; k <= last; ++k) next[index(k, j, i)] = solution[k - h] / rho_up[k];
                next[index(h - 1, j, i)] = next[index(last + 1, j, i)] = 0.0;
            }
            current.swap(next);
        }
        w.swap(current);
    }
};

std::vector<Real> snapshot(const Field<3>& field) {
    const auto data = field.get_host_data();
    std::vector<Real> values;
    values.reserve(data.size());
    for (int k = 0; k < int(data.extent(0)); ++k) for (int j = 0; j < int(data.extent(1)); ++j) for (int i = 0; i < int(data.extent(2)); ++i) values.push_back(data(k, j, i));
    return values;
}

int run_case(const Grid& grid, HaloExchanger& halo, int iterations, bool stretched) {
    const int nz = grid.get_local_total_points_z(), ny = grid.get_local_total_points_y(), nx = grid.get_local_total_points_x(), h = grid.get_halo_cells();
    Reference reference(grid, stretched);
    Field<1> rho("rho", {nz}), density("density", {nz}), mid("mid", {nz}), up("up", {nz});
    const auto upload = [](Field<1>& field, const std::vector<double>& values) {
        auto host = Kokkos::create_mirror(field.get_device_data());
        for (int k = 0; k < int(values.size()); ++k) host(k) = Real(values[k]);
        Kokkos::deep_copy(field.get_mutable_device_data(), host);
    };
    upload(rho, reference.rho); upload(density, reference.rho_up); upload(mid, reference.mid); upload(up, reference.up);
    Field<3> xi("xi", {nz, ny, nx}), eta("eta", {nz, ny, nx});
    Field<3> w("w", {nz, ny, nx}), previous("previous", {nz, ny, nx}), replay_w("replay_w", {nz, ny, nx}), replay_previous("replay_previous", {nz, ny, nx});
    const Real shift = stretched ? real(.25) : real(0.0);
    VerticalEllipticSolver direct(grid, halo, rho, density, mid, up, real(.01), shift);
    VerticalEllipticSolver replay(grid, halo, rho, density, mid, up, real(.01), shift);
    const auto sources = [&](int step) {
        auto x = Kokkos::create_mirror(xi.get_device_data()), e = Kokkos::create_mirror(eta.get_device_data());
        for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
            const int gj = grid.get_local_physical_start_y() + j - h, gi = grid.get_local_physical_start_x() + i - h;
            x(k, j, i) = reference.source(false, step, k, gj, gi);
            e(k, j, i) = reference.source(true, step, k, gj, gi);
        }
        Kokkos::deep_copy(xi.get_mutable_device_data(), x);
        Kokkos::deep_copy(eta.get_mutable_device_data(), e);
    };
    sources(0);
    VerticalEllipticSolver::prepare_execution();
    direct.solve(xi, eta, w, previous, iterations);
    Kokkos::fence();
    replay.solve(xi, eta, replay_w, replay_previous, iterations);
    Kokkos::fence();
    w.set_to_zero(); previous.set_to_zero(); replay_w.set_to_zero(); replay_previous.set_to_zero();
    Kokkos::fence();
#if defined(ENABLE_NCCL)
    Graph graph;
    const auto stream = Kokkos::Cuda().cuda_stream();
    MPI_Barrier(grid.get_comm());
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    replay.solve(xi, eta, replay_w, replay_previous, iterations);
    cuda_check(cudaStreamEndCapture(stream, &graph.graph));
    cuda_check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
#endif
    int failures = 0;
    for (int step = 0; step < 3; ++step) {
        sources(step);
        const auto before_x = snapshot(xi), before_e = snapshot(eta), before_w = snapshot(w);
        MPI_Barrier(grid.get_comm());
        direct.solve(xi, eta, w, previous, iterations);
        Kokkos::fence();
        MPI_Barrier(grid.get_comm());
#if defined(ENABLE_NCCL)
        cuda_check(cudaGraphLaunch(graph.executable, stream));
        cuda_check(cudaStreamSynchronize(stream));
#else
        replay.solve(xi, eta, replay_w, replay_previous, iterations);
        Kokkos::fence();
#endif
        reference.advance(step, iterations, shift);
        const auto actual = w.get_host_data();
        double maximum_error = 0.0, maximum_value = 0.0;
        bool finite = true, surfaces = true;
        for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
            const int gj = grid.get_local_physical_start_y() + j - h, gi = grid.get_local_physical_start_x() + i - h;
            const double expected = reference.w[reference.index(k, gj, gi)];
            finite = finite && std::isfinite(actual(k, j, i)) && std::isfinite(expected);
            maximum_error = std::max(maximum_error, std::abs(double(actual(k, j, i)) - expected));
            maximum_value = std::max(maximum_value, std::abs(expected));
            if (k < h || k > nz - h - 2) surfaces = surfaces && actual(k, j, i) == real(0.0);
        }
        int flags[4] = {int(snapshot(w) == snapshot(replay_w) && snapshot(previous) == snapshot(replay_previous)),
                        int(snapshot(xi) == before_x && snapshot(eta) == before_e), int(snapshot(previous) == before_w), int(finite && surfaces)};
        int global[4];
        MPI_Allreduce(flags, global, 4, MPI_INT, MPI_MIN, grid.get_comm());
        double local_errors[2] = {maximum_error, maximum_value}, global_errors[2];
        MPI_Allreduce(local_errors, global_errors, 2, MPI_DOUBLE, MPI_MAX, grid.get_comm());
        const double error = global_errors[0] / std::max(1e-12, global_errors[1]);
        const bool pass = global[0] && global[1] && global[2] && global[3] && error <= (sizeof(Real) == sizeof(double) ? 2e-10 : 5e-4);
        if (grid.get_mpi_rank() == 0) std::printf("ranks=%d iterations=%d stretched=%d step=%d exact=%d sources=%d history=%d surfaces_finite=%d reference_error=%.3e %s\n", grid.get_mpi_size(), iterations, int(stretched), step, global[0], global[1], global[2], global[3], error, pass ? "PASS" : "FAIL");
        failures += !pass;
    }
    Kokkos::fence();
    MPI_Barrier(grid.get_comm());
    return failures;
}

int run(const Grid& grid, HaloExchanger& halo) {
    int failures = 0;
    for (bool stretched : {false, true}) for (int iterations : {1, 4}) failures += run_case(grid, halo, iterations, stretched);
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 0, failures = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        Kokkos::initialize(argc, argv);
        if (argc != 2 || (ranks != 1 && ranks != 2 && ranks != 4)) fatal("Provide a configuration and use 1, 2 or 4 ranks.");
        {
            VVM::Utils::ConfigurationManager config(argv[1]);
            Grid grid(config);
#if defined(ENABLE_NCCL)
            ncclUniqueId id;
            if (rank == 0) nccl_check(ncclGetUniqueId(&id));
            MPI_Bcast(&id, int(sizeof(id)), MPI_BYTE, 0, grid.get_comm());
            ncclComm_t comm;
            nccl_check(ncclCommInitRank(&comm, ranks, id, rank));
            {
                HaloExchanger halo(config, grid, comm, Kokkos::Cuda().cuda_stream());
                failures = run(grid, halo);
                Kokkos::fence();
            }
            nccl_check(ncclCommDestroy(comm));
#else
            HaloExchanger halo(grid);
            failures = run(grid, halo);
#endif
        }
        Kokkos::finalize();
    } catch (const std::exception& error) { fatal(error.what()); }
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}

