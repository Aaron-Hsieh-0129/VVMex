#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindColumnRecovery.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
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
using VVM::Dynamics::HorizontalWindColumnRecovery;
using VVM::Utils::ConfigurationManager;

constexpr int nz = 6;
constexpr int top = 4;
const Real sentinel = real(-12345.0);

[[noreturn]] void fatal(const char* message) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::fprintf(stderr, "Rank %d fatal error: %s\n", rank, message);
    std::fflush(stderr);

    // A rank-local failure must not leave its peers waiting in communication.
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void require_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        fatal(operation);
    }
}

template<std::size_t Dim>
void append_values(std::vector<Real>& values, const Field<Dim>& field) {
    const auto host = field.get_host_data();

    // Copy values into independent storage. A CPU host mirror can alias a Field.
    if constexpr (Dim == 1) {
        for (std::size_t i = 0; i < host.extent(0); ++i) {
            values.push_back(host(i));
        }
    } else if constexpr (Dim == 2) {
        for (std::size_t j = 0; j < host.extent(0); ++j) {
            for (std::size_t i = 0; i < host.extent(1); ++i) {
                values.push_back(host(j, i));
            }
        }
    } else if constexpr (Dim == 3) {
        for (std::size_t k = 0; k < host.extent(0); ++k) {
            for (std::size_t j = 0; j < host.extent(1); ++j) {
                for (std::size_t i = 0; i < host.extent(2); ++i) {
                    values.push_back(host(k, j, i));
                }
            }
        }
    } else {
        static_assert(Dim >= 1 && Dim <= 3, "Unsupported snapshot dimension.");
    }
}

bool same_bits(const std::vector<Real>& first, const std::vector<Real>& second) {
    if (first.size() != second.size()) return false;
    return first.empty() || std::memcmp(first.data(), second.data(), first.size() * sizeof(Real)) == 0;
}

bool all_finite(const std::vector<Real>& values) {
    return std::all_of(values.begin(), values.end(), [](Real value) {
        return std::isfinite(static_cast<double>(value));
    });
}

struct Inputs {
    Field<2> rhs_psi;
    Field<2> rhs_chi;
    Field<2> current_psi;
    Field<2> current_chi;
    Field<2> previous_psi;
    Field<2> previous_chi;
    Field<3> w;
    Field<3> omega1;
    Field<3> omega2;
    Field<1> spacing;

    Inputs(int ny, int nx)
        : rhs_psi("chain_rhs_psi", {ny, nx}),
          rhs_chi("chain_rhs_chi", {ny, nx}),
          current_psi("chain_current_psi", {ny, nx}),
          current_chi("chain_current_chi", {ny, nx}),
          previous_psi("chain_previous_psi", {ny, nx}),
          previous_chi("chain_previous_chi", {ny, nx}),
          w("chain_w", {nz, ny, nx}),
          omega1("chain_omega1", {nz, ny, nx}),
          omega2("chain_omega2", {nz, ny, nx}),
          spacing("chain_spacing", {nz - 1}) {}

    void initialize(const Grid& grid, int replay) {
        const int nx = grid.get_local_total_points_x();
        const int ny = grid.get_local_total_points_y();
        const int h = grid.get_halo_cells();
        const Real two_pi = real(2.0) * std::acos(real(-1.0));
        const Real factor = replay == 3 ? real(0.0) : static_cast<Real>(replay + 1);

        Real radius = real(1.0);
        if (grid.geometry().kind() == GeometryKind::RegularLatLon) {
            radius = static_cast<const RegularLatLonGeometry&>(grid.geometry()).radius();
        }

        std::array<Field<2>*, 6> planes = {
            &rhs_psi, &rhs_chi, &current_psi,
            &current_chi, &previous_psi, &previous_chi
        };

        std::array<Field<2>::HostMirrorType, 6> host;
        for (std::size_t n = 0; n < planes.size(); ++n) {
            host[n] = planes[n]->get_host_data();
        }

        auto w_host = w.get_host_data();
        auto omega1_host = omega1.get_host_data();
        auto omega2_host = omega2.get_host_data();
        auto spacing_host = spacing.get_host_data();

        for (int j = 0; j < ny; ++j) {
            const Real y = static_cast<Real>(grid.get_local_physical_start_y() + j - h);
            const Real angle_y = two_pi * (y + real(0.5)) / static_cast<Real>(grid.get_global_points_y());

            for (int i = 0; i < nx; ++i) {
                const Real x = static_cast<Real>(grid.get_local_physical_start_x() + i - h);
                const Real angle_z = two_pi * (x + real(1.0)) / static_cast<Real>(grid.get_global_points_x());
                const Real angle_t = two_pi * (x + real(0.5)) / static_cast<Real>(grid.get_global_points_x());

                host[0](j, i) = factor * real(1e-10) * std::sin(angle_z);
                host[1](j, i) = factor * real(-6e-11) * std::cos(real(2.0) * angle_t);

                host[2](j, i) = factor * real(1000.0) * (std::sin(angle_z) * std::cos(angle_y) + real(0.3));
                host[3](j, i) = factor * real(700.0) * (std::cos(angle_t) * std::sin(angle_y) - real(0.2));

                host[4](j, i) = factor * real(800.0) * (std::cos(angle_z) * std::cos(angle_y) - real(0.1));
                host[5](j, i) = factor * real(400.0) * (std::sin(angle_t) * std::sin(angle_y) + real(0.15));

                for (int k = 0; k < nz; ++k) {
                    const Real level = static_cast<Real>(k);

                    w_host(k, j, i) = factor * real(0.1) * ((real(1.0) + real(0.125) * level) * std::sin(angle_t) + real(0.3) * std::cos(angle_y));
                    omega1_host(k, j, i) = factor * real(2e-4) / radius * (real(1.0) + real(0.125) * level + real(0.2) * std::cos(angle_t));
                    omega2_host(k, j, i) = factor * real(-3e-4) / radius * (real(1.0) + real(0.0625) * level + real(0.1) * std::sin(angle_y));
                }
            }
        }

        // Replay zero uses uniform spacing; later replays use stretched spacing.
        // Update values without replacing any captured allocation.
        for (int k = 0; k < nz - 1; ++k) {
            spacing_host(k) = replay == 0 ? real(100.0) : real(40.0) * static_cast<Real>(k + 1) + real(10.0) * static_cast<Real>(replay);
        }

        for (std::size_t n = 0; n < planes.size(); ++n) {
            Kokkos::deep_copy(planes[n]->get_mutable_device_data(), host[n]);
        }

        Kokkos::deep_copy(w.get_mutable_device_data(), w_host);
        Kokkos::deep_copy(omega1.get_mutable_device_data(), omega1_host);
        Kokkos::deep_copy(omega2.get_mutable_device_data(), omega2_host);
        Kokkos::deep_copy(spacing.get_mutable_device_data(), spacing_host);
    }

    std::vector<Real> values() const {
        std::vector<Real> result;

        append_values(result, rhs_psi);
        append_values(result, rhs_chi);
        append_values(result, current_psi);
        append_values(result, current_chi);
        append_values(result, previous_psi);
        append_values(result, previous_chi);
        append_values(result, w);
        append_values(result, omega1);
        append_values(result, omega2);
        append_values(result, spacing);

        return result;
    }
};

struct Outputs {
    Field<2> psi;
    Field<2> chi;
    Field<3> wind1;
    Field<3> wind2;

    Outputs(const std::string& prefix, int ny, int nx)
        : psi(prefix + "_psi", {ny, nx}),
          chi(prefix + "_chi", {ny, nx}),
          wind1(prefix + "_wind1", {nz, ny, nx}),
          wind2(prefix + "_wind2", {nz, ny, nx}) {}

    void reset() {
        Kokkos::deep_copy(psi.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(chi.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(wind1.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(wind2.get_mutable_device_data(), sentinel);
    }

    std::vector<Real> values() const {
        std::vector<Real> result;

        append_values(result, psi);
        append_values(result, chi);
        append_values(result, wind1);
        append_values(result, wind2);

        return result;
    }

    bool check_regions(const Grid& grid, int bottom, bool require_zero) const {
        const auto first = wind1.get_host_data();
        const auto second = wind2.get_host_data();

        const int nx = grid.get_local_total_points_x();
        const int ny = grid.get_local_total_points_y();
        const int h = grid.get_halo_cells();
        const int end_i = h + grid.get_local_physical_points_x();
        const int end_j = h + grid.get_local_physical_points_y();

        bool valid = true;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const bool written = k >= bottom && k <= top && j >= h && j < end_j && i >= h && i < end_i;

                    if (!written) {
                        valid = valid && first(k, j, i) == sentinel && second(k, j, i) == sentinel;
                    } else if (require_zero) {
                        valid = valid && first(k, j, i) == real(0.0) && second(k, j, i) == real(0.0);
                    }
                }
            }
        }

        return valid;
    }
};

#if defined(ENABLE_NCCL)

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
        fatal(message.c_str());
    }
}

void require_nccl(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        const std::string message = std::string(operation) + ": " + ncclGetErrorString(status);
        fatal(message.c_str());
    }
}

struct TestGraph {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    explicit TestGraph(cudaStream_t stream_in) : stream(stream_in) {}

    TestGraph(const TestGraph&) = delete;
    TestGraph& operator=(const TestGraph&) = delete;

    ~TestGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }

    void begin() {
        require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "Begin solver-column capture");
    }

    void finish() {
        require_cuda(cudaStreamEndCapture(stream, &graph), "End solver-column capture");

        if (!graph) {
            fatal("Solver-column capture returned a null graph.");
        }

        std::size_t node_count = 0;
        require_cuda(cudaGraphGetNodes(graph, nullptr, &node_count), "Read solver-column graph nodes");

        if (node_count == 0) {
            fatal("Solver-column capture recorded no nodes.");
        }

        require_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "Instantiate solver-column graph");
    }

    void launch() {
        require_cuda(cudaGraphLaunch(executable, stream), "Launch solver-column graph");
        require_cuda(cudaStreamSynchronize(stream), "Complete solver-column graph");
    }
};

#endif

int run_case(const Grid& grid, HaloExchanger& halo, int iterations, int bottom) {
    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const MPI_Comm comm = grid.get_comm();

    Inputs input(ny, nx);
    Outputs direct("chain_direct", ny, nx);
    Outputs replayed("chain_replayed", ny, nx);

    HorizontalEllipticSolver direct_solver(grid, halo);
    HorizontalEllipticSolver replay_solver(grid, halo);
    const HorizontalWindColumnRecovery recovery(grid.geometry());

    HorizontalEllipticSolver::Options options;
    options.iterations = iterations;
    options.diagonal_shift = real(0.25);
    options.refresh_initial_halos = true;

    const auto execute = [&](HorizontalEllipticSolver& solver, Outputs& output) {
        solver.make_extrapolated_guess(input.current_psi, input.previous_psi, output.psi);
        solver.make_extrapolated_guess(input.current_chi, input.previous_chi, output.chi);
        solver.solve_at_z_and_t(input.rhs_psi, output.psi, input.rhs_chi, output.chi, options);
        recovery.recover(output.psi, output.chi, input.w, input.omega1, input.omega2, input.spacing, output.wind1, output.wind2, bottom, top);
    };

#if defined(ENABLE_NCCL)
    static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>, "This NCCL replay test requires Kokkos::Cuda.");
    const Kokkos::Cuda execution;
    TestGraph graph(execution.cuda_stream());
    const char* execution_name = grid.get_mpi_size() > 1 ? "cuda_graph_nccl" : "cuda_graph";
#else
    const char* execution_name = "direct_repeat";
#endif

    // Catch while graph, solver, and field storage are still alive.
    // Abort immediately on an exception rather than unwinding while peers wait.
    try {
        input.initialize(grid, 0);
        direct.reset();
        replayed.reset();

        HorizontalWindColumnRecovery::prepare_execution();
        Kokkos::fence();

        // Explicitly prepare both solver paths and their halo communication.
        require_mpi(MPI_Barrier(comm), "Barrier before direct preparation");
        execute(direct_solver, direct);
        Kokkos::fence();

        require_mpi(MPI_Barrier(comm), "Barrier before replay preparation");
        execute(replay_solver, replayed);
        Kokkos::fence();

        direct.reset();
        replayed.reset();
        Kokkos::fence();

#if defined(ENABLE_NCCL)
        // No MPI barrier, host snapshot, reset, or fence occurs inside capture.
        require_mpi(MPI_Barrier(comm), "Barrier before collective capture");
        graph.begin();
        execute(replay_solver, replayed);
        graph.finish();
        require_mpi(MPI_Barrier(comm), "Barrier after collective capture");
#endif

        int failures = 0;
        std::vector<Real> previous_result;

        for (int replay = 0; replay < 4; ++replay) {
            input.initialize(grid, replay);
            direct.reset();
            replayed.reset();
            Kokkos::fence();

            const auto before = input.values();

            require_mpi(MPI_Barrier(comm), "Barrier before direct execution");
            execute(direct_solver, direct);
            Kokkos::fence();

            const bool direct_inputs = same_bits(before, input.values());

            require_mpi(MPI_Barrier(comm), "Barrier before replay execution");

#if defined(ENABLE_NCCL)
            graph.launch();
#else
            execute(replay_solver, replayed);
            Kokkos::fence();
#endif

            const bool replay_inputs = same_bits(before, input.values());
            const auto expected = direct.values();
            const auto actual = replayed.values();

            const bool exact = same_bits(expected, actual);
            const bool inputs = direct_inputs && replay_inputs;
            const bool regions = direct.check_regions(grid, bottom, replay == 3) && replayed.check_regions(grid, bottom, replay == 3);
            const bool updated = replay == 0 || !same_bits(previous_result, actual);
            const bool finite = all_finite(expected) && all_finite(actual);

            std::array<int, 5> local_flags = {
                exact ? 1 : 0,
                inputs ? 1 : 0,
                regions ? 1 : 0,
                updated ? 1 : 0,
                finite ? 1 : 0
            };

            std::array<int, 5> global_flags = {};
            require_mpi(MPI_Allreduce(local_flags.data(), global_flags.data(), static_cast<int>(global_flags.size()), MPI_INT, MPI_MIN, comm), "Reduce replay checks");

            const bool passed = std::all_of(global_flags.begin(), global_flags.end(), [](int value) {
                return value == 1;
            });

            if (grid.get_mpi_rank() == 0) {
                const char* geometry_name = grid.geometry().kind() == GeometryKind::Cartesian ? "cartesian" : "regular_latlon";

                std::printf("%s ranks=%d execution=%s iterations=%d bottom=%d replay=%d exact=%d inputs=%d regions=%d updated=%d finite=%d %s\n",
                    geometry_name, grid.get_mpi_size(), execution_name, iterations, bottom, replay,
                    global_flags[0], global_flags[1], global_flags[2], global_flags[3], global_flags[4], passed ? "PASS" : "FAIL");
                std::fflush(stdout);
            }

            if (!passed) {
                ++failures;
            }

            previous_result = actual;
        }

        Kokkos::fence();
        require_mpi(MPI_Barrier(comm), "Barrier before graph cleanup");
        return failures;
    } catch (const std::exception& error) {
        fatal(error.what());
    } catch (...) {
        fatal("Unknown failure during solver-column replay.");
    }
}

int run_tests(const Grid& grid, HaloExchanger& halo) {
    int failures = 0;

    for (const int iterations : {1, 4}) {
        for (const int bottom : {0, 2, top}) {
            failures += run_case(grid, halo, iterations, bottom);
        }
    }

    Kokkos::fence();
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int failures = 0;

    try {
        Kokkos::initialize(argc, argv);

        if (argc != 2) {
            throw std::runtime_error("Usage: test_horizontal_wind_solver_column_replay <configuration.json>");
        }

        if (ranks != 1 && ranks != 2 && ranks != 4) {
            throw std::runtime_error("This test supports 1, 2, or 4 MPI ranks.");
        }

        {
            ConfigurationManager config(argv[1]);
            Grid grid(config, MPI_COMM_WORLD);

#if defined(ENABLE_NCCL)
            ncclUniqueId id;
            if (rank == 0) {
                require_nccl(ncclGetUniqueId(&id), "Create NCCL unique ID");
            }

            require_mpi(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, MPI_COMM_WORLD), "Broadcast NCCL unique ID");

            ncclComm_t nccl_comm = nullptr;
            require_nccl(ncclCommInitRank(&nccl_comm, ranks, id, rank), "Initialize NCCL communicator");

            {
                const Kokkos::Cuda execution;
                HaloExchanger halo(config, grid, nccl_comm, execution.cuda_stream());

                try {
                    failures = run_tests(grid, halo);
                } catch (const std::exception& error) {
                    fatal(error.what());
                } catch (...) {
                    fatal("Unknown failure in distributed replay tests.");
                }

                Kokkos::fence();
                require_mpi(MPI_Barrier(MPI_COMM_WORLD), "Barrier before halo cleanup");
            }

            require_nccl(ncclCommDestroy(nccl_comm), "Destroy NCCL communicator");
#else
            HaloExchanger halo(grid);
            failures = run_tests(grid, halo);
#endif
        }

        int global_failures = 0;
        require_mpi(MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), "Reduce test failures");
        failures = global_failures;

        if (rank == 0) {
            std::printf("test_horizontal_wind_solver_column_replay: %d failure(s)\n", failures);
        }

        Kokkos::finalize();
    } catch (const std::exception& error) {
        fatal(error.what());
    } catch (...) {
        fatal("Unknown test initialization or cleanup failure.");
    }

    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
