#include "core/Field.hpp"
#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/solvers/HorizontalWindColumnRecovery.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometry;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::HorizontalWindColumnRecovery;

int failures = 0;

template<std::size_t Dim>
std::vector<Real> snapshot(const Field<Dim>& field) {
    const auto host = field.get_host_data();
    std::vector<Real> values;
    values.reserve(host.size());

    // Store independent values: a Kokkos host mirror may alias CPU storage.
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

    return values;
}

bool same_bits(const std::vector<Real>& first, const std::vector<Real>& second) {
    if (first.size() != second.size()) return false;
    return first.empty() || std::memcmp(first.data(), second.data(), first.size() * sizeof(Real)) == 0;
}

#if defined(KOKKOS_ENABLE_CUDA)

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct TestGraph {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool capturing = false;

    explicit TestGraph(cudaStream_t stream_in) : stream(stream_in) {}

    TestGraph(const TestGraph&) = delete;
    TestGraph& operator=(const TestGraph&) = delete;

    ~TestGraph() {
        // Best-effort cleanup if an operation throws during capture.
        if (capturing) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
        }

        cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }

    void begin() {
        require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "Begin wind-column capture");
        capturing = true;
    }

    void finish() {
        const cudaError_t status = cudaStreamEndCapture(stream, &graph);
        capturing = false;
        require_cuda(status, "End wind-column capture");

        if (!graph) {
            throw std::runtime_error("Wind-column capture returned a null graph.");
        }

        std::size_t node_count = 0;
        require_cuda(cudaGraphGetNodes(graph, nullptr, &node_count), "Read wind-column graph nodes");

        if (node_count == 0) {
            throw std::runtime_error("Wind-column capture recorded no nodes.");
        }

        require_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "Instantiate wind-column graph");
    }

    void launch() {
        require_cuda(cudaGraphLaunch(executable, stream), "Launch wind-column graph");
        require_cuda(cudaStreamSynchronize(stream), "Complete wind-column graph");
    }
};

#endif

void test_replay(const HorizontalGeometry& geometry, Real coordinate_scale, int bottom, int top) {
    const auto layout = geometry.layout();
    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;
    constexpr int nz = 6;
    const Real sentinel = real(-12345.0);

    Field<2> psi("replay_psi", {ny, nx});
    Field<2> chi("replay_chi", {ny, nx});
    Field<3> w("replay_w", {nz, ny, nx});
    Field<3> omega1("replay_omega1", {nz, ny, nx});
    Field<3> omega2("replay_omega2", {nz, ny, nx});
    Field<1> spacing("replay_spacing", {nz - 1});

    Field<3> direct1("replay_direct1", {nz, ny, nx});
    Field<3> direct2("replay_direct2", {nz, ny, nx});
    Field<3> replay1("replay_output1", {nz, ny, nx});
    Field<3> replay2("replay_output2", {nz, ny, nx});

    const HorizontalWindColumnRecovery recovery(geometry);

    const auto input_snapshot = [&]() {
        std::vector<Real> values;

        const auto append = [&](const auto& field) {
            const auto part = snapshot(field);
            values.insert(values.end(), part.begin(), part.end());
        };

        append(psi);
        append(chi);
        append(w);
        append(omega1);
        append(omega2);
        append(spacing);

        return values;
    };

    const auto initialize = [&](int replay) {
        auto psi_host = psi.get_host_data();
        auto chi_host = chi.get_host_data();
        auto w_host = w.get_host_data();
        auto omega1_host = omega1.get_host_data();
        auto omega2_host = omega2.get_host_data();
        auto spacing_host = spacing.get_host_data();

        // The last replay removes all forcing to expose stale output.
        const Real factor = replay == 3 ? real(0.0) : static_cast<Real>(replay + 1);
        const Real potential_scale = coordinate_scale * coordinate_scale * real(1e-6);
        const Real vorticity_scale = real(1e-4) / coordinate_scale;

        for (int j = 0; j < ny; ++j) {
            const Real y = static_cast<Real>(j - h);

            for (int i = 0; i < nx; ++i) {
                const Real x = static_cast<Real>(i - h);

                psi_host(j, i) = factor * potential_scale * (real(0.125) * x * x - real(0.25) * y + real(0.0625) * x * y);
                chi_host(j, i) = factor * potential_scale * (real(-0.375) * x + real(0.125) * y * y - real(0.03125) * x * y);

                for (int k = 0; k < nz; ++k) {
                    const Real level = static_cast<Real>(k);

                    w_host(k, j, i) = factor * real(0.01) * (x + real(2.0) * y + real(0.25) * level * x);
                    omega1_host(k, j, i) = factor * vorticity_scale * (real(1.0) + real(0.125) * level + real(0.03125) * y);
                    omega2_host(k, j, i) = -factor * vorticity_scale * (real(0.75) + real(0.0625) * level + real(0.015625) * x);
                }
            }
        }

        // Change spacing values too, without replacing the captured allocation.
        for (int k = 0; k < nz - 1; ++k) {
            spacing_host(k) = real(100.0) + real(10.0) * replay + real(20.0) * k;
        }

        Kokkos::deep_copy(psi.get_mutable_device_data(), psi_host);
        Kokkos::deep_copy(chi.get_mutable_device_data(), chi_host);
        Kokkos::deep_copy(w.get_mutable_device_data(), w_host);
        Kokkos::deep_copy(omega1.get_mutable_device_data(), omega1_host);
        Kokkos::deep_copy(omega2.get_mutable_device_data(), omega2_host);
        Kokkos::deep_copy(spacing.get_mutable_device_data(), spacing_host);
    };

    const auto run_direct = [&]() {
        recovery.recover(psi, chi, w, omega1, omega2, spacing, direct1, direct2, bottom, top);
    };

    const auto run_replay = [&]() {
        recovery.recover(psi, chi, w, omega1, omega2, spacing, replay1, replay2, bottom, top);
    };

    initialize(0);
    Kokkos::fence();

    // Prepare backend state without evaluating recovery or touching model fields.
    HorizontalWindColumnRecovery::prepare_execution();

#if defined(KOKKOS_ENABLE_CUDA)
    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "This CUDA replay test requires the default execution space used by recover() to be CUDA.");

    // Destroy the graph before its captured fields and recovery object.
    // This captures the first actual recover() call after explicit backend preparation.
    TestGraph graph(Kokkos::Cuda().cuda_stream());
    graph.begin();
    run_replay();
    graph.finish();
    const char* execution = "cuda_graph";
#else
    const char* execution = "cpu_repeat";
#endif

    std::vector<Real> previous1;
    std::vector<Real> previous2;

    for (int replay = 0; replay < 4; ++replay) {
        initialize(replay);

        Kokkos::deep_copy(direct1.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(direct2.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(replay1.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(replay2.get_mutable_device_data(), sentinel);
        Kokkos::fence();

        const auto before = input_snapshot();

        run_direct();
        Kokkos::fence();

        const bool direct_inputs = same_bits(before, input_snapshot());

#if defined(KOKKOS_ENABLE_CUDA)
        graph.launch();
#else
        run_replay();
        Kokkos::fence();
#endif

        const bool replay_inputs = same_bits(before, input_snapshot());
        const auto expected1 = snapshot(direct1);
        const auto expected2 = snapshot(direct2);
        const auto actual1 = snapshot(replay1);
        const auto actual2 = snapshot(replay2);

        const bool exact = same_bits(expected1, actual1) && same_bits(expected2, actual2);
        const bool updated = replay == 0 || !same_bits(previous1, actual1) || !same_bits(previous2, actual2);

        bool finite = true;
        bool untouched = true;
        bool zero_result = true;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const std::size_t index = (static_cast<std::size_t>(k) * ny + j) * nx + i;
                    const bool written = k >= bottom && k <= top && j >= h && j < ny - h && i >= h && i < nx - h;

                    finite = finite && std::isfinite(expected1[index]) && std::isfinite(expected2[index]);
                    finite = finite && std::isfinite(actual1[index]) && std::isfinite(actual2[index]);

                    if (!written) {
                        untouched = untouched && expected1[index] == sentinel && expected2[index] == sentinel;
                        untouched = untouched && actual1[index] == sentinel && actual2[index] == sentinel;
                    } else if (replay == 3) {
                        zero_result = zero_result && actual1[index] == real(0.0) && actual2[index] == real(0.0);
                    }
                }
            }
        }

        const bool inputs = direct_inputs && replay_inputs;
        const bool passed = exact && inputs && finite && untouched && updated && zero_result;

        std::printf("%s execution=%s bottom=%d top=%d replay=%d exact=%d inputs=%d untouched=%d updated=%d finite=%d zero=%d %s\n",
            geometry.name(), execution, bottom, top, replay, static_cast<int>(exact), static_cast<int>(inputs),
            static_cast<int>(untouched), static_cast<int>(updated), static_cast<int>(finite),
            static_cast<int>(zero_result), passed ? "PASS" : "FAIL");

        if (!passed) ++failures;

        previous1 = actual1;
        previous2 = actual2;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    try {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size != 1) throw std::runtime_error("This test requires one MPI rank.");

        HorizontalDomainLayout layout;
        layout.global_nx = 12;
        layout.global_ny = 10;
        layout.local_physical_nx = 12;
        layout.local_physical_ny = 10;
        layout.halo = 2;

        const Real radius = real(6371220.0);
        const CartesianGeometry cartesian(layout, real(2.0), real(3.0));
        const RegularLatLonGeometry rll(layout, real(0.08), real(0.04), real(0.0), real(-0.3), radius);

        const std::array<std::array<int, 2>, 3> ranges = {{{0, 4}, {2, 3}, {3, 3}}};

        for (const auto& range : ranges) {
            test_replay(cartesian, real(1.0), range[0], range[1]);
            test_replay(rll, radius, range[0], range[1]);
        }
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_horizontal_wind_column_replay: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::finalize();
    MPI_Finalize();

    return global_failures == 0 ? 0 : 1;
}
