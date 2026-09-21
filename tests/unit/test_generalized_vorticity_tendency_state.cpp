#include "core/BoundaryConditionManager.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "dynamics/DynamicalCore.hpp"
#include "dynamics/operators/GeneralizedVorticityTendency.hpp"
#include "dynamics/operators/GeneralizedBuoyancy.hpp"
#include "dynamics/temporal_schemes/TimeIntegrator.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(ENABLE_NCCL)
#include <nccl.h>
#endif

namespace {

using namespace VVM;
using Json = nlohmann::json;
using Values = std::vector<Real>;
using Vorticity = Dynamics::Operators::GeneralizedVorticityTendency;

void
require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Own the config directory, including cleanup on an exception. There are no
// external sounding, restart or P3 lookup-table dependencies in this test.
struct TemporaryConfig {
    std::filesystem::path directory;

    TemporaryConfig() {
        const auto pattern =
            (std::filesystem::temp_directory_path() / "vvm_tendency_state_XXXXXX").string();
        std::vector<char> path(pattern.begin(), pattern.end());
        path.push_back('\0');
        const char* created = ::mkdtemp(path.data());
        require(created != nullptr, "could not create test configuration directory");
        directory = created;
    }

    ~TemporaryConfig() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }

    std::string
    write() const {
        Json config = Json::parse(R"({
          "grid": {
            "horizontal": {
              "nx": 16, "ny": 8, "n_halo_cells": 2,
              "geometry": {
                "kind": "regular_latlon", "earth_radius_m": 6371220,
                "longitude_bounds_deg": [-1,1], "latitude_bounds_deg": [-1,1]
              },
              "topology": {"q1": "periodic", "q2": "bounded"}
            },
            "vertical": {"nz": 8, "type": "default", "dz": 250, "dz1": 250}
          },
          "simulation": {
            "idealized_test": "jung2019_barotropic", "dt_s": 1,
            "total_time_s": 2, "output_interval_s": 1
          },
          "initial_conditions": {"jung2019": {"case": 1}},
          "dynamics": {"solver": {
            "w_solver_method": "tridiagonal", "iteration": 10,
            "initial_iterations": 10, "vertical_iterations": 10, "WRXMU": 100
          }},
          "constants": {"gravity": 9.806, "Rd": 287.04, "Cp": 1004.5, "P0": 100000},
          "physics": {}, "output": {"engine": "HDF5"}
        })");
        for (const char* name : {"xi", "eta", "zeta"}) {
            for (const char* term : {"advection", "stretching", "twisting"}) {
                config["dynamics"]["prognostic_variables"][name]["tendency_terms"][term] = {
                    {"enable", true},
                    {"spatial_scheme", "Takacs"},
                    {"temporal_scheme", "AdamsBashforth2"}};
            }
        }
        // Keep this a valid Jung2019 RLL production configuration. Forward-Euler
        // consumption is tested independently below with TimeIntegrator; Jung2019
        // itself intentionally permits only the three AB2 vorticity terms above.
        const auto path = directory / "config.json";
        std::ofstream output(path);
        output << config.dump(2);
        output.close();
        require(static_cast<bool>(output), "could not write test configuration");
        return path.string();
    }
};

// Copy by logical indices. Host mirrors can alias CPU storage, and a View's
// allocation can include padding. Neither may hide a later input mutation.
Values
snapshot(const Core::Field<3>& field) {
    const auto values = field.get_host_data();
    Values result;
    result.reserve(values.size());
    for (std::size_t k = 0; k < values.extent(0); ++k) {
        for (std::size_t j = 0; j < values.extent(1); ++j) {
            for (std::size_t i = 0; i < values.extent(2); ++i) {
                result.push_back(values(k, j, i));
            }
        }
    }
    return result;
}

bool
bitwise_equal(const Values& first, const Values& second) {
    return first.size() == second.size() &&
           (first.empty() ||
               std::memcmp(first.data(), second.data(), first.size() * sizeof(Real)) == 0);
}

void
fill_inputs(
    Core::State& state, const Core::Grid& grid, const Core::Parameters& params, int version) {
    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int top = nz - h - 1;
    const Real amplitude = real(1.0) + real(0.137) * version;

    // Non-binary, non-unit density is deliberate: unit density and zero
    // vorticity would fail to expose the removed divide/multiply round-trip.
    for (const char* name : {"rhobar", "rhobar_up"}) {
        auto data = state.get_field<1>(name).get_host_data();
        for (int k = 0; k < nz; ++k) {
            data(k) = std::string(name) == "rhobar" ? real(1.07) + real(0.013) * k
                                                    : real(1.11) + real(0.017) * k;
        }
        Kokkos::deep_copy(state.get_field<1>(name).get_mutable_device_data(), data);
    }
    Kokkos::deep_copy(params.fact1_xi_eta.get_device_data(), real(0.9));
    Kokkos::deep_copy(params.fact2_xi_eta.get_device_data(), real(1.1));
    Kokkos::deep_copy(params.flex_height_coef_up.get_device_data(), real(1.0));
    Kokkos::deep_copy(params.flex_height_coef_mid.get_device_data(), real(1.0));
    Kokkos::deep_copy(state.get_field<2>("f_2d").get_mutable_device_data(), real(8.1e-5));

    for (const char* name : {"u_con", "v_con", "w", "xi_con", "eta_con", "zeta", "zeta_con"}) {
        auto data = state.get_field<3>(name).get_host_data();
        const std::string n(name);
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    Real value = real(0.0);
                    if (n == "u_con") {
                        value = real(8.731e-7) *
                                (real(1) + real(.019) * i + real(.027) * j + real(.043) * k);
                    }
                    else if (n == "v_con") {
                        value = -real(5.123e-7) *
                                (real(1) - real(.031) * i + real(.011) * j + real(.037) * k);
                    }
                    else if (n == "xi_con") {
                        value = real(1.23456789e-8) *
                                (real(1) + real(.017) * i + real(.021) * j + real(.019) * k);
                    }
                    else if (n == "eta_con") {
                        value = -real(2.7182818e-8) *
                                (real(1) + real(.023) * i - real(.017) * j + real(.013) * k);
                    }
                    else if (n == "w") {
                        value = real(.00025) * (k - h + 1) * (top - k) *
                                (real(1) + real(.01) * i + real(.02) * j);
                    }
                    else {
                        value = real(3.14159265e-4) *
                                (real(1) - real(.007) * i + real(.009) * j + real(.023) * k);
                    }
                    data(k, j, i) = amplitude * value;
                }
            }
        }
        Kokkos::deep_copy(state.get_field<3>(name).get_mutable_device_data(), data);
    }
    Kokkos::deep_copy(state.get_field<1>("thbar").get_mutable_device_data(), real(300));
    Kokkos::deep_copy(params.gravity, real(9.806));
    auto theta = state.get_field<3>("th").get_host_data();
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                theta(k, j, i) =
                    real(300) + amplitude * (real(.0625) * i + real(.125) * j + real(.03125) * k);
            }
        }
    }
    Kokkos::deep_copy(state.get_field<3>("th").get_mutable_device_data(), theta);

    // In this tendency-only fixture, deliberately unrelated physical caches
    // detect an accidental import/export of prognostic horizontal vorticity.
    for (const char* name : {"u", "v", "xi", "eta"}) {
        const Real value = std::string(name) == "u"    ? real(71)
                           : std::string(name) == "v"  ? real(-93)
                           : std::string(name) == "xi" ? real(123)
                                                       : real(-456);
        Kokkos::deep_copy(state.get_field<3>(name).get_mutable_device_data(), value);
    }
    for (const char* name : {"ITYPEU", "ITYPEV", "ITYPEW"}) {
        Kokkos::deep_copy(state.get_field<3>(name).get_mutable_device_data(), real(1));
    }
}

#if defined(KOKKOS_ENABLE_CUDA)
void
check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct CapturedTendency {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaStream_t stream = Kokkos::Cuda().cuda_stream();

    ~CapturedTendency() {
        if (executable) {
            cudaGraphExecDestroy(executable);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }

    void
    capture(Dynamics::DynamicalCore& core) {
        Kokkos::fence();
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        try {
            core.calculate_vorticity_tendencies();
        }
        catch (...) {
            cudaStreamEndCapture(stream, &graph);
            throw;
        }
        check_cuda(cudaStreamEndCapture(stream, &graph), "end capture");
        check_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
            "instantiate graph");
    }

    void
    run() const {
        check_cuda(cudaGraphLaunch(executable, stream), "launch graph");
        check_cuda(cudaStreamSynchronize(stream), "complete graph");
    }
};
#endif

void
run_test(const std::string& path
#if defined(ENABLE_NCCL)
    ,
    ncclComm_t comm,
    cudaStream_t stream
#endif
) {
    Utils::ConfigurationManager config(path);
    Core::Grid grid(config, MPI_COMM_WORLD);
    Core::Parameters params(config, grid);
#if defined(ENABLE_NCCL)
    Core::State state(config, params, grid, comm, stream);
    Core::HaloExchanger halo(config, grid, comm, stream);
#else
    Core::State state(config, params, grid);
    Core::HaloExchanger halo(grid);
#endif
    Core::BoundaryConditionManager boundary(grid, true);
    params.max_topo_idx = grid.get_halo_cells();
    fill_inputs(state, grid, params, 1);

    const std::array<int, 3> dims{grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()};

    // Jung2019 has no Forward-Euler vorticity terms, so State correctly does
    // not allocate these buffers from configuration. Add explicit scratch only
    // for the isolated mixed AB2/FE TimeIntegrator contract tested below.
    for (const char* name : {"xi", "eta", "zeta"}) {
        const std::string field_name = "fe_tendency_" + std::string(name);
        if (!state.has_field(field_name)) {
            state.add_field<3>(field_name, dims);
        }
    }

    Dynamics::DynamicalCore core(config, grid, params, state, halo, boundary);
    Vorticity reference(grid.geometry());
    Dynamics::Operators::GeneralizedBuoyancy buoyancy(grid.geometry());
    Core::Field<3> part("canonical_term_reference",
        {grid.get_local_total_points_z(),
            grid.get_local_total_points_y(),
            grid.get_local_total_points_x()});

    const std::array<const char*, 3> equations = {"xi", "eta", "zeta"};
    const std::array<const char*, 12> protected_fields = {"xi_con",
        "eta_con",
        "zeta",
        "zeta_con",
        "xi",
        "eta",
        "u_con",
        "v_con",
        "u",
        "v",
        "w",
        "th"};

    // This reference checks orchestration/representation, not a second
    // independent discretization of the already-tested numerical operators.
    const auto verify = [&](const auto& evaluate, const std::string& label) {
        const std::size_t slot = state.get_step() % 2;
        std::array<Values, 12> before;
        for (std::size_t n = 0; n < protected_fields.size(); ++n) {
            before[n] = snapshot(state.get_field<3>(protected_fields[n]));
        }

        std::array<Values, 3> expected, magnitudes, inactive;
        for (std::size_t n = 0; n < equations.size(); ++n) {
            const std::string name(equations[n]);
            for (const auto term : {Vorticity::Term::Transport,
                     Vorticity::Term::Stretching,
                     Vorticity::Term::Twisting}) {
                part.set_to_zero();
                reference.add_from_canonical_state(state, grid, params, part, name, term);
                const auto values = snapshot(part);
                if (expected[n].empty()) {
                    expected[n].assign(values.size(), real(0));
                    magnitudes[n].assign(values.size(), real(0));
                }
                for (std::size_t p = 0; p < values.size(); ++p) {
                    require(std::isfinite(values[p]), label + ": nonfinite reference");
                    expected[n][p] += values[p];
                    magnitudes[n][p] += std::abs(values[p]);
                }
            }
            require(*std::max_element(magnitudes[n].begin(), magnitudes[n].end()) > real(1e-20),
                label + ": ineffective zero tendency fixture for " + name);
            // Active history must be replaced, while the other AB2 slot must
            // stay bitwise intact. Sentinels also detect whole-array rescaling.
            for (std::size_t history = 0; history < 2; ++history) {
                Kokkos::deep_copy(state.get_field<3>("d_" + name + "_" + std::to_string(history))
                                      .get_mutable_device_data(),
                    real(1e-9) * real(17 + n + 3 * history));
            }
            inactive[n] =
                snapshot(state.get_field<3>("d_" + name + "_" + std::to_string(1 - slot)));
        }
        evaluate();
        Kokkos::fence();

        for (std::size_t n = 0; n < protected_fields.size(); ++n) {
            require(bitwise_equal(before[n], snapshot(state.get_field<3>(protected_fields[n]))),
                label + ": tendency evaluation modified " + protected_fields[n]);
        }

        const Real tolerance = real(256) * std::numeric_limits<Real>::epsilon();
        std::array<Values, 3> active;
        for (std::size_t n = 0; n < equations.size(); ++n) {
            const std::string name(equations[n]);
            active[n] = snapshot(state.get_field<3>("d_" + name + "_" + std::to_string(slot)));
            require(bitwise_equal(inactive[n],
                        snapshot(state.get_field<3>("d_" + name + "_" + std::to_string(1 - slot)))),
                label + ": inactive AB2 history was changed for " + name);
            for (std::size_t p = 0; p < active[n].size(); ++p) {
                const Real scale = std::max(magnitudes[n][p], real(1e-30));
                require(std::isfinite(active[n][p]) &&
                            std::abs(active[n][p] - expected[n][p]) <= tolerance * scale,
                    label + ": incorrect canonical AB2 tendency for " + name);
            }
        }
        return active;
    };

    const auto fill_canonical_fe = [&](std::size_t component) {
        const std::string name(equations[component]);
        auto& fe = state.get_field<3>("fe_tendency_" + name);
        fe.set_to_zero();

        // These are valid generalized canonical sources even though the dry
        // Jung2019 production configuration intentionally does not enable them.
        reference
            .add_from_canonical_state(state, grid, params, fe, name, Vorticity::Term::Planetary);

        if (component < 2) {
            const int h = grid.get_halo_cells();
            const int end = grid.get_local_total_points_z() - h - 1;
            if (component == 0) {
                buoyancy.add_xi_tendency(state.get_field<3>("th"),
                    state.get_field<1>("thbar"),
                    params.gravity,
                    fe,
                    h,
                    end);
            }
            else {
                buoyancy.add_eta_tendency(state.get_field<3>("th"),
                    state.get_field<1>("thbar"),
                    params.gravity,
                    fe,
                    h,
                    end);
            }
        }
    };

    for (std::size_t slot = 0; slot < 3; ++slot) {
        state.set_step(slot);
        fill_inputs(state, grid, params, 1);
        const std::string label = "AB2 slot " + std::to_string(slot);
        const auto direct = [&]() {
            core.calculate_vorticity_tendencies();
        };
        verify(direct, label + " warmup");
        verify(direct, label + " repeat");
        {
#if defined(KOKKOS_ENABLE_CUDA)
            // Host-selected AB2 parity is fixed in a captured graph. Capture one
            // graph for each parity; do not claim replay dynamically reads step_.
            CapturedTendency graph;
            graph.capture(core);
#endif
            for (int version : {2, 3}) {
                fill_inputs(state, grid, params, version);
#if defined(KOKKOS_ENABLE_CUDA)
                const auto replay = verify([&]() {
                    graph.run();
                }, label + " replay");
                const auto ordinary = verify(direct, label + " changed-input direct");
                for (std::size_t n = 0; n < replay.size(); ++n) {
                    require(bitwise_equal(replay[n], ordinary[n]),
                        label + ": graph/direct tendency mismatch");
                }
#else
                verify(direct, label + " changed-input direct");
#endif
            }
        } // Destroy parity graph before AB2 swaps prognostic storage.

        // Test the real NumericalMethod/TimeIntegrator consumer. Physics caches
        // are deliberately unrelated and must not be used as update targets.
        verify(direct, label + " before mixed AB2/FE advance");
        const int h = grid.get_halo_cells(), nz = dims[0], ny = dims[1], nx = dims[2];
        const int top = nz - h - 1;
        const Real dt = real(0.5), tolerance = real(256) * std::numeric_limits<Real>::epsilon();
        const auto physical_xi = snapshot(state.get_field<3>("xi"));
        const auto physical_eta = snapshot(state.get_field<3>("eta"));
        for (std::size_t n = 0; n < equations.size(); ++n) {
            const std::string name(equations[n]);
            const std::string target_name = n == 0 ? "xi_con" : n == 1 ? "eta_con" : "zeta";
            auto& target = state.get_field<3>(target_name);
            auto& previous = state.get_field<3>(target_name + "_m");
            const auto old = snapshot(target);
            const auto now =
                snapshot(state.get_field<3>("d_" + name + "_" + std::to_string(slot % 2)));
            const auto prev =
                snapshot(state.get_field<3>("d_" + name + "_" + std::to_string(1 - slot % 2)));
            fill_canonical_fe(n);
            const auto fe = snapshot(state.get_field<3>("fe_tendency_" + name));

            // Production configuration remains legal AB2-only Jung2019. This
            // isolated integrator verifies that an additional canonical FE
            // group is consumed without touching physical compatibility fields.
            Dynamics::TimeIntegrator mixed_integrator(name, true, true);
            mixed_integrator.step(state, grid, params, dt, target, &previous);
            const auto advanced = snapshot(target);
            require(bitwise_equal(old, snapshot(previous)),
                label + ": AB2 previous state lost canonical storage ownership");
            for (int k = (n == 2 ? top : h); k < (n == 2 ? top + 1 : top); ++k) {
                for (int j = h; j < ny - h; ++j) {
                    for (int i = h; i < nx - h; ++i) {
                        const std::size_t p = (static_cast<std::size_t>(k) * ny + j) * nx + i;
                        const Real ab =
                            slot == 0 ? now[p] : real(1.5) * now[p] - real(.5) * prev[p];
                        const Real expected_value = (old[p] + dt * ab) + dt * fe[p];
                        const Real scale = std::max({std::abs(old[p]),
                            std::abs(dt * ab),
                            std::abs(dt * fe[p]),
                            real(1e-30)});
                        require(std::isfinite(advanced[p]) &&
                                    std::abs(advanced[p] - expected_value) <= tolerance * scale,
                            label + ": mixed AB2/FE update is not canonical");
                    }
                }
            }
        }
        require(bitwise_equal(physical_xi, snapshot(state.get_field<3>("xi"))) &&
                    bitwise_equal(physical_eta, snapshot(state.get_field<3>("eta"))),
            label + ": integration wrote physical compatibility fields");
    }
    std::cout << "PASS: RLL tendency evaluation preserves prognostic state; "
                 "AB2 histories and isolated mixed AB2/FE explicit-target updates are canonical";
#if defined(KOKKOS_ENABLE_CUDA)
    std::cout << "; changed-input graph replay passed";
#endif
    std::cout << '\n';
}

} // namespace

int
main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        std::cerr << "test_rll_vorticity_tendency_state requires exactly one MPI rank\n";
        MPI_Finalize();
        return 2;
    }
    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
    int result = 0;
#if defined(ENABLE_NCCL)
    ncclComm_t comm = nullptr;
#endif
    try {
        TemporaryConfig temporary;
#if defined(ENABLE_NCCL)
        ncclUniqueId id;
        require(ncclGetUniqueId(&id) == ncclSuccess, "NCCL unique id failed");
        require(ncclCommInitRank(&comm, 1, id, 0) == ncclSuccess, "NCCL initialization failed");
        run_test(temporary.write(), comm, Kokkos::Cuda().cuda_stream());
#else
        run_test(temporary.write());
#endif
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
#if defined(ENABLE_NCCL)
    if (comm) {
        ncclCommDestroy(comm);
    }
#endif
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
