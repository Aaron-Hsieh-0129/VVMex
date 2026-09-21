#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/solvers/HorizontalWindTopologyConstraint.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

using namespace VVM;
using Json = nlohmann::json;
using Values = std::vector<Real>;

void
require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct TemporaryConfig {
    std::filesystem::path directory;

    TemporaryConfig() {
        const auto pattern =
            (std::filesystem::temp_directory_path() / "vvm_topology_constraint_XXXXXX").string();
        std::vector<char> path(pattern.begin(), pattern.end());
        path.push_back('\0');
        const char* created = ::mkdtemp(path.data());
        require(created != nullptr, "could not create topology-constraint test directory");
        directory = created;
    }

    ~TemporaryConfig() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }

    std::string
    write(const std::string& name, const std::string& q2_topology) const {
        Json config = Json::parse(R"({
          "grid": {
            "horizontal": {
              "nx": 12,
              "ny": 8,
              "n_halo_cells": 2,
              "geometry": {
                "kind": "regular_latlon",
                "earth_radius_m": 6371220.0,
                "longitude_bounds_deg": [0.0, 360.0],
                "latitude_bounds_deg": [-30.0, 30.0]
              },
              "topology": {"q1": "periodic", "q2": "bounded"}
            },
            "vertical": {
              "nz": 6,
              "type": "default",
              "dz": 250.0,
              "dz1": 250.0
            }
          }
        })");

        config["grid"]["horizontal"]["topology"]["q2"] = q2_topology;

        if (q2_topology == "periodic") {
            config["grid"]["horizontal"]["geometry"]["experimental_periodic_latitude"] = true;
        }

        const auto path = directory / (name + ".json");
        std::ofstream output(path);
        output << config.dump(2);
        output.close();
        require(static_cast<bool>(output), "could not write topology-constraint config");
        return path.string();
    }
};

Values
snapshot_physical(const Core::Field<3>& field, const Core::Grid& grid) {
    const auto data = field.get_host_data();
    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Values result;
    result.reserve(static_cast<std::size_t>(nz) * static_cast<std::size_t>(ny - 2 * h) *
                   static_cast<std::size_t>(nx - 2 * h));

    for (int k = 0; k < nz; ++k) {
        for (int j = h; j < ny - h; ++j) {
            for (int i = h; i < nx - h; ++i) {
                result.push_back(data(k, j, i));
            }
        }
    }

    return result;
}

bool
close_vectors(const Values& actual, const Values& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }

    const Real tolerance = real(2048.0) * std::numeric_limits<Real>::epsilon();

    for (std::size_t n = 0; n < actual.size(); ++n) {
        const Real scale =
            std::max(real(1.0), std::max(std::abs(actual[n]), std::abs(expected[n])));

        if (!std::isfinite(actual[n]) || std::abs(actual[n] - expected[n]) > tolerance * scale) {
            return false;
        }
    }

    return true;
}

void
fill_periodic_harmonic(
    Core::State& state, const Core::Grid& grid, Real q1_covariant, Real q2_harmonic) {
    using Core::Geometry::HorizontalLocation;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    const auto h1_u =
        grid.geometry().device_view(HorizontalLocation::U).contravariant_to_physical.a11;

    const auto h1_v =
        grid.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a11;

    auto u = state.get_field<3>("u").get_mutable_device_data();
    auto v = state.get_field<3>("v").get_mutable_device_data();

    Kokkos::parallel_for("FillPeriodicConstraintHarmonic",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            (void)k;
            u(k, j, i) = q1_covariant / h1_u(j, i);
            v(k, j, i) = q2_harmonic / h1_v(j, i);
        });
}

void
fill_channel_state(Core::State& state, const Core::Grid& grid, Real q1_covariant) {
    using Core::Geometry::HorizontalLocation;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    const auto h1_u =
        grid.geometry().device_view(HorizontalLocation::U).contravariant_to_physical.a11;

    auto u = state.get_field<3>("u").get_mutable_device_data();
    auto v = state.get_field<3>("v").get_mutable_device_data();

    Kokkos::parallel_for("FillChannelConstraintState",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u(k, j, i) = q1_covariant / h1_u(j, i);
            v(k, j, i) = real(4.0) + real(0.03) * k - real(0.02) * j + real(0.01) * i;
        });
}

void
add_harmonic_drift(Core::State& state,
    const Core::Grid& grid,
    Real q1_covariant_increment,
    Real q2_harmonic_increment,
    bool change_q2) {
    using Core::Geometry::HorizontalLocation;

    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    const auto h1_u =
        grid.geometry().device_view(HorizontalLocation::U).contravariant_to_physical.a11;

    const auto h1_v =
        grid.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a11;

    auto u = state.get_field<3>("u").get_mutable_device_data();
    auto v = state.get_field<3>("v").get_mutable_device_data();

    Kokkos::parallel_for("AddConstraintHarmonicDrift",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u(k, j, i) += q1_covariant_increment / h1_u(j, i);

            if (change_q2) {
                v(k, j, i) += q2_harmonic_increment / h1_v(j, i);
            }
        });
}

void
run_periodic(const std::string& path
#if defined(ENABLE_NCCL)
    ,
    ncclComm_t comm,
    cudaStream_t stream
#endif
) {
    Utils::ConfigurationManager config(path);
    Core::Grid grid(config, MPI_COMM_WORLD);
#if defined(ENABLE_NCCL)
    Core::State state(config, grid, comm, stream);
#else
    Core::State state(config, grid);
#endif

    const auto& geometry =
        static_cast<const Core::Geometry::RegularLatLonGeometry&>(grid.geometry());
    const Real radius = geometry.radius();

    fill_periodic_harmonic(state, grid, real(11.0) * radius, real(-3.0) * radius);
    Kokkos::fence();

    const Values initial_u = snapshot_physical(state.get_field<3>("u"), grid);
    const Values initial_v = snapshot_physical(state.get_field<3>("v"), grid);

    auto constraint = Dynamics::make_regular_lat_lon_circulation_constraint(grid, state);

    // Periodic cycles belong to the incoming physical wind. Capture before
    // recovery, then restore that target after the diagnostic changes it.
    constraint->before_recovery(true);

    add_harmonic_drift(state, grid, real(2.25) * radius, real(0.75) * radius, true);
    constraint->after_recovery(true);
    Kokkos::fence();

    require(close_vectors(snapshot_physical(state.get_field<3>("u"), grid), initial_u),
        "periodic q1 cycle was not restored after initial recovery");
    require(close_vectors(snapshot_physical(state.get_field<3>("v"), grid), initial_v),
        "periodic q2 cycle was not restored after initial recovery");

    constraint->before_recovery(false);
    add_harmonic_drift(state, grid, real(-1.5) * radius, real(0.4) * radius, true);
    constraint->after_recovery(false);
    Kokkos::fence();

    require(close_vectors(snapshot_physical(state.get_field<3>("u"), grid), initial_u),
        "periodic q1 cycle drifted on later recovery");
    require(close_vectors(snapshot_physical(state.get_field<3>("v"), grid), initial_v),
        "periodic q2 cycle drifted on later recovery");
}

void
run_channel(const std::string& path
#if defined(ENABLE_NCCL)
    ,
    ncclComm_t comm,
    cudaStream_t stream
#endif
) {
    Utils::ConfigurationManager config(path);
    Core::Grid grid(config, MPI_COMM_WORLD);
#if defined(ENABLE_NCCL)
    Core::State state(config, grid, comm, stream);
#else
    Core::State state(config, grid);
#endif

    const auto& geometry =
        static_cast<const Core::Geometry::RegularLatLonGeometry&>(grid.geometry());
    const Real radius = geometry.radius();

    fill_channel_state(state, grid, real(7.0) * radius);
    Kokkos::fence();

    auto constraint = Dynamics::make_regular_lat_lon_circulation_constraint(grid, state);

    // A bounded-q2 channel captures the wall circulation AFTER its first
    // recovery. before_recovery(true) must therefore be a no-op.
    constraint->before_recovery(true);

    add_harmonic_drift(state, grid, real(1.25) * radius, real(0.0), false);
    Kokkos::fence();

    const Values first_recovered_u = snapshot_physical(state.get_field<3>("u"), grid);
    const Values first_recovered_v = snapshot_physical(state.get_field<3>("v"), grid);

    constraint->after_recovery(true);
    Kokkos::fence();

    require(close_vectors(snapshot_physical(state.get_field<3>("u"), grid), first_recovered_u),
        "channel initial target was captured before, rather than after, recovery");
    require(close_vectors(snapshot_physical(state.get_field<3>("v"), grid), first_recovered_v),
        "channel constraint modified the q2 wind");

    add_harmonic_drift(state, grid, real(-2.0) * radius, real(0.0), false);
    constraint->after_recovery(false);
    Kokkos::fence();

    require(close_vectors(snapshot_physical(state.get_field<3>("u"), grid), first_recovered_u),
        "channel wall circulation drifted on later recovery");
    require(close_vectors(snapshot_physical(state.get_field<3>("v"), grid), first_recovered_v),
        "channel correction modified the q2 wind");
}

} // namespace

int
main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 1) {
        if (rank == 0) {
            std::cerr << "test_horizontal_wind_topology_constraint requires one MPI rank\n";
        }
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
        const std::string channel = temporary.write("channel", "bounded");
        const std::string periodic = temporary.write("periodic", "periodic");

#if defined(ENABLE_NCCL)
        ncclUniqueId id;
        require(ncclGetUniqueId(&id) == ncclSuccess, "NCCL unique id failed");
        require(ncclCommInitRank(&comm, 1, id, 0) == ncclSuccess, "NCCL initialization failed");
        const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
        run_periodic(periodic, comm, stream);
        run_channel(channel, comm, stream);
#else
        run_periodic(periodic);
        run_channel(channel);
#endif

        std::cout << "PASS: topology constraint lifecycle preserves periodic cycles and "
                     "bounded-q2 wall circulation\n";
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
