#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/HorizontalVectorConversion.hpp"
#include "dynamics/solvers/HorizontalWindTopologyConstraint.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
require(const bool condition, const std::string& message) {

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

        Json config = Json::parse(
            R"({
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
                      "topology": {
                        "q1": "periodic",
                        "q2": "bounded"
                      }
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
snapshot_recovered_domain(const Core::Field<3>& field, const Core::Grid& grid) {

    const auto data = field.get_host_data();

    const int h = grid.get_halo_cells();

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int bottom = h - 1;

    const int top = nz - h - 1;

    Values result;

    result.reserve(static_cast<std::size_t>(top - bottom + 1) *
                   static_cast<std::size_t>(ny - 2 * h) * static_cast<std::size_t>(nx - 2 * h));

    for (int k = bottom; k <= top; ++k) {

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
    Core::State& state, const Core::Grid& grid, const Real q1_covariant, const Real q2_harmonic) {

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
            u(k, j, i) = q1_covariant / h1_u(j, i);

            v(k, j, i) = q2_harmonic / h1_v(j, i);
        });
}

void
fill_channel_state(Core::State& state, const Core::Grid& grid, const Real q1_covariant) {

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

            v(k, j, i) = real(4.0) + real(0.03) * static_cast<Real>(k) -
                         real(0.02) * static_cast<Real>(j) + real(0.01) * static_cast<Real>(i);
        });
}

void
populate_covariant_wind_from_physical(const Core::Grid& grid,
    Core::State& state,
    Core::Field<3>& covariant_q1_wind,
    Core::Field<3>& covariant_q2_wind) {

    using Core::Geometry::HorizontalLocation;
    using Dynamics::Operators::HorizontalVectorConversion;

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const auto u = state.get_field<3>("u").get_device_data();

    const auto v = state.get_field<3>("v").get_device_data();

    const auto h1_at_u =
        grid.geometry().device_view(HorizontalLocation::U).contravariant_to_physical.a11;

    const auto h2_at_v =
        grid.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a22;

    auto q1_cov = covariant_q1_wind.get_mutable_device_data();

    auto q2_cov = covariant_q2_wind.get_mutable_device_data();

    Kokkos::parallel_for("PopulateTopologyConstraintCovariantWind",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            q1_cov(k, j, i) =
                HorizontalVectorConversion::physical_to_covariant(u(k, j, i), h1_at_u(j, i));

            q2_cov(k, j, i) =
                HorizontalVectorConversion::physical_to_covariant(v(k, j, i), h2_at_v(j, i));
        });
}

void
add_covariant_harmonic_drift(const Core::Grid& grid,
    Core::Field<3>& covariant_q1_wind,
    Core::Field<3>& covariant_q2_wind,
    const Real q1_covariant_increment,
    const Real q2_harmonic_increment,
    const bool change_q2) {

    using Core::Geometry::HorizontalLocation;

    const int h = grid.get_halo_cells();

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    const int bottom = h - 1;

    const int top = nz - h - 1;

    const auto h1_at_v =
        grid.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a11;

    const auto h2_at_v =
        grid.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a22;

    auto q1_cov = covariant_q1_wind.get_mutable_device_data();

    auto q2_cov = covariant_q2_wind.get_mutable_device_data();

    Kokkos::parallel_for("AddTopologyConstraintCovariantDrift",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            q1_cov(k, j, i) += q1_covariant_increment;

            if (change_q2) {

                // The historical periodic harmonic is expressed in physical
                // meridional wind as
                //
                //     delta V = H / h1.
                //
                // Since the topology constraint now operates on the
                // covariant component,
                //
                //     u_2 = h2 * V,
                //
                // the corresponding covariant harmonic is
                //
                //     delta u_2 = H * h2 / h1.
                q2_cov(k, j, i) += q2_harmonic_increment * h2_at_v(j, i) / h1_at_v(j, i);
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

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    Core::Field<3> covariant_q1_wind("test_periodic_covariant_q1_wind", {nz, ny, nx});

    Core::Field<3> covariant_q2_wind("test_periodic_covariant_q2_wind", {nz, ny, nx});

    fill_periodic_harmonic(state, grid, real(11.0) * radius, real(-3.0) * radius);

    populate_covariant_wind_from_physical(grid, state, covariant_q1_wind, covariant_q2_wind);

    Kokkos::fence();

    const Values target_q1 = snapshot_recovered_domain(covariant_q1_wind, grid);

    const Values target_q2 = snapshot_recovered_domain(covariant_q2_wind, grid);

    auto constraint = Dynamics::make_regular_lat_lon_circulation_constraint(grid,
        state,
        covariant_q1_wind,
        covariant_q2_wind);

    // For a fully periodic RLL topology the target cycles belong to the
    // incoming physical model state. Capture those targets before the
    // generalized wind diagnostic replaces the recovered wind.
    constraint->before_recovery(true);

    // Emulate a recovered generalized-coordinate wind containing q1 and q2
    // harmonic drift.
    add_covariant_harmonic_drift(grid,
        covariant_q1_wind,
        covariant_q2_wind,
        real(2.25) * radius,
        real(0.75) * radius,
        true);

    constraint->after_recovery(true);

    Kokkos::fence();

    require(close_vectors(snapshot_recovered_domain(covariant_q1_wind, grid), target_q1),
        "periodic covariant q1 cycle was not restored after initial recovery");

    require(close_vectors(snapshot_recovered_domain(covariant_q2_wind, grid), target_q2),
        "periodic covariant q2 cycle was not restored after initial recovery");

    constraint->before_recovery(false);

    // A later diagnostic may drift again. The originally captured target must
    // still be enforced.
    add_covariant_harmonic_drift(grid,
        covariant_q1_wind,
        covariant_q2_wind,
        real(-1.5) * radius,
        real(0.4) * radius,
        true);

    constraint->after_recovery(false);

    Kokkos::fence();

    require(close_vectors(snapshot_recovered_domain(covariant_q1_wind, grid), target_q1),
        "periodic covariant q1 cycle drifted on later recovery");

    require(close_vectors(snapshot_recovered_domain(covariant_q2_wind, grid), target_q2),
        "periodic covariant q2 cycle drifted on later recovery");
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

    const int nz = grid.get_local_total_points_z();

    const int ny = grid.get_local_total_points_y();

    const int nx = grid.get_local_total_points_x();

    Core::Field<3> covariant_q1_wind("test_channel_covariant_q1_wind", {nz, ny, nx});

    Core::Field<3> covariant_q2_wind("test_channel_covariant_q2_wind", {nz, ny, nx});

    fill_channel_state(state, grid, real(7.0) * radius);

    populate_covariant_wind_from_physical(grid, state, covariant_q1_wind, covariant_q2_wind);

    Kokkos::fence();

    auto constraint = Dynamics::make_regular_lat_lon_circulation_constraint(grid,
        state,
        covariant_q1_wind,
        covariant_q2_wind);

    // A bounded-q2 channel captures the wall circulation from the first
    // recovered covariant wind. before_recovery(true) must therefore remain
    // a no-op.
    constraint->before_recovery(true);

    add_covariant_harmonic_drift(grid,
        covariant_q1_wind,
        covariant_q2_wind,
        real(1.25) * radius,
        real(0.0),
        false);

    Kokkos::fence();

    const Values first_recovered_q1 = snapshot_recovered_domain(covariant_q1_wind, grid);

    const Values first_recovered_q2 = snapshot_recovered_domain(covariant_q2_wind, grid);

    // The first bounded-q2 recovery defines the target. Therefore its
    // correction is zero and the recovered covariant wind must remain
    // unchanged.
    constraint->after_recovery(true);

    Kokkos::fence();

    require(close_vectors(snapshot_recovered_domain(covariant_q1_wind, grid), first_recovered_q1),
        "channel initial target was captured before, rather than after, "
        "the first covariant recovery");

    require(close_vectors(snapshot_recovered_domain(covariant_q2_wind, grid), first_recovered_q2),
        "channel initial constraint modified covariant q2 wind");

    // Introduce later q1 circulation drift. The constraint must restore the
    // first recovered q1 cycle while leaving q2 untouched.
    add_covariant_harmonic_drift(grid,
        covariant_q1_wind,
        covariant_q2_wind,
        real(-2.0) * radius,
        real(0.0),
        false);

    constraint->after_recovery(false);

    Kokkos::fence();

    require(close_vectors(snapshot_recovered_domain(covariant_q1_wind, grid), first_recovered_q1),
        "channel covariant wall circulation drifted on later recovery");

    require(close_vectors(snapshot_recovered_domain(covariant_q2_wind, grid), first_recovered_q2),
        "channel correction modified covariant q2 wind");
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

            std::cerr << "test_horizontal_wind_topology_constraint "
                         "requires one MPI rank\n";
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

        std::cout << "PASS: topology constraint lifecycle preserves "
                     "periodic covariant cycles and bounded-q2 wall "
                     "circulation\n";
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
