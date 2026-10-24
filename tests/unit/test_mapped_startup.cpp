#include "core/initialization/StartupCollectives.hpp"
#include "core/geometry/HorizontalBoundaryDistances.hpp"
#include "io/RestartVariables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace {
using namespace VVM;

void
require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool
close(Real actual, Real expected) {
    return std::isfinite(actual) &&
           std::abs(actual - expected) <= real(256.0) * std::numeric_limits<Real>::epsilon() *
                                              std::max(real(1.0), std::abs(expected));
}

struct TemporaryConfig {
    std::filesystem::path directory;

    TemporaryConfig() {
        const auto pattern =
            (std::filesystem::temp_directory_path() / "vvm_mapped_startup_XXXXXX").string();
        std::vector<char> path(pattern.begin(), pattern.end());
        path.push_back('\0');
        const char* created = ::mkdtemp(path.data());
        require(created != nullptr, "could not create mapped startup test directory");
        directory = created;
    }

    ~TemporaryConfig() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }

    std::string
    write(const std::string& name, const std::array<Real, 4>& basis) const {
        auto config = nlohmann::json::parse(R"({
            "grid": {
                "horizontal": {
                    "nx": 13, "ny": 11, "n_halo_cells": 2,
                    "geometry": {"kind": "affine", "dq1": 3.0, "dq2": 5.0},
                    "topology": {"q1": "periodic", "q2": "bounded"}
                },
                "vertical": {"nz": 6, "type": "default", "dz": 250.0, "dz1": 250.0}
            },
            "dynamics": {"prognostic_variables": {"xi": {}, "eta": {}, "zeta": {}, "th": {}}}
        })");
        config["grid"]["horizontal"]["geometry"]["basis"] = basis;
        const auto path = directory / (name + ".json");
        std::ofstream output(path);
        output << config.dump(2);
        output.close();
        require(static_cast<bool>(output), "could not write mapped startup test config");
        return path.string();
    }
};

void
check_collectives(const Core::Grid& grid, const Core::State& state, int rank, int ranks) {
    const Real value = Real(rank + 1);
    const Real local[3]{value, real(2.0) * value, real(1.0)};
    Real total[3]{};
    Core::StartupCollectives::sum(grid, state, local, total, 3);
    const Real expected = Real(ranks) * Real(ranks + 1) / real(2.0);
    require(close(total[0], expected) && close(total[1], real(2.0) * expected) &&
                close(total[2], Real(ranks)),
        "startup reduction lost rank contributions");
#if defined(ENABLE_NCCL)
    const auto gathered = Core::StartupCollectives::all_gather(state, {value, real(2.0) * value});
    require(gathered.size() == static_cast<size_t>(2 * ranks), "startup gather has wrong size");
    for (int r = 0; r < ranks; ++r) {
        require(close(gathered[2 * r], Real(r + 1)) &&
                    close(gathered[2 * r + 1], real(2.0) * Real(r + 1)),
            "startup gather changed communicator order");
    }
#endif
}

void
check_distances(const Core::Grid& grid, const Core::State& state) {
    Core::Geometry::HorizontalBoundaryDistances distances(grid, state);
    const auto west = distances.west.get_host_data(), east = distances.east.get_host_data();
    const auto south = distances.south.get_host_data(), north = distances.north.get_host_data();
    const auto metric = grid.geometry().device_view(Core::Geometry::HorizontalLocation::T);
    const Real width =
        grid.geometry().dq1() * metric.sqrt_g.constant / std::sqrt(metric.g_cov.a22.constant);
    const Real height =
        grid.geometry().dq2() * metric.sqrt_g.constant / std::sqrt(metric.g_cov.a11.constant);
    const int h = grid.get_halo_cells();
    for (int j = 0; j < grid.get_local_physical_points_y(); ++j) {
        for (int i = 0; i < grid.get_local_physical_points_x(); ++i) {
            const Real expected_west =
                (Real(grid.get_local_physical_start_x() + i) + real(0.5)) * width;
            const Real expected_south =
                (Real(grid.get_local_physical_start_y() + j) + real(0.5)) * height;
            require(close(west(j + h, i + h), expected_west) &&
                        close(east(j + h, i + h),
                            Real(grid.get_global_points_x()) * width - expected_west) &&
                        close(south(j + h, i + h), expected_south) &&
                        close(north(j + h, i + h),
                            Real(grid.get_global_points_y()) * height - expected_south),
                "mapped boundary distance has an incorrect distributed prefix or total");
        }
    }
}

void
check_restart_contract(const Utils::ConfigurationManager& config,
    const Core::Grid& grid,
    const Core::State& state,
    bool mixed) {
    const auto inferred = IO::infer_restart_variables(config, state);
    const std::array<std::string, 4> canonical{"xi_con", "eta_con", "u_con", "v_con"};
    for (const auto& name : canonical) {
        require(std::find(inferred.vars_3d.begin(), inferred.vars_3d.end(), name) !=
                    inferred.vars_3d.end(),
            "mapped restart inference omitted canonical state");
    }
    auto variables = inferred;
    require(IO::resolve_canonical_restart_variables(variables,
                grid,
                [](const std::string&) {
        return true;
    }),
        "complete canonical checkpoint was not selected");
    require(variables.vars_3d == inferred.vars_3d, "canonical fields were removed from selection");

    variables = inferred;
    bool rejected = false;
    bool restored = false;
    try {
        restored = IO::resolve_canonical_restart_variables(variables, grid, [](const std::string&) {
            return false;
        });
    }
    catch (const std::runtime_error&) {
        rejected = true;
    }
    require(!restored, "legacy checkpoint was marked canonical");
    require(rejected == mixed, "legacy checkpoint compatibility does not match the map");
    if (!mixed) {
        for (const auto& name : canonical) {
            require(std::find(variables.vars_3d.begin(), variables.vars_3d.end(), name) ==
                        variables.vars_3d.end(),
                "diagonal legacy checkpoint still requires unavailable canonical fields");
        }
    }

    variables = inferred;
    rejected = false;
    try {
        IO::resolve_canonical_restart_variables(variables, grid, [](const std::string& name) {
            return name == "xi_con";
        });
    }
    catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "partial canonical checkpoint was accepted");
}
} // namespace

int
main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
#if defined(ENABLE_NCCL)
    ncclComm_t comm = nullptr;
#endif
    try {
#if defined(ENABLE_NCCL)
        ncclUniqueId id;
        if (rank == 0) {
            Core::StartupCollectives::check_nccl(ncclGetUniqueId(&id));
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        Core::StartupCollectives::check_nccl(ncclCommInitRank(&comm, ranks, id, rank));
#endif
        TemporaryConfig temporary;
        for (bool mixed : {false, true}) {
            const std::array<Real, 4> basis{real(2.0),
                mixed ? real(0.5) : real(0.0),
                real(0.0),
                real(1.5)};
            Utils::ConfigurationManager config(
                temporary.write(mixed ? "mixed" : "diagonal", basis));
            Core::Grid grid(config, MPI_COMM_WORLD);
#if defined(ENABLE_NCCL)
            Core::State state(config, grid, comm, Kokkos::Cuda().cuda_stream());
#else
            Core::State state(config, grid);
#endif
            check_collectives(grid, state, rank, ranks);
            check_distances(grid, state);
            check_restart_contract(config, grid, state, mixed);
        }
#if defined(ENABLE_NCCL)
        Core::StartupCollectives::check_nccl(ncclCommDestroy(comm));
#endif
    }
    catch (const std::exception& error) {
        std::cerr << "Rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
