#include "core/BoundaryConditionManager.hpp"
#include "core/Grid.hpp"
#include "core/Initializer.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::BoundaryConditionManager;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Core::HorizontalEdgeTopology;
using VVM::Core::Initializer;
using VVM::Core::Parameters;
using VVM::Core::State;
using VVM::Utils::ConfigurationManager;
using Json = nlohmann::json;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

template<typename Function>
void expect_invalid_argument(Function function, const std::string& label) {
    bool rejected = false;

    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    check(rejected, label);
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const std::string pattern = (std::filesystem::temp_directory_path() / "vvm_horizontal_config_XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');

        const char* directory = ::mkdtemp(buffer.data());
        if (!directory) throw std::runtime_error("Cannot create temporary test directory.");
        path_ = directory;
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

struct Communication {
#if defined(ENABLE_NCCL)
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;

    Communication() {
        ncclUniqueId id;
        require(ncclGetUniqueId(&id), "ncclGetUniqueId");
        require(ncclCommInitRank(&comm, 1, id, 0), "ncclCommInitRank");
        stream = Kokkos::Cuda().cuda_stream();
    }

    ~Communication() {
        if (comm) ncclCommDestroy(comm);
    }

    static void require(ncclResult_t status, const char* operation) {
        if (status != ncclSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
        }
    }
#endif
};

void write_json(const std::filesystem::path& path, const Json& value) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot create " + path.string());

    output << value.dump(4) << '\n';
    output.close();

    if (!output) throw std::runtime_error("Cannot write " + path.string());
}

Json make_configuration(bool fixed, int mode) {
    Json config;

    // Keep vertical configuration unchanged to isolate horizontal consumers.
    config["grid"] = {
        {"nz", 4},
        {"dz", 100.0},
        {"dz1", 100.0},
        {"vertical_coordinate_type", "default"}
    };

    if (mode == 0) {
        config["grid"]["nx"] = 4;
        config["grid"]["ny"] = 4;
        config["grid"]["n_halo_cells"] = 2;
        config["grid"]["dx"] = 100.0;
        config["grid"]["dy"] = 100.0;
        config["grid"]["fix_lonlat"] = fixed;
        config["grid"]["boundary_condition"] = {
            {"x", "periodic"}, {"y", "periodic"}
        };
    } else {
        config["grid"]["horizontal"] = {
            {"nx", 4},
            {"ny", 4},
            {"n_halo_cells", 2},
            {"geometry", {
                {"kind", "cartesian"},
                {"dx", 100.0},
                {"dy", 100.0},
                {"fix_lonlat", fixed}
            }},
            {"topology", {
                {"q1", "periodic"},
                {"q2", "periodic"}
            }}
        };

        if (mode == 2) {
            config["grid"]["nx"] = 8;
            config["grid"]["ny"] = 8;
            config["grid"]["n_halo_cells"] = 3;
            config["grid"]["dx"] = 200.0;
            config["grid"]["dy"] = 200.0;
            config["grid"]["fix_lonlat"] = !fixed;
            config["grid"]["boundary_condition"] = {
                {"x", "zero_gradient"}, {"y", "zero_gradient"}
            };
        }
    }

    config["constants"] = {
        {"gravity", 9.81},
        {"Rd", 287.0},
        {"P0", 100000.0},
        {"Cp", 1004.0}
    };

    config["simulation"] = {
        {"dt_s", 1.0},
        {"idealized_test", "2dbubble"}
    };

    config["dynamics"]["solver"] = {
        {"WRXMU", 0.25},
        {"iteration", 4}
    };

    config["dynamics"]["tracers"]["config_probe"] = {
        {"enable", true}
    };

    return config;
}

Real longitude_value(int j, int i) {
    return real(100.0) + real(0.25) * i + real(0.125) * j;
}

Real latitude_value(int j, int i) {
    return real(10.0) + real(0.125) * i + real(0.25) * j;
}

Real tracer_value(int k, int j, int i) {
    return static_cast<Real>(10000 * k + 100 * j + i);
}

void run_case(const std::filesystem::path& path, bool expected_fixed, const Communication& communication) {
    const ConfigurationManager config(path.string());
    const Grid grid(config);
    Parameters parameters(config, grid);

#if defined(ENABLE_NCCL)
    State state(config, parameters, grid, communication.comm, communication.stream);
    HaloExchanger halo(config, grid, communication.comm, communication.stream);
#else
    (void)communication;
    State state(config, parameters, grid);
    HaloExchanger halo(grid);
#endif

    Initializer initializer(config, grid, parameters, state, halo);
    BoundaryConditionManager boundaries(grid);

    const auto& horizontal = grid.horizontal_specification();
    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    check(horizontal.fix_lonlat == expected_fixed, "Resolved fix_lonlat matches the test case.");

    const bool bounded_x = horizontal.topology.q1 == HorizontalEdgeTopology::Bounded;
    const bool bounded_y = horizontal.topology.q2 == HorizontalEdgeTopology::Bounded;

    // This compatibility call must agree with settings already selected by Grid.
    const std::string x_bc = bounded_x ? "zero_gradient" : "periodic";
    const std::string y_bc = bounded_y ? "zero_gradient" : "periodic";
    boundaries.initialize_bc_types(x_bc, y_bc);

    expect_invalid_argument([&]() {
        boundaries.initialize_bc_types(bounded_x ? "periodic" : "zero_gradient", y_bc);
    }, "Reject a contradictory x boundary override.");

    expect_invalid_argument([&]() {
        boundaries.initialize_bc_types(x_bc, bounded_y ? "periodic" : "zero_gradient");
    }, "Reject a contradictory y boundary override.");

    expect_invalid_argument([&]() {
        boundaries.initialize_bc_types("not_a_boundary", y_bc);
    }, "Reject an unknown boundary string.");

    auto& lon_device = state.get_field<2>("lon").get_mutable_device_data();
    auto& lat_device = state.get_field<2>("lat").get_mutable_device_data();
    auto lon_host = Kokkos::create_mirror_view(lon_device);
    auto lat_host = Kokkos::create_mirror_view(lat_device);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            lon_host(j, i) = longitude_value(j, i);
            lat_host(j, i) = latitude_value(j, i);
        }
    }

    Kokkos::deep_copy(lon_device, lon_host);
    Kokkos::deep_copy(lat_device, lat_host);

    initializer.initialize_geographic_coordinates();

    lon_host = state.get_field<2>("lon").get_host_data();
    lat_host = state.get_field<2>("lat").get_host_data();

    bool coordinates_correct = true;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Real expected_lon = expected_fixed ? real(120.95) : longitude_value(j, i);
            const Real expected_lat = expected_fixed ? real(23.458) : latitude_value(j, i);
            coordinates_correct = coordinates_correct && lon_host(j, i) == expected_lon;
            coordinates_correct = coordinates_correct && lat_host(j, i) == expected_lat;
        }
    }

    check(coordinates_correct, "Fixed coordinates are applied, or existing coordinates are preserved.");

    auto& tracer = state.get_field<3>("config_probe");
    auto& tracer_device = tracer.get_mutable_device_data();
    auto tracer_host = Kokkos::create_mirror_view(tracer_device);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                tracer_host(k, j, i) = tracer_value(k, j, i);
            }
        }
    }

    Kokkos::deep_copy(tracer_device, tracer_host);

    // No periodic exchange here: horizontal periodic halos must remain untouched.
    // A bounded singleton direction retains its existing zero-gradient filling.
    boundaries.apply_horizontal_bcs(tracer);
    tracer_host = tracer.get_host_data();

    bool horizontal_correct = true;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int source_j = bounded_y ? std::clamp(j, h, ny - h - 1) : j;
                const int source_i = bounded_x ? std::clamp(i, h, nx - h - 1) : i;
                horizontal_correct = horizontal_correct
                    && tracer_host(k, j, i) == tracer_value(k, source_j, source_i);
            }
        }
    }

    check(horizontal_correct, "Boundary manager follows resolved horizontal topology.");

    initializer.apply_tracer_boundary_conditions();
    tracer_host = tracer.get_host_data();

    bool tracer_correct = true;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int source_k = std::clamp(k, h, nz - h - 1);
                const int source_j = bounded_y ? std::clamp(j, h, ny - h - 1) : j;
                const int source_i = bounded_x ? std::clamp(i, h, nx - h - 1) : i;
                tracer_correct = tracer_correct
                    && tracer_host(k, j, i) == tracer_value(source_k, source_j, source_i);
            }
        }
    }

    check(tracer_correct, "Tracer initialization preserves horizontal policy and fills vertical halos.");
    Kokkos::fence();
}

void check_bounded_guard(const std::filesystem::path& path) {
    const ConfigurationManager config(path.string());

    // Grid construction must remain usable by bounded-domain component tests.
    const Grid grid(config);
    bool rejected = false;

    try {
        BoundaryConditionManager boundaries(grid);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("Bounded horizontal topology") != std::string::npos;
    }

    check(rejected, "Active bounded directions remain rejected by the model boundary manager.");
}

void run_tests(const Communication& communication) {
    TemporaryDirectory temporary;
    const auto path = temporary.path() / "case.json";

    for (bool fixed : {false, true}) {
        for (int mode = 0; mode < 3; ++mode) {
            write_json(path, make_configuration(fixed, mode));
            run_case(path, fixed, communication);
        }
    }

    // Missing structured fix_lonlat must default to false, not use legacy true.
    Json omitted = make_configuration(false, 2);
    omitted["grid"]["horizontal"]["geometry"].erase("fix_lonlat");
    write_json(path, omitted);
    run_case(path, false, communication);

    for (const std::string axis : {"q1", "q2"}) {
        Json bounded = make_configuration(false, 1);
        bounded["grid"]["horizontal"]["topology"][axis] = "bounded";

        write_json(path, bounded);
        check_bounded_guard(path);

        // Only a singleton direction is allowed through the existing guard.
        bounded["grid"]["horizontal"][axis == "q1" ? "nx" : "ny"] = 1;
        write_json(path, bounded);
        run_case(path, false, communication);
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

        Communication communication;
        run_tests(communication);
        Kokkos::fence();
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_horizontal_configuration_consumers: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) {
        std::puts("test_horizontal_configuration_consumers: PASS");
    }

    Kokkos::finalize();
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
