// What VVM_CLEO_Interface promises before CLEO itself is ever touched: the
// constructor registers the fields the rest of the model reads, under P3's
// names, units and staggering, so RRTMGP and land see CLEO's cloud and
// precipitation without knowing which scheme produced them.
//
// The constructor reads no CLEO input file, so neither does this test; the
// coupling itself is covered by test_cleo_coupling.
//
//   test_cleo_interface <cartesian config> [<regular_latlon config>]
//
// With the second argument it also checks that a non-Cartesian grid is
// refused. CLEO is built with CLEO_DOMAIN=cartesian (externals/cleo), so its
// gridboxes are boxes of dx*dy*dz; coupling them to a latitude-longitude grid
// would put every droplet in a cell of the wrong volume, silently.
//
// Variants of the Cartesian config check the other cases CLEO would run
// wrongly without an error: a bounded x or y edge (CLEO wraps superdroplets
// periodically), raised terrain (they would fall into it) and a run that does
// not end on a coupling (CLEO would run past total_time_s).
#include "core/Grid.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "physics/cleo/VVM_cleo_process_interface.hpp"
#include "utils/ConfigurationManager.hpp"
#include "../../externals/json/json.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

int failures = 0;
int rank = 0;

void
fail(const std::string& message) {
    ++failures;
    if (rank == 0) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

void
check(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

// Builds Grid, State and HaloExchanger for `config_path` the way main.cpp does
// for the active communication backend, and hands them to `body`.
void
with_model_state(const char* config_path,
    const std::function<void(const VVM::Utils::ConfigurationManager&,
        const VVM::Core::Grid&,
        const VVM::Core::Parameters&,
        VVM::Core::HaloExchanger&,
        VVM::Core::State&)>& body) {
    VVM::Utils::ConfigurationManager config(config_path);
    VVM::Core::Grid grid(config, MPI_COMM_WORLD);
    VVM::Core::Parameters parameters(config, grid);
#if defined(ENABLE_NCCL)
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, size, id, rank);
    cudaStream_t stream = Kokkos::Cuda().cuda_stream();
    {
        VVM::Core::State state(config, parameters, grid, nccl_comm, stream);
        VVM::Core::HaloExchanger halo(config, grid, nccl_comm, stream);
        body(config, grid, parameters, halo, state);
    }
    ncclCommDestroy(nccl_comm);
#else
    VVM::Core::State state(config, parameters, grid);
    VVM::Core::HaloExchanger halo(grid);
    body(config, grid, parameters, halo, state);
#endif
}

struct ExpectedField {
    const char* name;
    const char* units;
};

// Each of these is read by another process when CLEO replaces P3: qc, nc, qi
// and the effective radii by RRTMGP, the precipitation by land, qp by the
// buoyancy. The units are P3's (VVM_p3_process_interface.cpp).
constexpr ExpectedField centered_fields[] = {
    {"qp", "kg kg-1"},
    {"qc", "kg kg-1"},
    {"qr", "kg kg-1"},
    {"qi", "kg kg-1"},
    {"nc", "kg-1"},
    {"nr", "kg-1"},
    {"diag_eff_radius_qc", "um"},
    {"diag_eff_radius_qr", "um"},
    {"diag_eff_radius_qi", "um"},
};

constexpr ExpectedField surface_fields[] = {
    {"precip_liq_surf_mass", "kg m-2 s-1"},
    {"precip_liq_surf_flux", "kg m-2 s-1"},
    {"precip_ice_surf_mass", "kg m-2 s-1"},
    {"precip_ice_surf_flux", "kg m-2 s-1"},
};

template <size_t N, typename View>
bool
all_equal(const View& host, double value) {
    bool ok = true;
    if constexpr (N == 3) {
        for (size_t k = 0; k < host.extent(0); ++k) {
            for (size_t j = 0; j < host.extent(1); ++j) {
                for (size_t i = 0; i < host.extent(2); ++i) {
                    ok = ok && host(k, j, i) == value;
                }
            }
        }
    }
    else {
        for (size_t j = 0; j < host.extent(0); ++j) {
            for (size_t i = 0; i < host.extent(1); ++i) {
                ok = ok && host(j, i) == value;
            }
        }
    }
    return ok;
}

void
check_registered_fields(const VVM::Utils::ConfigurationManager& config,
    const VVM::Core::Grid& grid,
    const VVM::Core::Parameters& parameters,
    VVM::Core::HaloExchanger& halo,
    VVM::Core::State& state) {
    // A field that already exists belongs to whoever made it: the constructor
    // must add what is missing and leave the rest alone, values included.
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const VVM::Real sentinel = VVM::real(7.0);
    state.add_field<3>("qp",
        {nz, ny, nx},
        VVM::Core::FieldMetadata{VVM::Core::GridStaggering::Centered,
            "kg kg-1",
            "total hydrometeor mass mixing ratio"});
    Kokkos::deep_copy(state.get_field<3>("qp").get_mutable_device_data(), sentinel);

    VVM::Physics::VVM_CLEO_Interface cleo(config, grid, parameters, halo, state);

    for (const auto& expected : centered_fields) {
        if (!state.has_field(expected.name)) {
            fail(std::string("constructor did not register ") + expected.name);
            continue;
        }
        const auto& field = state.get_field<3>(expected.name);
        const auto& data = field.get_device_data();
        check(data.extent(0) == static_cast<size_t>(nz) &&
                  data.extent(1) == static_cast<size_t>(ny) &&
                  data.extent(2) == static_cast<size_t>(nx),
            std::string(expected.name) + " is not (local_total_z, y, x) with halos");
        check(field.get_metadata().grid_staggering == VVM::Core::GridStaggering::Centered,
            std::string(expected.name) + " is not cell-centred");
        check(field.get_metadata().units == expected.units,
            std::string(expected.name) + " has units '" + field.get_metadata().units +
                "', expected '" + expected.units + "' (P3's)");
    }

    for (const auto& expected : surface_fields) {
        if (!state.has_field(expected.name)) {
            fail(std::string("constructor did not register ") + expected.name);
            continue;
        }
        const auto& field = state.get_field<2>(expected.name);
        const auto& data = field.get_device_data();
        check(data.extent(0) == static_cast<size_t>(ny) &&
                  data.extent(1) == static_cast<size_t>(nx),
            std::string(expected.name) + " is not (local_total_y, x) with halos");
        check(field.get_metadata().grid_staggering == VVM::Core::GridStaggering::Surface,
            std::string(expected.name) + " is not a surface field");
        check(field.get_metadata().units == expected.units,
            std::string(expected.name) + " has units '" + field.get_metadata().units +
                "', expected '" + expected.units + "'");
        check(all_equal<2>(field.get_host_data(), 0.0),
            std::string(expected.name) + " does not start at zero");
    }

    if (state.has_field("qp")) {
        check(all_equal<3>(state.get_field<3>("qp").get_host_data(), sentinel),
            "constructor overwrote a qp that already existed");
    }

    // CLEO is warm-phase: no ice, ever. qi and the ice precipitation stay at
    // zero for land and RRTMGP to read.
    if (state.has_field("qi")) {
        check(all_equal<3>(state.get_field<3>("qi").get_host_data(), 0.0),
            "qi is not zero after construction");
    }

    // RRTMGP clamps the ice radius to [5, 140] um, so a zero would read as the
    // smallest, most absorbing ice it can represent. 25 um is what P3 gives an
    // ice-free cell; halos included, since radiation reads whole columns.
    if (state.has_field("diag_eff_radius_qi")) {
        check(all_equal<3>(state.get_field<3>("diag_eff_radius_qi").get_host_data(), 25.0),
            "diag_eff_radius_qi is not 25 um everywhere (P3's ice-free value)");
    }
}

void
check_non_cartesian_refused(const VVM::Utils::ConfigurationManager& config,
    const VVM::Core::Grid& grid,
    const VVM::Core::Parameters& parameters,
    VVM::Core::HaloExchanger& halo,
    VVM::Core::State& state) {
    if (grid.geometry().kind() == VVM::Core::Geometry::GeometryKind::Cartesian) {
        fail("the regular_latlon configuration built a Cartesian grid; wrong config passed?");
        return;
    }
    bool threw = false;
    try {
        VVM::Physics::VVM_CLEO_Interface cleo(config, grid, parameters, halo, state);
    }
    catch (const std::exception& e) {
        threw = true;
        if (rank == 0) {
            std::printf("non-Cartesian grid refused as expected: %s\n", e.what());
        }
    }
    check(threw, "VVM_CLEO_Interface accepted a non-Cartesian grid; CLEO is built Cartesian");
}

// config_path with the value at `pointer` replaced, written to a temporary file.
std::filesystem::path
config_variant(const char* config_path, const std::string& pointer, const nlohmann::json& value) {
    std::ifstream in(config_path);
    nlohmann::json config = nlohmann::json::parse(in);
    config[nlohmann::json::json_pointer(pointer)] = value;
    std::string name = pointer;
    for (auto& ch : name) {
        ch = ch == '/' ? '_' : ch;
    }
    const auto path = std::filesystem::temp_directory_path() /
                      ("test_cleo_interface" + name + "_" + std::to_string(getpid()) + ".json");
    std::ofstream(path) << config.dump(2);
    return path;
}

// Must be refused with a message naming `reason`; any other error would mean
// the check under test never ran. `then` runs after construction.
void
check_refused(const char* config_path,
    const std::string& what,
    const std::string& reason,
    const std::function<void(
        const VVM::Core::Grid&, VVM::Core::State&, VVM::Physics::VVM_CLEO_Interface&)>& then = {}) {
    with_model_state(config_path,
        [&](const VVM::Utils::ConfigurationManager& config,
            const VVM::Core::Grid& grid,
            const VVM::Core::Parameters& parameters,
            VVM::Core::HaloExchanger& halo,
            VVM::Core::State& state) {
        std::string message;
        try {
            VVM::Physics::VVM_CLEO_Interface cleo(config, grid, parameters, halo, state);
            if (then) {
                then(grid, state, cleo);
            }
        }
        catch (const std::exception& e) {
            message = e.what();
        }
        if (message.find(reason) == std::string::npos) {
            fail(what + " was not refused for " + reason +
                 (message.empty() ? "" : "; error was: " + message));
        }
        else if (rank == 0) {
            std::printf("%s refused as expected: %s\n", what.c_str(), message.c_str());
        }
    });
}

void
check_variant_refused(const char* config_path,
    const std::string& pointer,
    const nlohmann::json& value,
    const std::string& reason) {
    const auto path = config_variant(config_path, pointer, value);
    check_refused(path.c_str(), pointer + " = " + value.dump(), reason);
    std::filesystem::remove(path);
}

void
check_refusals(const char* config_path) {
    check_variant_refused(config_path,
        "/grid/boundary_condition/x",
        "zero_gradient",
        "boundary_condition");
    check_variant_refused(config_path,
        "/grid/boundary_condition/y",
        "zero_gradient",
        "boundary_condition");
    // 10.5 s is ten 1 s couplings and half of an eleventh.
    check_variant_refused(config_path, "/simulation/total_time_s", 10.5, "total_time_s");
    // One column's ground raised to the first physical level. CLEO's own
    // inputs do not exist here, so only a check made before reading them can
    // give the expected message.
    check_refused(config_path,
        "raised topo",
        "flat terrain",
        [](const VVM::Core::Grid& grid,
            VVM::Core::State& state,
            VVM::Physics::VVM_CLEO_Interface& cleo) {
        const int h = grid.get_halo_cells();
        auto topo = state.get_field<2>("topo").get_host_data();
        topo(h, h) = h;
        Kokkos::deep_copy(state.get_field<2>("topo").get_mutable_device_data(), topo);
        cleo.initialize(state);
    });
}

} // namespace

int
main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2 && argc != 3) {
        if (rank == 0) {
            std::fprintf(stderr, "usage: %s <cartesian config> [<regular_latlon config>]\n",
                argv[0]);
        }
        MPI_Finalize();
        return 2;
    }

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
    try {
        with_model_state(argv[1], check_registered_fields);
        check_refusals(argv[1]);
        if (argc == 3) {
            with_model_state(argv[2], check_non_cartesian_refused);
        }
    }
    catch (const std::exception& e) {
        fail(std::string("unexpected exception: ") + e.what());
    }
    Kokkos::finalize();

    if (rank == 0 && failures == 0) {
        std::printf("test_cleo_interface: all checks passed\n");
    }
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
