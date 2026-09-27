#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/geometry/HorizontalGeometry.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdio>
#include <string>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

int g_rank = 0;
int g_failures = 0;

void
check(bool condition, const char* message) {
    if (!condition) {
        ++g_failures;
        std::fprintf(stderr, "Rank %d FAIL: %s\n", g_rank, message);
    }
    else if (g_rank == 0) {
        std::fprintf(stdout, "[PASS] %s\n", message);
    }
}

void
check_string(const std::string& actual, const std::string& expected, const char* message) {

    if (actual != expected) {
        ++g_failures;

        std::fprintf(stderr,
            "Rank %d FAIL: %s\n"
            "  got : \"%s\"\n"
            "  want: \"%s\"\n",
            g_rank,
            message,
            actual.c_str(),
            expected.c_str());
    }
    else if (g_rank == 0) {
        std::fprintf(stdout, "[PASS] %s\n", message);
    }
}

void
check_field_shape(const VVM::Core::Field<3>& field, const VVM::Core::Grid& grid, const char* name) {

    const auto& view = field.get_device_data();

    const bool correct = static_cast<int>(view.extent(0)) == grid.get_local_total_points_z() &&
                         static_cast<int>(view.extent(1)) == grid.get_local_total_points_y() &&
                         static_cast<int>(view.extent(2)) == grid.get_local_total_points_x();

    const std::string message = std::string(name) + " has the full local State dimensions";

    check(correct, message.c_str());
}

void
check_field(const VVM::Core::State& state,
    const VVM::Core::Grid& grid,
    const char* name,
    VVM::Core::GridStaggering staggering,
    const std::string& units,
    const std::string& long_name,
    const std::string& comment) {

    {
        const std::string message = std::string("State contains field ") + name;

        check(state.has_field(name), message.c_str());
    }

    if (!state.has_field(name)) {
        return;
    }

    const auto& field = state.get_field<3>(name);

    const auto& metadata = field.get_metadata();

    {
        const std::string message = std::string(name) + " has the expected staggering";

        check(metadata.grid_staggering == staggering, message.c_str());
    }

    {
        const std::string message = std::string(name) + " has the expected units";

        check_string(metadata.units, units, message.c_str());
    }

    {
        const std::string message = std::string(name) + " has the expected long_name";

        check_string(metadata.long_name, long_name, message.c_str());
    }

    {
        const std::string message = std::string(name) + " documents the expected convention";

        check_string(metadata.comment, comment, message.c_str());
    }

    check_field_shape(field, grid, name);
}

void
check_legacy_fields_remain(const VVM::Core::State& state) {

    const char* fields[] = {
        "u",
        "v",
        "w",
        "xi",
        "eta",
        "zeta",
    };

    for (const char* name : fields) {
        const std::string message = std::string("legacy physical field remains available: ") + name;

        check(state.has_field(name), message.c_str());
    }
}

void
run_tests(const VVM::Core::State& state, const VVM::Core::Grid& grid) {

    using VVM::Core::GridStaggering;
    using VVM::Core::Geometry::GeometryKind;

    const GeometryKind kind = grid.geometry().kind();

    const bool cartesian = kind == GeometryKind::Cartesian;

    const bool rll = kind == GeometryKind::RegularLatLon;

    check(cartesian || rll, "geometry is Cartesian or RegularLatLon");

    if (!(cartesian || rll)) {
        return;
    }

    if (g_rank == 0) {
        std::fprintf(stdout,
            "\nState coordinate-field test\n"
            "geometry = %s\n\n",
            cartesian ? "Cartesian" : "RegularLatLon");
    }

    /*
     * q1/q2 are physical distances in Cartesian coordinates,
     * therefore:
     *
     *     u^1, u^2 : m s^-1
     *     omega^1, omega^2 : s^-1
     *
     * q1=lambda and q2=phi in RLL coordinates, therefore:
     *
     *     u^1, u^2 : s^-1
     *     omega^1, omega^2 : m^-1 s^-1
     */
    const std::string velocity_units = rll ? "s-1" : "m s-1";

    const std::string horizontal_vorticity_units = rll ? "m-1 s-1" : "s-1";

    check_legacy_fields_remain(state);

    check_field(state,
        grid,
        "u_con",
        GridStaggering::StaggeredX,
        velocity_units,
        "contravariant q1 wind component at U",
        "Canonical dynamical-core component u^1");

    check_field(state,
        grid,
        "v_con",
        GridStaggering::StaggeredY,
        velocity_units,
        "contravariant q2 wind component at V",
        "Canonical dynamical-core component u^2");

    check_field(state,
        grid,
        "xi_con",
        GridStaggering::StaggeredYZ,
        horizontal_vorticity_units,
        "contravariant q1 relative vorticity at V",
        "VVM convention: xi_con = omega^1");

    check_field(state,
        grid,
        "eta_con",
        GridStaggering::StaggeredXZ,
        horizontal_vorticity_units,
        "negative contravariant q2 relative vorticity at U",
        "VVM convention: eta_con = -omega^2");

    check_field(state,
        grid,
        "zeta_con",
        GridStaggering::StaggeredXY,
        "s-1",
        "contravariant vertical relative vorticity at Z",
        "VVM convention: zeta_con = omega^3");
}

} // namespace

int
main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    if (argc != 2) {
        if (g_rank == 0) {
            std::fprintf(stderr, "usage: %s <config.json>\n", argv[0]);
        }

        MPI_Finalize();
        return 2;
    }

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));

    int exit_code = 0;

    {
        VVM::Utils::ConfigurationManager config(argv[1]);

        VVM::Core::Grid grid(config, MPI_COMM_WORLD);

#if defined(ENABLE_NCCL)

        int mpi_size = 1;

        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        ncclUniqueId nccl_id;

        if (g_rank == 0) {
            const ncclResult_t result = ncclGetUniqueId(&nccl_id);

            if (result != ncclSuccess) {
                std::fprintf(stderr, "ncclGetUniqueId failed: %s\n", ncclGetErrorString(result));

                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

        ncclComm_t nccl_comm;

        const ncclResult_t init_result = ncclCommInitRank(&nccl_comm, mpi_size, nccl_id, g_rank);

        if (init_result != ncclSuccess) {
            std::fprintf(stderr,
                "Rank %d: ncclCommInitRank failed: %s\n",
                g_rank,
                ncclGetErrorString(init_result));

            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        cudaStream_t stream = Kokkos::Cuda().cuda_stream();

        /*
         * Deliberately do NOT construct Parameters here.
         *
         * State field registration depends only on ConfigurationManager
         * and Grid.  This is important because Parameters still exposes
         * Cartesian dx/dy and intentionally rejects RLL construction.
         */
        VVM::Core::State state(config, grid, nccl_comm, stream);

        run_tests(state, grid);

        ncclCommDestroy(nccl_comm);

#else

        /*
         * Deliberately do NOT construct Parameters here.
         */
        VVM::Core::State state(config, grid);

        run_tests(state, grid);

#endif

        int global_failures = 0;

        MPI_Allreduce(&g_failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (g_rank == 0) {
            std::fprintf(stdout,
                "\n%s: %d failure(s)\n",
                global_failures == 0 ? "OK" : "FAILED",
                global_failures);
        }

        exit_code = global_failures == 0 ? 0 : 1;
    }

    Kokkos::finalize();

    MPI_Finalize();

    return exit_code;
}
