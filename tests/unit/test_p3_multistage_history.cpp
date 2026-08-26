#include "core/BoundaryConditionManager.hpp"
#include "core/Grid.hpp"
#include "core/HaloExchanger.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "dynamics/DynamicalCore.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdio>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
        if (rank == 0) std::fprintf(stderr, "usage: %s <config.json>\n", argv[0]);
        MPI_Finalize();
        return 2;
    }

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
    int exit_code = 0;
    {
        VVM::Utils::ConfigurationManager config(argv[1]);
        VVM::Core::Grid grid(config, MPI_COMM_WORLD);
        VVM::Core::Parameters parameters(config, grid);

#if defined(ENABLE_NCCL)
        ncclUniqueId id;
        if (rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclComm_t nccl_comm;
        ncclCommInitRank(&nccl_comm, 1, id, rank);
        cudaStream_t stream = Kokkos::Cuda().cuda_stream();
        VVM::Core::State state(config, parameters, grid, nccl_comm, stream);
        VVM::Core::HaloExchanger halo(config, grid, nccl_comm, stream);
#else
        VVM::Core::State state(config, parameters, grid);
        VVM::Core::HaloExchanger halo(grid);
#endif

        VVM::Core::BoundaryConditionManager boundary_conditions(grid);
        boundary_conditions.initialize_bc_types("periodic", "periodic");
        VVM::Dynamics::DynamicalCore dynamical_core(
            config, grid, parameters, state, halo, boundary_conditions);

        if (!state.has_field("th_m") || !state.has_field("qv_m")) {
            if (rank == 0) {
                std::fprintf(stderr,
                             "P3 with MUSCL/SSPRK2 did not allocate th_m and qv_m\n");
            }
            exit_code = 1;
        }

        if (exit_code == 0) {
            parameters.max_topo_idx = grid.get_halo_cells() - 1;

            auto fill_state_field = [&](const char* name, VVM::Real value) {
                Kokkos::deep_copy(
                    Kokkos::DefaultExecutionSpace(),
                    state.get_field<3>(name).get_mutable_device_data(), value);
            };
            fill_state_field("th", VVM::real(300.0));
            fill_state_field("u", VVM::real(0.0));
            fill_state_field("v", VVM::real(0.0));
            fill_state_field("w", VVM::real(0.0));
            fill_state_field("ITYPEU", VVM::real(1.0));
            fill_state_field("ITYPEV", VVM::real(1.0));
            fill_state_field("ITYPEW", VVM::real(1.0));
            Kokkos::deep_copy(
                Kokkos::DefaultExecutionSpace(),
                state.get_field<1>("rhobar").get_mutable_device_data(),
                VVM::real(1.0));
            Kokkos::deep_copy(
                Kokkos::DefaultExecutionSpace(),
                state.get_field<1>("rhobar_up").get_mutable_device_data(),
                VVM::real(1.0));

            auto check_qv_history = [&](VVM::Real pre_advection) {
                Kokkos::deep_copy(
                    Kokkos::DefaultExecutionSpace(),
                    state.get_field<3>("qv").get_mutable_device_data(),
                    pre_advection);
                dynamical_core.update_thermodynamics(VVM::real(1.0));

                const auto history = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),
                    state.get_field<3>("qv_m").get_device_data());
                const auto current = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),
                    state.get_field<3>("qv").get_device_data());
                const int h = grid.get_halo_cells();
                return history(h, h, h) == pre_advection &&
                       current(h, h, h) == pre_advection;
            };

            const VVM::Real first = VVM::real(0.01);
            const VVM::Real second = VVM::real(0.02);
            if (!check_qv_history(first) || !check_qv_history(second)) {
                if (rank == 0) {
                    std::fprintf(
                        stderr,
                        "SSPRK2 did not refresh qv_m from pre-advection qv\n");
                }
                exit_code = 1;
            }
        }

#if defined(ENABLE_NCCL)
        ncclCommDestroy(nccl_comm);
#endif
    }
    Kokkos::finalize();
    MPI_Finalize();
    return exit_code;
}
