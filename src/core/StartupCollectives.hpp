#ifndef VVM_CORE_STARTUP_COLLECTIVES_HPP
#define VVM_CORE_STARTUP_COLLECTIVES_HPP

#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>
#include "core/State.hpp"

namespace VVM::Core::StartupCollectives {

// Host-facing collectives for initialization only. NCCL owns the data exchange
// in GPU builds; the blocking copies keep temporary storage alive until done.
#if defined(ENABLE_NCCL)
inline void
check_nccl(ncclResult_t result) {
    if (result != ncclSuccess) {
        throw std::runtime_error(
            std::string("Startup NCCL collective: ") + ncclGetErrorString(result));
    }
}

inline void
finish(cudaStream_t stream) {
    const auto result = cudaStreamSynchronize(stream);
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("Startup NCCL synchronization: ") + cudaGetErrorString(result));
    }
}
#endif

inline void
sum(const Grid& grid, const State& state, const Real* local, Real* global, int count) {
#if defined(ENABLE_NCCL)
    (void)grid;
    Kokkos::View<Real*> send("startup sum input", count), receive("startup sum output", count);
    Kokkos::deep_copy(send, Kokkos::View<const Real*, Kokkos::HostSpace>(local, count));
    check_nccl(ncclAllReduce(send.data(),
        receive.data(),
        count,
        VVM_NCCL_REAL,
        ncclSum,
        state.get_nccl_comm(),
        state.get_cuda_stream()));
    finish(state.get_cuda_stream());
    Kokkos::deep_copy(Kokkos::View<Real*, Kokkos::HostSpace>(global, count), receive);
#else
    (void)state;
    const int result = MPI_Allreduce(local, global, count, VVM_MPI_REAL, MPI_SUM, grid.get_comm());
    if (result != MPI_SUCCESS) {
        throw std::runtime_error("Startup MPI_Allreduce failed.");
    }
#endif
}

#if defined(ENABLE_NCCL)
inline std::vector<Real>
all_gather(const State& state, const std::vector<Real>& local) {
    int ranks = 0;
    check_nccl(ncclCommCount(state.get_nccl_comm(), &ranks));
    const size_t count = local.size();
    Kokkos::View<Real*> send("startup gather input", count),
        receive("startup gather output", count * ranks);
    Kokkos::deep_copy(send, Kokkos::View<const Real*, Kokkos::HostSpace>(local.data(), count));
    check_nccl(ncclAllGather(send.data(),
        receive.data(),
        count,
        VVM_NCCL_REAL,
        state.get_nccl_comm(),
        state.get_cuda_stream()));
    finish(state.get_cuda_stream());
    std::vector<Real> global(count * ranks);
    Kokkos::deep_copy(Kokkos::View<Real*, Kokkos::HostSpace>(global.data(), global.size()),
        receive);
    return global;
}
#endif

} // namespace VVM::Core::StartupCollectives
#endif
