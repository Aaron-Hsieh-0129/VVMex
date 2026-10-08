#pragma once

#include "utils/ConfigurationManager.hpp"
#include <mpi.h>
#if defined(ENABLE_NCCL)
#include <nccl.h>
#include <cuda_runtime.h>
#endif

namespace VVM::Driver {

// Both model lifetimes are contained within the application's existing
// MPI/Kokkos lifetime. No extra initialization or distributed redistribution.
void run_one_way_coupled(const Utils::ConfigurationManager& config, MPI_Comm comm
#if defined(ENABLE_NCCL)
    , ncclComm_t nccl_comm, cudaStream_t stream
#endif
);

} // namespace VVM::Driver
