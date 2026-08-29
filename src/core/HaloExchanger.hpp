#ifndef VVM_CORE_HALOEXCHANGER_HPP
#define VVM_CORE_HALOEXCHANGER_HPP

// Keep this public include stable while each transport owns its implementation.
#if defined(ENABLE_NCCL)
#include "HaloExchangerNCCL.hpp"
#else
#include "HaloExchangerMPI.hpp"
#endif

#endif // VVM_CORE_HALOEXCHANGER_HPP
