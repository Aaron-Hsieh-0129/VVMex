#ifndef VVM_TYPES_HPP
#define VVM_TYPES_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>

#if defined(ENABLE_NCCL)
#include <nccl.h>
#endif

namespace VVM {

#ifdef VVM_USE_DOUBLE_PRECISION
    using Real = double;
    #define VVM_MPI_REAL MPI_DOUBLE
#else
    using Real = float;
    #define VVM_MPI_REAL MPI_FLOAT
#endif

#if defined(ENABLE_NCCL)
    #ifdef VVM_USE_DOUBLE_PRECISION
        #define VVM_NCCL_REAL ncclDouble
    #else
        #define VVM_NCCL_REAL ncclFloat
    #endif
#endif

template<typename T>
KOKKOS_INLINE_FUNCTION
constexpr Real real(const T val) {
    return static_cast<Real>(val);
}

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

} // namespace VVM
#endif
