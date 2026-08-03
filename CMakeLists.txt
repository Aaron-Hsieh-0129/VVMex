cmake_minimum_required(VERSION 3.20)

project(VVMex LANGUAGES CXX C Fortran)

cmake_policy(SET CMP0135 NEW)

if(NOT DEFINED ENV{VVM_ROOT})
    message(FATAL_ERROR "
    [Error] VVM_ROOT environment variable is not set!
    Please set the project root directory path before running CMake.
    Example: export VVM_ROOT=/path/to/VVMex
    Please ensure you have sourced your env_setup.sh file.
    ")
endif()

message(STATUS "Detected VVM_ROOT: $ENV{VVM_ROOT}")
add_compile_definitions(VVM_ROOT_DIR="$ENV{VVM_ROOT}")

# OFF builds the whole model for the CPU: Kokkos falls back to OpenMP, the MPI
# communication path replaces NCCL, and the land model's OpenACC directives
# degrade to comments. Every GPU-specific block below is gated on this, so the
# default ON build is byte-for-byte the same as before this option existed.
option(VVM_ENABLE_GPU "Build for GPU (Kokkos CUDA + NCCL). OFF builds a CPU-only model." ON)
if(VVM_ENABLE_GPU)
    message(STATUS "VVM Execution Backend: GPU (Kokkos CUDA)")
else()
    message(STATUS "VVM Execution Backend: CPU (Kokkos OpenMP)")
endif()


# ==========================================
# Auto-setup Git Hooks
# Automatically configures Git to use .githooks/ when a developer runs CMake
# ==========================================
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} config core.hooksPath .githooks
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_HOOK_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(GIT_HOOK_RESULT EQUAL 0)
        message(STATUS "Successfully configured Git to use '.githooks/' directory")
    else()
        message(WARNING "Failed to configure Git hooks automatically")
    endif()
endif()


# ------------------------------
# Compiler-specific settings
# ------------------------------
# Add a flag to a FORCEd cache variable exactly once.
#
# These are FORCE cache writes that read their own previous value, so a plain
# "${VAR} <flag>" append duplicates the flag on every reconfigure. That is not
# cosmetic: it changes the compile line, which forces a full rebuild of every
# object each time the project is reconfigured. Stripping any existing copies
# first makes this idempotent and also repairs a cache that already accumulated
# duplicates from earlier configures.
macro(vvm_set_flag_once VAR FLAG DOC)
    string(REPLACE "${FLAG}" "" _vvm_flags "${${VAR}}")
    string(REGEX REPLACE "  +" " " _vvm_flags "${_vvm_flags}")
    string(STRIP "${_vvm_flags}" _vvm_flags)
    set(${VAR} "${_vvm_flags} ${FLAG}" CACHE STRING "${DOC}" FORCE)
endmacro()

if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(STATUS "Applying NVHPC dependency flag workaround.")
    # Disable automatic dependency generation flags that NVHPC might set
    set(CMAKE_CXX_DEPENDS "")
    set(CMAKE_C_DEPENDS "")
    set(CMAKE_DEPFILE_FLAGS_CXX "")
    set(CMAKE_DEPFILE_FLAGS_C "")

    if(VVM_ENABLE_GPU)
        vvm_set_flag_once(CMAKE_CUDA_FLAGS "-Xcompiler -mp"  "CUDA OpenMP flags")
        vvm_set_flag_once(CMAKE_CUDA_FLAGS "-use_fast_math"  "CUDA flags")
    endif()
    vvm_set_flag_once(CMAKE_CXX_FLAGS  "-fast"           "CXX flags")
    vvm_set_flag_once(CMAKE_C_FLAGS    "-fast"           "C flags")
endif()

option(VVM_USE_DOUBLE_PRECISION "Use double precision (double) instead of single precision (float)" ON)
if(VVM_USE_DOUBLE_PRECISION)
    message(STATUS "VVM Precision: DOUBLE (FP64)")
    add_compile_definitions(VVM_USE_DOUBLE_PRECISION)
else()
    message(STATUS "VVM Precision: SINGLE (FP32)")
endif()

option(SCREAM_DOUBLE_PRECISION "Use double precision (double) instead of single precision (float)" ON)
if(SCREAM_DOUBLE_PRECISION)
    message(STATUS "P3 Precision: DOUBLE (FP64)")
    add_compile_definitions(SCREAM_DOUBLE_PRECISION)
else()
    message(STATUS "P3 Precision: SINGLE (FP32)")
endif()

option(RRTMGP_USE_DOUBLE_PRECISION "Use double precision inside the VVM RRTMGP radiation interface" ON)
if(RRTMGP_USE_DOUBLE_PRECISION)
    message(STATUS "RRTMGP Internal Precision: DOUBLE (FP64)")
    add_compile_definitions(VVM_RRTMGP_USE_DOUBLE_PRECISION)
else()
    message(STATUS "RRTMGP Internal Precision: SINGLE (FP32)")
endif()
# EAMXX_ENABLE_GPU selects between pulling the P3 / physics_functions impl headers
# into every translation unit (GPU) and linking the ETI objects (CPU). The compile
# definition and the option read by src/physics/p3 must agree, or P3 compiles
# against headers whose symbols are never instantiated.
if(VVM_ENABLE_GPU)
    add_compile_definitions(EAMXX_ENABLE_GPU)
else()
    set(EAMXX_ENABLE_GPU OFF CACHE BOOL "Enable GPU kernels" FORCE)
endif()
add_compile_definitions(RRTMGP_ENABLE_KOKKOS)
option(ENABLE_NCCL "Enable NCCL for communication" ${VVM_ENABLE_GPU})
if(ENABLE_NCCL AND NOT VVM_ENABLE_GPU)
    message(FATAL_ERROR "ENABLE_NCCL=ON requires VVM_ENABLE_GPU=ON (NCCL has no CPU path).")
endif()
if(ENABLE_NCCL)
    add_compile_definitions(-DENABLE_NCCL)
    message(STATUS "Building with NCCL support")
else()
    message(STATUS "Building with Standard MPI support")
endif()




# ------------------------------
# Standard C++ settings
# ------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(VVM_ENABLE_GPU)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# ------------------------------
# Kokkos
# ------------------------------
if(VVM_ENABLE_GPU)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend for Kokkos" FORCE)
    set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Enable Kokkos RDC")
else()
    set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA backend for Kokkos" FORCE)
    set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OFF CACHE BOOL "Enable Kokkos RDC" FORCE)
endif()
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend for Kokkos" FORCE)

# NVHPC + Kokkos CUDA arch
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" AND VVM_ENABLE_GPU)
    set(Kokkos_CUDA_OPTIONS "enable_lambda;enable_ldg" CACHE STRING "Kokkos CUDA options" FORCE)
endif()

option(BUILD_TESTS "Build the unit and integration tests" OFF)
message(STATUS "Build tests: ${BUILD_TESTS}")

# ------------------------------
# Fortran flags
# ------------------------------
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -Wno-line-truncation -Wno-argument-mismatch")
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mallocatable=03 -Mdclchk -Mfreeform -Mextend -fast")
endif()

# ------------------------------
# MPI
# ------------------------------
set(MPI_Fortran_FOUND TRUE)
set(MPI_Fortran_WORKS TRUE)
find_package(MPI REQUIRED COMPONENTS CXX C Fortran)
if (MPI_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
else()
    message(FATAL_ERROR "MPI not found.")
endif()

# Repair a cache poisoned by an older revision of this file, which blanked the
# OpenMP flag variables with CACHE ... FORCE further down. find_package(Kokkos)
# below calls find_dependency(OpenMP), and FindOpenMP fails outright on a blank
# cached value -- so such a tree could be configured exactly once and never again.
# Dropping the empty entries lets FindOpenMP re-detect them.
foreach(_vvm_omp_lang CXX C Fortran)
    if(DEFINED CACHE{OpenMP_${_vvm_omp_lang}_FLAGS}
       AND "$CACHE{OpenMP_${_vvm_omp_lang}_FLAGS}" STREQUAL "")
        message(STATUS "Clearing blank cached OpenMP_${_vvm_omp_lang}_FLAGS so it can be re-detected")
        unset(OpenMP_${_vvm_omp_lang}_FLAGS CACHE)
    endif()
endforeach()

# ------------------------------
# Kokkos
# ------------------------------
find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
    message(STATUS "Kokkos not found externally. Fetching via FetchContent.")
    include(FetchContent)
    FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.5.01.tar.gz
    )
    FetchContent_MakeAvailable(Kokkos)
endif()

# ------------------------------
# NVHPC
# ------------------------------
find_library(NVCPUMATH_LIBRARY nvcpumath
    HINTS ${NVHPC_DIR}/compilers/lib
)
if(NOT NVCPUMATH_LIBRARY)
    message(FATAL_ERROR "Could not find NVIDIA CPU Math Library (libnvcpumath.so)")
else()
    message(STATUS "Found NVIDIA CPU Math Library: ${NVCPUMATH_LIBRARY}")
endif()

# find_library(NVTOOLSEXT_LIBRARY nvToolsExt
#     HINTS ${NVHPC_DIR}/cuda/lib64
# )
# if(NOT NVTOOLSEXT_LIBRARY)
#     message(FATAL_ERROR "Could not find NVIDIA Tools Extension Library (libnvToolsExt.so)")
# else()
#     message(STATUS "Found NVIDIA Tools Extension Library: ${NVTOOLSEXT_LIBRARY}")
# endif()

if(ENABLE_NCCL)
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS 
        ${NVHPC_DIR}/comm_libs/nccl/include
    )
    find_library(NCCL_LIBRARY nccl
        HINTS 
        ${NVHPC_DIR}/comm_libs/nccl/lib
    )
    if (NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        message(STATUS "Found NCCL: ${NCCL_INCLUDE_DIR}, ${NCCL_LIBRARY}")
    else()
        message(FATAL_ERROR "Could not find NCCL. Please check hints in Root CMakeLists.txt")
    endif()
endif()


# ------------------------------
# NetCDF-C
# ------------------------------
find_path(NETCDF_C_INCLUDE_DIR netcdf.h
          HINTS ${NETCDF_C_DIR}/include
          NO_DEFAULT_PATH)
find_library(NETCDF_C_LIBRARY NAMES netcdf
             HINTS ${NETCDF_C_DIR}/lib
             NO_DEFAULT_PATH)

if (NETCDF_C_INCLUDE_DIR AND NETCDF_C_LIBRARY)
    add_library(NetCDF::netcdf UNKNOWN IMPORTED GLOBAL)
    set_target_properties(NetCDF::netcdf PROPERTIES
                          IMPORTED_LOCATION "${NETCDF_C_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_C_INCLUDE_DIR}")
    set(NETCDF_C_FOUND TRUE)
    message(STATUS "NetCDF-C (for fallback) found manually at: ${NETCDF_C_LIBRARY}")
else()
    message(FATAL_ERROR "NetCDF-C library or includes not found at the specified path: ${NETCDF_C_INSTALL_DIR}")
endif()

# ------------------------------
# NetCDF-Fortran
# ------------------------------
find_path(NETCDF_FORTRAN_INCLUDE_DIR NAMES netcdf.mod HINTS "${NETCDF_Fortran_DIR}/include" NO_DEFAULT_PATH)
find_library(NETCDF_FORTRAN_LIBRARY NAMES netcdff HINTS "${NETCDF_Fortran_DIR}/lib" NO_DEFAULT_PATH)

if (NOT TARGET NetCDF::NetCDF_Fortran)
    add_library(NetCDF::NetCDF_Fortran UNKNOWN IMPORTED)
    set_property(TARGET NetCDF::NetCDF_Fortran PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_FORTRAN_INCLUDE_DIR}"
        IMPORTED_LOCATION "${NETCDF_FORTRAN_LIBRARY}"
        INTERFACE_LINK_LIBRARIES NetCDF::NetCDF_C)
endif()

# ------------------------------
# EKAT
# ------------------------------
set (EKAT_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/externals/ekat/cmake)
list(APPEND CMAKE_MODULE_PATH
     ${EKAT_CMAKE_PATH}
     ${EKAT_CMAKE_PATH}/tpls
)
set (EKAT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals/ekat)
set(EKAT_ENABLE_ALL_PACKAGES ON)
set(EKAT_DISABLE_TPL_WARNINGS ON)
set (EKAT_ENABLE_TESTS OFF)
set(EKAT_ENABLE_EXAMPLES OFF CACHE BOOL "")
add_subdirectory(${EKAT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/externals/ekat)

include(EkatSetCompilerFlags)
include(EkatMpiUtils)

set(EKAT_ENABLE_NETCDF ON CACHE BOOL "")
set(EKAT_ENABLE_PNETCDF ON CACHE BOOL "")
set(EKAT_ENABLE_YAML ON CACHE BOOL "")
set(EKAT_ENABLE_CUDA ${Kokkos_ENABLE_CUDA} CACHE BOOL "Pass VVM CUDA setting to Ekat" FORCE)
set(Kokkos_DIR ${Kokkos_DIR} CACHE PATH "FORCE Ekat to use VVM's Kokkos" FORCE)


message(STATUS "Aggressively cleaning global OpenMP flags set by subdirectories (e.g., EKAT)...")
# 1. Use REGEX to remove '-mp' as a standalone word (handles spaces correctly)
string(REGEX REPLACE " -mp( |$)" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE " -mp( |$)" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REGEX REPLACE " -mp( |$)" " " CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}")
# 2. Handle if '-mp' is the *first* flag
string(REGEX REPLACE "^-mp( |$)" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "^-mp( |$)" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REGEX REPLACE "^-mp( |$)" " " CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}")
# 3. Mask the OpenMP flag variables that ekat might have set.
#
# These are deliberately *normal* variables, not cache writes. A normal variable
# shadows the cache entry of the same name for the rest of this directory and every
# subdirectory added below, which is all the masking this needs.
#
# Writing "" into the cache instead breaks the *next* configure: find_package(Kokkos)
# runs earlier in this file, KokkosConfig.cmake calls find_dependency(OpenMP), and
# FindOpenMP reads the blanked cache entry back and fails with
#   Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS)
# leaving the build tree configurable exactly once. Worse, `cmake --build` then
# reports success while building nothing, so a stale binary silently survives.
set(OpenMP_CXX_FLAGS "")
set(OpenMP_C_FLAGS "")
set(OpenMP_Fortran_FLAGS "")

function(CLEAN_FLAG_VAR var_name flag_to_remove)
    string(REPLACE " " ";" flag_list "${${var_name}}")
    list(REMOVE_ITEM flag_list "${flag_to_remove}")
    string(REPLACE ";" " " flag_string "${flag_list}")
    set(${var_name} "${flag_string}" PARENT_SCOPE)
endfunction()

clean_flag_var(CMAKE_CXX_FLAGS "-Wext-lambda-captures-this")


# ------------------------------
# ADIOS2
# ------------------------------
find_package(ADIOS2 REQUIRED CXX MPI)
message(STATUS "ADIOS2: ${ADIOS2_DIR}")

# An ADIOS2 built with Kokkos support calls Kokkos::initialize() itself, from
# adios2::helper::KokkosInit(). That defeats the host-only IO server: src/main.cpp
# deliberately does not initialize Kokkos on IO ranks, but ADIOS2 does it moments
# later and each IO rank opens a CUDA context (measured: 520 MiB) it never uses.
#
# Warn rather than fail: this only degrades IO ranks, the model itself is fine. It
# is worth saying loudly because an ADIOS2_DIR pointing at a path that does not
# exist makes CMake fall back to whatever else is on the search path *silently*,
# so the feature would go missing with no other symptom.
if(ADIOS2_HAVE_Kokkos)
    message(WARNING
        "ADIOS2 at ${ADIOS2_DIR} was built with Kokkos support.\n"
        "   IO server ranks will each occupy a GPU even though they never compute.\n"
        "   Rebuild ADIOS2 with -DADIOS2_USE_Kokkos=OFF and point ADIOS2_DIR at it.\n"
        "   (VVM never passes ADIOS2 a Kokkos view, so nothing else is affected.)")
else()
    message(STATUS "ADIOS2 has no Kokkos support -- IO server ranks stay off the GPU")
endif()

if(TARGET adios2::cxx11_mpi)
    target_link_libraries(adios2::cxx11_mpi INTERFACE MPI::MPI_CXX)
endif()

# ------------------------------
# HDF5
# ------------------------------
find_path(HDF5_INCLUDE_DIR hdf5.h HINTS ${HDF5_DIR}/include NO_DEFAULT_PATH)
find_library(HDF5_LIBRARY NAMES hdf5 HINTS ${HDF5_DIR}/lib NO_DEFAULT_PATH)
find_library(HDF5_HL_LIBRARY NAMES hdf5_hl HINTS ${HDF5_DIR}/lib NO_DEFAULT_PATH)

if(HDF5_INCLUDE_DIR AND HDF5_LIBRARY AND HDF5_HL_LIBRARY)
    add_library(HDF5::HDF5 UNKNOWN IMPORTED)
    set_target_properties(HDF5::HDF5 PROPERTIES
                          IMPORTED_LOCATION "${HDF5_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}")
    find_library(Z_LIBRARY z)
    find_library(DL_LIBRARY dl)
    find_library(M_LIBRARY m)
    target_link_libraries(HDF5::HDF5 INTERFACE ${Z_LIBRARY} ${DL_LIBRARY} ${M_LIBRARY} MPI::MPI_CXX)

    add_library(HDF5::HL UNKNOWN IMPORTED)
    set_target_properties(HDF5::HL PROPERTIES
                          IMPORTED_LOCATION "${HDF5_HL_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}")
    target_link_libraries(HDF5::HL INTERFACE HDF5::HDF5)
endif()

# ------------------------------
# pnetcdf
# ------------------------------
find_path(PNETCDF_INCLUDE_DIR pnetcdf.h
          HINTS ${PNETCDF_DIR}/include
          NO_DEFAULT_PATH)
find_library(PNETCDF_LIBRARY NAMES pnetcdf
             HINTS ${PNETCDF_DIR}/lib
             NO_DEFAULT_PATH)

if (PNETCDF_INCLUDE_DIR AND PNETCDF_LIBRARY)
    add_library(PnetCDF::pnetcdf UNKNOWN IMPORTED GLOBAL)
    set_target_properties(PnetCDF::pnetcdf PROPERTIES
                          IMPORTED_LOCATION "${PNETCDF_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${PNETCDF_INCLUDE_DIR}")
    find_library(M_LIBRARY m)
    target_link_libraries(PnetCDF::pnetcdf INTERFACE MPI::MPI_CXX ${M_LIBRARY})
    set(PNETCDF_FOUND TRUE)
    message(STATUS "PnetCDF found manually at: ${PNETCDF_LIBRARY}")
else()
    message(FATAL_ERROR "PnetCDF library or includes not found at the specified path: ${PNETCDF_DIR}")
endif()


# ------------------------------
# Project sources
# ------------------------------
include_directories(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src)


# Build tests if requested
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "=== Compiler flags summary ===")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
message(STATUS "Fortran Compiler: ${CMAKE_Fortran_COMPILER}")
message(STATUS "Fortran Flags: ${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "==============================")
