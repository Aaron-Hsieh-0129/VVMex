cmake_minimum_required(VERSION 3.20)

# ==========================================
# Toolchain discovery (must run before project())
#
# A preset names where things are installed -- NVHPC_DIR for the compiler stack,
# one prefix per dependency below -- and everything derivable is derived here.
# Any derived value can still be set explicitly, which is how a machine with a
# non-NVHPC MPI overrides it.
# ==========================================
set(NVHPC_DIR "" CACHE PATH "NVHPC install root (the <version> directory)")
set(VVM_MPI_ROOT "" CACHE PATH "MPI install providing mpic++/mpicc/mpifort; derived from NVHPC_DIR if empty")
set(VVM_GCC_TOOLCHAIN "" CACHE PATH "GCC toolchain for NVHPC to compile against (--gcc-toolchain=)")

if(NOT VVM_MPI_ROOT AND NVHPC_DIR)
    # NVHPC ships HPC-X under comm_libs/<cuda>/hpcx/hpcx-<ver>/ompi. Newest wins
    # when a release carries more than one; pin VVM_MPI_ROOT to override.
    file(GLOB _vvm_hpcx "${NVHPC_DIR}/comm_libs/*/hpcx/hpcx-*/ompi")
    list(SORT _vvm_hpcx)
    list(REVERSE _vvm_hpcx)
    foreach(_vvm_cand IN LISTS _vvm_hpcx)
        if(EXISTS "${_vvm_cand}/bin/mpic++")
            set(VVM_MPI_ROOT "${_vvm_cand}" CACHE PATH "MPI install" FORCE)
            break()
        endif()
    endforeach()
    unset(_vvm_hpcx)
    unset(_vvm_cand)
endif()

if(VVM_MPI_ROOT)
    if(NOT CMAKE_CXX_COMPILER AND EXISTS "${VVM_MPI_ROOT}/bin/mpic++")
        set(CMAKE_CXX_COMPILER "${VVM_MPI_ROOT}/bin/mpic++" CACHE FILEPATH "C++ compiler")
    endif()
    if(NOT CMAKE_C_COMPILER AND EXISTS "${VVM_MPI_ROOT}/bin/mpicc")
        set(CMAKE_C_COMPILER "${VVM_MPI_ROOT}/bin/mpicc" CACHE FILEPATH "C compiler")
    endif()
    if(NOT CMAKE_Fortran_COMPILER AND EXISTS "${VVM_MPI_ROOT}/bin/mpifort")
        set(CMAKE_Fortran_COMPILER "${VVM_MPI_ROOT}/bin/mpifort" CACHE FILEPATH "Fortran compiler")
    endif()
endif()

# NVHPC compiles against the headers of whichever GCC it is pointed at; the
# runtime libstdc++ has to match or standalone binaries fault at startup.
if(VVM_GCC_TOOLCHAIN)
    foreach(_vvm_lang CXX C Fortran)
        if(NOT CMAKE_${_vvm_lang}_FLAGS MATCHES "gcc-toolchain")
            string(STRIP "--gcc-toolchain=${VVM_GCC_TOOLCHAIN} ${CMAKE_${_vvm_lang}_FLAGS}" _vvm_flags)
            set(CMAKE_${_vvm_lang}_FLAGS "${_vvm_flags}" CACHE STRING "${_vvm_lang} flags" FORCE)
        endif()
    endforeach()
    unset(_vvm_lang)
    unset(_vvm_flags)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(VVMex LANGUAGES CXX C Fortran)

message(STATUS "MPI wrappers: ${VVM_MPI_ROOT}")

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

# ==========================================
# Dependency prefixes
#
# Every dependency is named by one variable holding one install prefix, so a
# preset reads the same way for all of them:
#
#   HDF5_DIR / NETCDF_C_DIR / NETCDF_Fortran_DIR / PNETCDF_DIR / Kokkos_DIR / ADIOS2_DIR
#
# Kokkos and ADIOS2 are found with find_package(), whose own convention is that
# <Pkg>_DIR names the directory holding the package config file rather than the
# prefix above it. Both spellings are accepted here: a value that is not already
# a config directory is treated as a prefix and handed to find_package() as a
# search root instead, which is why all six can be written the same way.
# ==========================================
# Where a package config lives under a prefix is not portable: lib/cmake/<pkg> on
# some installs, lib64/cmake/<pkg> on others. find_package() only searches lib64
# when FIND_LIBRARY_USE_LIB64_PATHS is on, and it is off on Debian-derived
# systems, so the lookup is done here instead of relying on that.
set(VVM_LIBDIR_CANDIDATES lib lib64)
if(CMAKE_LIBRARY_ARCHITECTURE)
    list(APPEND VVM_LIBDIR_CANDIDATES lib/${CMAKE_LIBRARY_ARCHITECTURE})
endif()
list(APPEND VVM_LIBDIR_CANDIDATES share)

foreach(_vvm_pkg Kokkos ADIOS2)
    string(TOLOWER "${_vvm_pkg}" _vvm_pkg_lc)
    if(${_vvm_pkg}_DIR
       AND NOT EXISTS "${${_vvm_pkg}_DIR}/${_vvm_pkg}Config.cmake"
       AND NOT EXISTS "${${_vvm_pkg}_DIR}/${_vvm_pkg_lc}-config.cmake")
        set(_vvm_found "")
        foreach(_vvm_libdir IN LISTS VVM_LIBDIR_CANDIDATES)
            foreach(_vvm_name ${_vvm_pkg} ${_vvm_pkg_lc})
                set(_vvm_cand "${${_vvm_pkg}_DIR}/${_vvm_libdir}/cmake/${_vvm_name}")
                if(EXISTS "${_vvm_cand}/${_vvm_pkg}Config.cmake"
                   OR EXISTS "${_vvm_cand}/${_vvm_pkg_lc}-config.cmake")
                    set(_vvm_found "${_vvm_cand}")
                    break()
                endif()
            endforeach()
            if(_vvm_found)
                break()
            endif()
        endforeach()
        if(_vvm_found)
            message(STATUS "${_vvm_pkg}: ${${_vvm_pkg}_DIR} -> ${_vvm_found}")
            set(${_vvm_pkg}_DIR "${_vvm_found}" CACHE PATH "Config directory for ${_vvm_pkg}" FORCE)
        else()
            # Not a prefix we recognise; hand it to find_package as a search root
            # and let it report the failure with its own diagnostics.
            set(${_vvm_pkg}_ROOT "${${_vvm_pkg}_DIR}" CACHE PATH "Install prefix for ${_vvm_pkg}" FORCE)
            unset(${_vvm_pkg}_DIR CACHE)
        endif()
    endif()
endforeach()
unset(_vvm_pkg)
unset(_vvm_pkg_lc)
unset(_vvm_found)
unset(_vvm_cand)
unset(_vvm_libdir)
unset(_vvm_name)

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

# Strip a flag from a FORCEd cache variable, so switching an option back off
# actually removes what switching it on added.
macro(vvm_unset_flag VAR FLAG DOC)
    string(REPLACE "${FLAG}" "" _vvm_flags "${${VAR}}")
    string(REGEX REPLACE "  +" " " _vvm_flags "${_vvm_flags}")
    string(STRIP "${_vvm_flags}" _vvm_flags)
    set(${VAR} "${_vvm_flags}" CACHE STRING "${DOC}" FORCE)
endmacro()

if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(STATUS "Applying NVHPC dependency flag workaround.")
    # Disable automatic dependency generation flags that NVHPC might set
    set(CMAKE_CXX_DEPENDS "")
    set(CMAKE_C_DEPENDS "")
    set(CMAKE_DEPFILE_FLAGS_CXX "")
    set(CMAKE_DEPFILE_FLAGS_C "")

    # No CMAKE_CUDA_FLAGS here. This project declares LANGUAGES CXX C Fortran and
    # has no .cu sources -- device code reaches nvcc through nvcc_wrapper as CXX
    # -- so CUDA-language flags are never consumed. Setting them wrote a cache
    # entry that appeared on no compile line, which read like fast-math was in
    # effect when it was not. If CUDA is ever enabled as a language, add the
    # flags then, and note that nvcc's -use_fast_math only affects single
    # precision. Heal caches written before this change, so a reconfigured tree
    # does not keep advertising flags nothing reads.
    vvm_unset_flag(CMAKE_CUDA_FLAGS "-use_fast_math" "CUDA flags")
    vvm_unset_flag(CMAKE_CUDA_FLAGS "-Xcompiler -mp" "CUDA flags")

    # -fast belongs to the optimized configurations, not to every build. CMake's
    # NVHPC defaults already put it in *_RELEASE (as they do for Fortran, which
    # this block never touched), so adding it globally only ever reached the
    # configurations that should not have it: a Debug build came out as
    # `-fast -g -O0`, which still vectorizes and reassociates -- its FP
    # reduction differs from a plain -O0 build. Strip any copy an earlier
    # configure left in the cache, and state the intent per configuration.
    vvm_unset_flag(CMAKE_CXX_FLAGS "-fast" "CXX flags")
    vvm_unset_flag(CMAKE_C_FLAGS   "-fast" "C flags")
    foreach(_vvm_opt_config RELEASE RELWITHDEBINFO MINSIZEREL)
        vvm_set_flag_once(CMAKE_CXX_FLAGS_${_vvm_opt_config} "-fast"
                          "CXX ${_vvm_opt_config} flags")
        vvm_set_flag_once(CMAKE_C_FLAGS_${_vvm_opt_config}   "-fast"
                          "C ${_vvm_opt_config} flags")
    endforeach()
endif()

# Bit-identical results between a GPU build and a CPU-only build.
#
# Two things have to change together. nvcc and nvc++ both contract a*b+c into an
# FMA, but in a multi-product expression they need not fuse the same multiply,
# so a Takacs flux term differs in the last ULP between backends -- that is the
# first divergence between a GPU and a CPU-only run of advection_u. And
# Kokkos::parallel_reduce combines partial sums in a backend-specific order, so
# State's horizontal mean switches to a fixed row-then-rank order under the same
# switch. Measured cost: ~0.2% on GPU (2dbubble) and ~1.2% on CPU
# (advection_u).
#
# OFF by default because both changes move results in the last ULP, and the
# v1.0.0 baselines encode the default. Runs with P3, RRTMGP or Noah still differ
# across backends even with this on, because log/pow/tanh have separate device
# and host implementations. See docs/developer-guides/reproducibility.md.
option(VVM_DETERMINISTIC_FP "Disable FP contraction so CPU and GPU agree bit-for-bit" OFF)
if(VVM_DETERMINISTIC_FP)
    message(STATUS "VVM deterministic FP: ON (no FMA contraction, fixed-order reductions)")
    add_compile_definitions(VVM_DETERMINISTIC_FP)
    if(VVM_ENABLE_GPU)
        # .cpp files reach nvcc through nvcc_wrapper, so the device switch has
        # to travel in the CXX flags as well as the CUDA ones.
        vvm_set_flag_once(CMAKE_CXX_FLAGS  "--fmad=false" "CXX flags")
        vvm_set_flag_once(CMAKE_CUDA_FLAGS "--fmad=false" "CUDA flags")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        vvm_set_flag_once(CMAKE_CXX_FLAGS "-Mnofma" "CXX flags")
        vvm_set_flag_once(CMAKE_C_FLAGS   "-Mnofma" "C flags")
    endif()
else()
    vvm_unset_flag(CMAKE_CXX_FLAGS  "--fmad=false" "CXX flags")
    vvm_unset_flag(CMAKE_CUDA_FLAGS "--fmad=false" "CUDA flags")
    vvm_unset_flag(CMAKE_CXX_FLAGS  "-Mnofma"      "CXX flags")
    vvm_unset_flag(CMAKE_C_FLAGS    "-Mnofma"      "C flags")
endif()

option(VVM_USE_DOUBLE_PRECISION "Use double precision (double) instead of single precision (float)" ON)
if(VVM_USE_DOUBLE_PRECISION)
    message(STATUS "VVM Precision: DOUBLE (FP64)")
    add_compile_definitions(VVM_USE_DOUBLE_PRECISION)
else()
    message(STATUS "VVM Precision: SINGLE (FP32)")
endif()

option(SCREAM_DOUBLE_PRECISION "Use double precision in P3 and the shared EAMxx types" ${VVM_USE_DOUBLE_PRECISION})
if(SCREAM_DOUBLE_PRECISION)
    message(STATUS "P3 Precision: DOUBLE (FP64)")
    add_compile_definitions(SCREAM_DOUBLE_PRECISION)
else()
    message(STATUS "P3 Precision: SINGLE (FP32)")
endif()

option(RRTMGP_USE_DOUBLE_PRECISION "Use double precision inside the VVM RRTMGP radiation interface" ${VVM_USE_DOUBLE_PRECISION})
if(RRTMGP_USE_DOUBLE_PRECISION)
    message(STATUS "RRTMGP Internal Precision: DOUBLE (FP64)")
    add_compile_definitions(VVM_RRTMGP_USE_DOUBLE_PRECISION)
else()
    message(STATUS "RRTMGP Internal Precision: SINGLE (FP32)")
endif()
if(NOT (VVM_USE_DOUBLE_PRECISION STREQUAL SCREAM_DOUBLE_PRECISION AND
        VVM_USE_DOUBLE_PRECISION STREQUAL RRTMGP_USE_DOUBLE_PRECISION))
    message(WARNING
        "Mixed precision: VVM=${VVM_USE_DOUBLE_PRECISION} P3=${SCREAM_DOUBLE_PRECISION} "
        "RRTMGP=${RRTMGP_USE_DOUBLE_PRECISION}. This builds and runs, but set all three "
        "unless the mix is deliberate.")
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

# vvm_land is linked by nvfortran with -acc, so NVHPC drives that link through
# pgacclnk, which ignores -lcudart_static and prints
#   pgacclnk: -lcudart_static switch ignored, please use -static-nvidia instead
# on every relink. Which CUDA runtime reaches that line comes from the Kokkos
# install: most expose CUDA::cudart, some expose CUDA::cudart_static. Swap it for
# the shared runtime so the ignored switch is never passed in the first place.
#
# Done at the library level rather than with nvcc's -cudart switch, because that
# switch is fatal on this link: nvfortran-Error-Unknown switch: -cudart.
if(VVM_ENABLE_GPU AND TARGET CUDA::cudart)
    # Kokkos::CUDA is where the runtime actually sits; the rest are listed so a
    # differently packaged install is caught too.
    foreach(_vvm_kk Kokkos::CUDA Kokkos::kokkos Kokkos::kokkoscore
                    Kokkos::kokkoscontainers Kokkos::kokkosalgorithms Kokkos::kokkossimd)
        if(NOT TARGET ${_vvm_kk})
            continue()
        endif()
        get_target_property(_vvm_alias ${_vvm_kk} ALIASED_TARGET)
        if(_vvm_alias)
            continue()
        endif()
        get_target_property(_vvm_kk_libs ${_vvm_kk} INTERFACE_LINK_LIBRARIES)
        if(_vvm_kk_libs AND _vvm_kk_libs MATCHES "cudart_static")
            string(REPLACE "CUDA::cudart_static" "CUDA::cudart" _vvm_kk_libs "${_vvm_kk_libs}")
            set_target_properties(${_vvm_kk} PROPERTIES INTERFACE_LINK_LIBRARIES "${_vvm_kk_libs}")
            message(STATUS "${_vvm_kk}: using the shared CUDA runtime (pgacclnk ignores -lcudart_static)")
        endif()
    endforeach()
    unset(_vvm_kk)
    unset(_vvm_alias)
    unset(_vvm_kk_libs)
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
             HINTS ${NETCDF_C_DIR}/lib ${NETCDF_C_DIR}/lib64
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
find_library(NETCDF_FORTRAN_LIBRARY NAMES netcdff HINTS "${NETCDF_Fortran_DIR}/lib" "${NETCDF_Fortran_DIR}/lib64" NO_DEFAULT_PATH)

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
find_library(HDF5_LIBRARY NAMES hdf5 HINTS ${HDF5_DIR}/lib ${HDF5_DIR}/lib64 NO_DEFAULT_PATH)
find_library(HDF5_HL_LIBRARY NAMES hdf5_hl HINTS ${HDF5_DIR}/lib ${HDF5_DIR}/lib64 NO_DEFAULT_PATH)

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
             HINTS ${PNETCDF_DIR}/lib ${PNETCDF_DIR}/lib64
             NO_DEFAULT_PATH)

if (PNETCDF_INCLUDE_DIR AND PNETCDF_LIBRARY)
    add_library(PnetCDF::pnetcdf UNKNOWN IMPORTED GLOBAL)
    set_target_properties(PnetCDF::pnetcdf PROPERTIES
                          IMPORTED_LOCATION "${PNETCDF_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${PNETCDF_INCLUDE_DIR}")
    find_library(M_LIBRARY m)
    target_link_libraries(PnetCDF::pnetcdf INTERFACE MPI::MPI_CXX NetCDF::netcdf ${M_LIBRARY})
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
