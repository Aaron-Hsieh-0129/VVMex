# ---------------------------------------------------------------------------
# CPU builds: keep the tests off the GPU.
#
# ctest runs the binary directly, so it inherits the caller's LD_LIBRARY_PATH.
# A CPU Kokkos install has the same SONAME (libkokkoscore.so.4.7) as the CUDA
# one, and the shell environment here carries the CUDA prefix, so the CPU binary
# loads the CUDA Kokkos and opens a GPU context: the tests occupy a GPU and stop
# testing the CPU backend at all. RUNPATH cannot fix this -- it is searched after
# LD_LIBRARY_PATH, and nvc++ discards -Wl,--disable-new-dtags -- so prepend the
# CPU library directory for every test instead.
# ---------------------------------------------------------------------------
if(NOT VVM_ENABLE_GPU)
    set(VVM_CPU_RUNTIME_DIRS "")
    if(Kokkos_DIR)
        string(REGEX REPLACE "/cmake/Kokkos/?$" "" _vvm_kokkos_libdir "${Kokkos_DIR}")
        list(APPEND VVM_CPU_RUNTIME_DIRS "${_vvm_kokkos_libdir}")
    endif()
    if(ADIOS2_DIR)
        string(REGEX REPLACE "/cmake/adios2/?$" "" _vvm_adios2_libdir "${ADIOS2_DIR}")
        list(APPEND VVM_CPU_RUNTIME_DIRS "${_vvm_adios2_libdir}")
        # ADIOS2 itself is installed in lib64 in the F1 prefix, while its
        # libfabric dependency is installed in the sibling lib directory.
        # Keep both ahead of the Intel module's older libfabric at test time.
        get_filename_component(_vvm_adios2_prefix "${_vvm_adios2_libdir}" DIRECTORY)
        if(EXISTS "${_vvm_adios2_prefix}/lib")
            list(APPEND VVM_CPU_RUNTIME_DIRS "${_vvm_adios2_prefix}/lib")
        endif()
    endif()
    if(HDF5_DIR)
        foreach(_vvm_dep_libdir "${HDF5_DIR}/lib64" "${HDF5_DIR}/lib")
            if(EXISTS "${_vvm_dep_libdir}")
                list(APPEND VVM_CPU_RUNTIME_DIRS "${_vvm_dep_libdir}")
            endif()
        endforeach()
    endif()
    list(REMOVE_DUPLICATES VVM_CPU_RUNTIME_DIRS)

    # OpenMP is the execution space for a CPU build, and ctest launches the binary
    # directly rather than through submit.py/core_run.sh, so nothing else sets the
    # thread count. Without this each rank inherits whatever OMP_NUM_THREADS the
    # caller happens to have -- usually unset, which means "one thread per core"
    # and makes multi-rank tests oversubscribe badly.
    #
    # VVM_TEST_CPU_THREADS is defined near the top of this file, because the
    # PROCESSORS weight each test declares is derived from it. 64 rather than 16:
    # measured on blaze, the two slowest single-rank cases go 2dbubble
    # 283 s -> 108 s and mountain 405 s -> 134 s, and the results do not move --
    # 64-thread output is bit-for-bit identical to 16-thread over all 272710
    # values of 2dbubble, and mountain still matches its SHA-256 digest. Thread
    # count is not one of the axes that changes CPU arithmetic (unlike rank count
    # for the horizontal mean), so raising it costs no reference data.
    # OMP_PROC_BIND from the caller's environment undoes --bind-to none one level
    # down: with every rank holding the same full CPU mask, "spread" puts every
    # rank's master thread on place 0, so all the ranks of one test busy-spin
    # their MPI progress loops on CPU 0 while the workers idle elsewhere. It
    # only bites the multi-rank tiers -- a single-rank test has CPU 0 to itself
    # and is unaffected. Measured on blaze with OMP_PROC_BIND=spread exported,
    # multirank_twisting_r4 took 1323 s against 4 s for the same case at one
    # rank; pinning OMP_PROC_BIND=false here brings it to 6 s. core_run.sh
    # passes -x OMP_PROC_BIND=false for the same reason, and ctest is the one
    # launcher that does not go through it. Binding does not affect results, so
    # this costs no reference data.
    set(VVM_TEST_ENV_MODS "OMP_NUM_THREADS=set:${VVM_TEST_CPU_THREADS}"
                          "OMP_PROC_BIND=set:false")

    if(VVM_CPU_RUNTIME_DIRS)
        string(REPLACE ";" ":" VVM_CPU_LD_PREPEND "${VVM_CPU_RUNTIME_DIRS}")
        message(STATUS "CPU tests: prepending to LD_LIBRARY_PATH: ${VVM_CPU_LD_PREPEND}")
        list(APPEND VVM_TEST_ENV_MODS
             "LD_LIBRARY_PATH=path_list_prepend:${VVM_CPU_LD_PREPEND}")
    endif()

    cmake_host_system_information(RESULT VVM_HOST_CORES QUERY NUMBER_OF_LOGICAL_CORES)
    message(STATUS "CPU tests: OMP_NUM_THREADS=${VVM_TEST_CPU_THREADS} per rank "
                   "(host reports ${VVM_HOST_CORES} logical cores)")
    # The check is per rank *count*, not per rank: the multirank tier runs 4 of
    # them at once, so a thread count that fits a single-rank case can still
    # oversubscribe by 4x once that tier is on.
    if(VVM_TEST_MULTIRANK)
        set(VVM_TEST_PEAK_RANKS 4)
    else()
        set(VVM_TEST_PEAK_RANKS 1)
    endif()
    math(EXPR VVM_TEST_PEAK_THREADS "${VVM_TEST_CPU_THREADS} * ${VVM_TEST_PEAK_RANKS}")
    if(VVM_TEST_PEAK_THREADS GREATER VVM_HOST_CORES)
        message(WARNING
            "CPU tests are set to ${VVM_TEST_CPU_THREADS} threads/rank and the widest "
            "tier enabled runs ${VVM_TEST_PEAK_RANKS} rank(s), so they peak at "
            "${VVM_TEST_PEAK_THREADS} threads on a host with ${VVM_HOST_CORES} logical "
            "cores. They will oversubscribe and run slowly. Results stay correct -- "
            "thread count does not change them. Override with "
            "-DVVM_TEST_CPU_THREADS=<n>.")
    endif()

    get_property(VVM_ALL_TESTS DIRECTORY PROPERTY TESTS)
    if(VVM_ALL_TESTS)
        set_tests_properties(${VVM_ALL_TESTS} PROPERTIES
            ENVIRONMENT_MODIFICATION "${VVM_TEST_ENV_MODS}")
    endif()
endif()
