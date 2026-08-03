find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rundata/input_configs/default_cases)
# one_gpu_per_rank.sh only assigns CUDA_VISIBLE_DEVICES, so it has nothing to do
# in a CPU build.
if(VVM_ENABLE_GPU)
    set(GPU_WRAP ${TEST_DIR}/scripts/one_gpu_per_rank.sh)
else()
    set(GPU_WRAP "")
endif()

# Reference data is backend-specific. CPU and GPU agree only to a few ulp, which
# the 1e-6 tolerance absorbs for the advection cases but not for 2dbubble (a
# convective case amplifies it to ~2e-2 over 120 steps), and never for the
# SHA-256 digests. So each backend is gated against references generated on that
# backend; cross-backend agreement is a separate, looser question.
if(VVM_ENABLE_GPU)
    set(VVM_BASELINE_DIR  ${TEST_DIR}/baselines)
    set(VVM_REFERENCE_DIR ${TEST_DIR}/references)
else()
    set(VVM_BASELINE_DIR  ${TEST_DIR}/baselines_cpu)
    set(VVM_REFERENCE_DIR ${TEST_DIR}/references_cpu)
endif()
message(STATUS "Test reference data: ${VVM_BASELINE_DIR}, ${VVM_REFERENCE_DIR}")

# OpenMPI binds each rank to a single core by default. On a CPU build that pins
# every OpenMP thread of a rank onto one CPU, so raising OMP_NUM_THREADS makes
# the run slower rather than faster (measured: advection_u 20 s -> 45 s at 16
# threads, Cpus_allowed_list: 0). core_run.sh passes --bind-to none for the same
# reason. The GPU path is one thread per rank, so leave its binding alone.
if(VVM_ENABLE_GPU)
    set(VVM_MPI_BIND_ARGS "")
else()
    set(VVM_MPI_BIND_ARGS --bind-to none)
endif()

# Use VVM_ROOT as the runtime working directory for CTest tests.
# Root CMakeLists.txt already requires VVM_ROOT, so this should normally exist.
set(VVM_TEST_WORKDIR "$ENV{VVM_ROOT}")
if(VVM_TEST_WORKDIR STREQUAL "")
  message(FATAL_ERROR "VVM_ROOT is not set. Please export VVM_ROOT=/path/to/VVMex before configuring tests.")
endif()

file(TO_CMAKE_PATH "${VVM_TEST_WORKDIR}" VVM_TEST_WORKDIR)

# How long each default_cases test integrates. Short enough to stay a test,
# long enough that a numerical regression has shown up in the output.
set(VVM_TEST_SECONDS 120 CACHE STRING "Simulated seconds for default_cases tests")

# ---------------------------------------------------------------------------
# Test tiers, by the resource they need -- not by how much we trust them.
#
#   default                1 GPU,  ~40 s   always registered; plain `ctest`
#   VVM_TEST_PHYSICS       1 GPU,  ~80 s   P3 / RRTMGP / Noah land
#   VVM_TEST_MULTIRANK     4 GPUs, ~30 s   rank invariance (2-D decomposition)
#   VVM_TEST_LARGE         8 GPUs, ~140 s  2048^2, needs the 961 MB input file
#
# The optional tiers are opt-in at CONFIGURE time rather than label-filtered,
# because `ctest` with no arguments runs every registered test -- `-L` filters
# down to a label, it does not hold anything back. Registering them only when
# asked is what actually keeps them off a default run on a 1-GPU machine.
#
#   cmake -B build -DVVM_TEST_PHYSICS=ON -DVVM_TEST_MULTIRANK=ON
#
# They also carry labels, so once enabled you can still select a tier with
# `ctest -L multirank`.
#
# Every tier is a bit-for-bit gate: with random_perturbation disabled (see
# make_test_config.py) each case is exactly reproducible, so there is no
# tolerance to tune. A failure means bits moved.
#
# Multi-rank tiers need that many *physical* GPUs. Several NCCL ranks sharing one
# device is unsupported and silently changes results, so they cannot be faked on
# a smaller machine -- see tests/scripts/one_gpu_per_rank.sh.
# ---------------------------------------------------------------------------
option(VVM_TEST_PHYSICS   "Register physics-tier tests (1 GPU, ~80 s)"                OFF)
option(VVM_TEST_MULTIRANK "Register rank-invariance tests (needs 4 GPUs, ~30 s)"      OFF)
option(VVM_TEST_LARGE     "Register the 2048^2 test (needs 8 GPUs, 961 MB input)"     OFF)

# Report the tiers the way the rest of the configure output reports its options,
# including what each one needs and how to switch it on.
function(_report_test_tier LABEL STATE NEEDS TIME_S OPT_NAME)
    if(STATE)
        message(STATUS "  Test tier ${LABEL}: ON   (${NEEDS}, ~${TIME_S})")
    else()
        message(STATUS "  Test tier ${LABEL}: OFF  (${NEEDS}, ~${TIME_S}) -- enable with -D${OPT_NAME}=ON")
    endif()
endfunction()

message(STATUS "Test tiers (all are bit-for-bit gates; simulated seconds per case: ${VVM_TEST_SECONDS}):")
message(STATUS "  Test tier default  : ON   (1 GPU, ~36 s) -- always registered")
_report_test_tier("physics  " ${VVM_TEST_PHYSICS}   "1 GPU"  "72 s"  VVM_TEST_PHYSICS)
_report_test_tier("multirank" ${VVM_TEST_MULTIRANK} "4 GPUs" "40 s"  VVM_TEST_MULTIRANK)
_report_test_tier("large    " ${VVM_TEST_LARGE}     "8 GPUs" "114 s" VVM_TEST_LARGE)

# The original hand-written regression cases, comparing against stored baseline
# files. Small enough that a full baseline is practical.
function(add_vvm_test TEST_NAME VARS_TO_TEST)
    # The checked-in configs write to ./build/testing_output_<case>/, so the run
    # and the verify step only agree when the binary directory happens to be
    # named "build" -- in any other tree the run writes to ./build/... while the
    # verify looks in ${CMAKE_BINARY_DIR}/... and reports "model produced no
    # output". Stamp a copy of the config with this tree's output path instead,
    # the same way add_case_test() does through make_test_config.py.
    set(ODIR ${CMAKE_BINARY_DIR}/testing_output_${TEST_NAME})
    set(CFG ${CMAKE_BINARY_DIR}/test_configs/${TEST_NAME}.json)
    file(READ ${TEST_DIR}/configs/${TEST_NAME}.json VVM_TEST_CFG)
    string(REGEX REPLACE "\"output_dir\"[ \t]*:[ \t]*\"[^\"]*\""
                         "\"output_dir\": \"${ODIR}/\""
                         VVM_TEST_CFG "${VVM_TEST_CFG}")
    file(WRITE ${CFG} "${VVM_TEST_CFG}")

    add_test(NAME Run_${TEST_NAME}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                     $<TARGET_FILE:vvm> ${CFG}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    separate_arguments(VARS_LIST UNIX_COMMAND "${VARS_TO_TEST}")

    add_test(NAME Verify_${TEST_NAME}
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/verify_output.py
             --baseline ${VVM_BASELINE_DIR}/${TEST_NAME}.h5
             --test_out ${ODIR}/vvm_output_000001.h5
             --vars ${VARS_LIST}
             --tol 1e-6)

    set_tests_properties(Run_${TEST_NAME}    PROPERTIES RESOURCE_LOCK gpu)
    set_tests_properties(Verify_${TEST_NAME} PROPERTIES DEPENDS Run_${TEST_NAME})
endfunction()

# A default_cases case, checked bit-for-bit against a stored per-variable digest.
# Full baselines do not scale here (rcemip output is ~1 GB), so check_output.py
# compares SHA-256 digests instead.
#   add_case_test(<case> <ranks> <label>)   label "" => default tier
function(add_case_test CASE_NAME RANKS LABEL)
    set(EXTRA_CFG_ARGS ${ARGN})
    if(LABEL STREQUAL "")
        set(TAG ${CASE_NAME})
    else()
        set(TAG ${LABEL}_${CASE_NAME})
    endif()
    set(CFG ${CMAKE_BINARY_DIR}/test_configs/${TAG}.json)
    set(ODIR ${CMAKE_BINARY_DIR}/testing_output_${TAG})

    add_test(NAME Prep_${TAG}
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/make_test_config.py
                     --case ${CASE_NAME} --source-dir ${CASE_DIR}
                     --out-config ${CFG} --out-dir ${ODIR}
                     --seconds ${VVM_TEST_SECONDS} ${EXTRA_CFG_ARGS}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    add_test(NAME Run_${TAG}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS} ${VVM_MPI_BIND_ARGS}
                     ${GPU_WRAP} $<TARGET_FILE:vvm> ${CFG}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    add_test(NAME Verify_${TAG}
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/check_output.py
                     --output ${ODIR}/vvm_output_000001.h5
                     --reference ${VVM_REFERENCE_DIR}/${CASE_NAME}.json)

    set_tests_properties(Run_${TAG}    PROPERTIES DEPENDS Prep_${TAG} RESOURCE_LOCK gpu)
    set_tests_properties(Verify_${TAG} PROPERTIES DEPENDS Run_${TAG})
    if(NOT LABEL STREQUAL "")
        set_tests_properties(Prep_${TAG} Run_${TAG} Verify_${TAG} PROPERTIES LABELS ${LABEL})
    endif()
endfunction()

# Same case at two rank counts, compared against each other. No stored reference:
# the assertion is that the answer does not depend on the decomposition. 4 ranks
# gives a 2-D (2x2) split, which is what exposes index bugs -- 2 ranks splits one
# direction only and misses them.
function(add_rank_invariance_test CASE_NAME RANKS)
    set(TAG multirank_${CASE_NAME})
    set(CFG1 ${CMAKE_BINARY_DIR}/test_configs/${TAG}_r1.json)
    set(CFGN ${CMAKE_BINARY_DIR}/test_configs/${TAG}_r${RANKS}.json)
    set(ODIR1 ${CMAKE_BINARY_DIR}/testing_output_${TAG}_r1)
    set(ODIRN ${CMAKE_BINARY_DIR}/testing_output_${TAG}_r${RANKS})

    add_test(NAME Prep_${TAG}
             COMMAND ${CMAKE_COMMAND}
                     -DPY=${Python3_EXECUTABLE} -DSCRIPT=${TEST_DIR}/scripts/make_test_config.py
                     -DCASE=${CASE_NAME} -DSRC=${CASE_DIR}
                     -DCFG1=${CFG1} -DCFGN=${CFGN} -DODIR1=${ODIR1} -DODIRN=${ODIRN}
                     -DSECONDS=${VVM_TEST_SECONDS}
                     -P ${TEST_DIR}/scripts/prep_pair.cmake
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    add_test(NAME Run_${TAG}_r1
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                     ${GPU_WRAP} $<TARGET_FILE:vvm> ${CFG1}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    add_test(NAME Run_${TAG}_r${RANKS}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS} ${VVM_MPI_BIND_ARGS}
                     ${GPU_WRAP} $<TARGET_FILE:vvm> ${CFGN}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    add_test(NAME Verify_${TAG}
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/compare_outputs.py
                     --a ${ODIR1}/vvm_output_000001.h5
                     --b ${ODIRN}/vvm_output_000001.h5
                     --label-a "1 rank" --label-b "${RANKS} ranks")

    set_tests_properties(Run_${TAG}_r1       PROPERTIES DEPENDS Prep_${TAG} RESOURCE_LOCK gpu)
    set_tests_properties(Run_${TAG}_r${RANKS} PROPERTIES DEPENDS Run_${TAG}_r1 RESOURCE_LOCK gpu)
    set_tests_properties(Verify_${TAG}       PROPERTIES DEPENDS Run_${TAG}_r${RANKS})
    set_tests_properties(Prep_${TAG} Run_${TAG}_r1 Run_${TAG}_r${RANKS} Verify_${TAG}
                         PROPERTIES LABELS multirank)
endfunction()

# --- unit: no GPU, no MPI, <1 s -------------------------------------------
# Plain executables over header-only logic. No device, no ranks, no resource
# lock, so they run alongside everything else and on any machine.
function(add_vvm_unit_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_DIR}/unit/${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS unit)
endfunction()

# When radiation and the surface process recompute their coefficients.
add_vvm_unit_test(test_process_scheduling)

# What the stale-SST cleanup in main() is allowed to delete, and that it treats
# every character in output_dir / output_filename_prefix as a filename character
# rather than shell syntax.
add_vvm_unit_test(test_sst_path)

# How a restart run recovers its clock: the metadata stored inside the file
# decides, a stored time and step must agree, and a file with neither is only
# usable through an explicit opt-in. Pure decision logic.
add_vvm_unit_test(test_restart_time)

# --- unit, on files: no GPU, 1-2 ranks, <1 s ------------------------------
# Same idea as add_vvm_unit_test, but the code under test reads real HDF5 and
# NetCDF files, and PnetCDF needs MPI. No device is involved, so the multi-rank
# run needs no extra GPU and is registered unconditionally -- it is what checks
# that every rank recovers the same clock.
#   add_vvm_file_unit_test(<name> <ranks>)
function(add_vvm_file_unit_test TEST_NAME RANKS)
    add_executable(${TEST_NAME} ${TEST_DIR}/unit/${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${TEST_NAME} PRIVATE
                          HDF5::HL PnetCDF::pnetcdf NetCDF::netcdf MPI::MPI_CXX)

    foreach(N 1 ${RANKS})
        add_test(NAME ${TEST_NAME}_r${N}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${N} ${VVM_MPI_BIND_ARGS}
                         $<TARGET_FILE:${TEST_NAME}>
                 WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
        set_tests_properties(${TEST_NAME}_r${N} PROPERTIES LABELS unit)
    endforeach()
endfunction()

# Restart metadata in the files themselves, including the regression this all
# exists for: write a restart file, rename it, load it, and the recovered time
# and step do not move.
add_vvm_file_unit_test(test_restart_metadata_io 2)

# --- unit, on the device: 1 GPU per rank, <10 s ----------------------------
# Same shape as add_vvm_unit_test, but the code under test is a State method, so
# the binary needs Kokkos, MPI and a GPU. Registered at one rank always and at
# RANKS ranks only when the multirank tier is on, since extra ranks need extra
# physical GPUs (see one_gpu_per_rank.sh).
#   add_vvm_device_unit_test(<name> <config> <ranks>)
function(add_vvm_device_unit_test TEST_NAME CONFIG RANKS)
    add_executable(${TEST_NAME} ${TEST_DIR}/unit/${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${TEST_NAME} PRIVATE vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

    add_test(NAME ${TEST_NAME}_r1
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                     ${GPU_WRAP} $<TARGET_FILE:${TEST_NAME}> ${CONFIG}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
    set_tests_properties(${TEST_NAME}_r1 PROPERTIES LABELS unit RESOURCE_LOCK gpu)

    if(VVM_TEST_MULTIRANK)
        add_test(NAME ${TEST_NAME}_r${RANKS}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS} ${VVM_MPI_BIND_ARGS}
                         ${GPU_WRAP} $<TARGET_FILE:${TEST_NAME}> ${CONFIG}
                 WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
        set_tests_properties(${TEST_NAME}_r${RANKS}
                             PROPERTIES LABELS "unit;multirank" RESOURCE_LOCK gpu)
    endif()
endfunction()

# State::calculate_horizontal_mean(): the API shared by the NCCL and the
# standard-MPI backend. The 2-rank run is what pins the global reduction.
add_vvm_device_unit_test(test_horizontal_mean ${TEST_DIR}/configs/2dbubble.json 2)

# HaloExchanger: the NCCL and the standard-MPI implementation are separate code,
# and only a value-level check keeps them exchanging the same cells.
add_vvm_device_unit_test(test_halo_exchange ${TEST_DIR}/configs/2dbubble.json 2)

# --- default: 1 GPU, ~40 s -------------------------------------------------
add_vvm_test(advection_u "u th xi eta zeta")
add_vvm_test(advection_v "v th xi eta zeta")
add_vvm_test(advection_w "w th xi eta")
add_vvm_test(stretching "u v w xi eta zeta")
add_vvm_test(twisting "u v w xi eta zeta")
add_vvm_test(2dbubble "u v w xi eta zeta th")
# Terrain + the only physics-off default_case beyond the 32^2 set. 7 s.
add_case_test(mountain 1 "")

# --- physics: 1 GPU, ~80 s. P3, RRTMGP and the Noah land model -------------
if(VVM_TEST_PHYSICS)
    add_case_test(evergreen          1 physics)
    add_case_test(grass              1 physics)
    add_case_test(urban              1 physics)
    add_case_test(sea_grass_mountain 1 physics)
    add_case_test(sea_urban_mountain 1 physics)
    add_case_test(rcemip             1 physics)
endif()

# --- multirank: needs 4 real GPUs, ~30 s ----------------------------------
# 4 ranks gives a 2x2 decomposition. This is the tier that would have caught the
# initialiser bug where each rank stamped its own perturbation from local indices;
# 2 ranks splits one direction only and misses it.
if(VVM_TEST_MULTIRANK)
    add_rank_invariance_test(advection_u 4)
    add_rank_invariance_test(twisting    4)
    add_rank_invariance_test(2dbubble    4)
endif()

# --- large: needs 8 GPUs and the 961 MB taiwanvvm_2048.nc, ~140 s ---------
if(VVM_TEST_LARGE)
    # Two overrides the shipped config forces on us:
    #   fields  - the full ~60-field dump is ~46 GB at 2048^2 and overruns the
    #             synchronous HDF5 writer used by the tests.
    #   chunk   - default_cases still carries column_chunk_size 16384, which OOMs
    #             on 8 GPUs; 8192 is the working value.
    add_case_test(taiwanvvm_2048 8 large --fields u,v,w,th --rrtmgp-chunk 8192)
endif()

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
    endif()
    list(REMOVE_DUPLICATES VVM_CPU_RUNTIME_DIRS)

    # OpenMP is the execution space for a CPU build, and ctest launches the binary
    # directly rather than through submit.py/core_run.sh, so nothing else sets the
    # thread count. Without this each rank inherits whatever OMP_NUM_THREADS the
    # caller happens to have -- usually unset, which means "one thread per core"
    # and makes multi-rank tests oversubscribe badly.
    set(VVM_TEST_CPU_THREADS 16 CACHE STRING
        "OMP_NUM_THREADS per rank for CPU tests")

    set(VVM_TEST_ENV_MODS "OMP_NUM_THREADS=set:${VVM_TEST_CPU_THREADS}")

    if(VVM_CPU_RUNTIME_DIRS)
        string(REPLACE ";" ":" VVM_CPU_LD_PREPEND "${VVM_CPU_RUNTIME_DIRS}")
        message(STATUS "CPU tests: prepending to LD_LIBRARY_PATH: ${VVM_CPU_LD_PREPEND}")
        list(APPEND VVM_TEST_ENV_MODS
             "LD_LIBRARY_PATH=path_list_prepend:${VVM_CPU_LD_PREPEND}")
    endif()

    cmake_host_system_information(RESULT VVM_HOST_CORES QUERY NUMBER_OF_LOGICAL_CORES)
    message(STATUS "CPU tests: OMP_NUM_THREADS=${VVM_TEST_CPU_THREADS} per rank "
                   "(host reports ${VVM_HOST_CORES} logical cores)")
    if(VVM_TEST_CPU_THREADS GREATER VVM_HOST_CORES)
        message(WARNING
            "CPU tests are set to ${VVM_TEST_CPU_THREADS} threads/rank but this host has "
            "only ${VVM_HOST_CORES} logical cores, so they will oversubscribe and run "
            "slowly. Results stay correct. Override with -DVVM_TEST_CPU_THREADS=<n>.")
    endif()

    get_property(VVM_ALL_TESTS DIRECTORY PROPERTY TESTS)
    if(VVM_ALL_TESTS)
        set_tests_properties(${VVM_ALL_TESTS} PROPERTIES
            ENVIRONMENT_MODIFICATION "${VVM_TEST_ENV_MODS}")
    endif()
endif()
