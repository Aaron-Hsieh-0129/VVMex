find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rundata/input_configs/default_cases)
set(GPU_WRAP ${TEST_DIR}/scripts/one_gpu_per_rank.sh)

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
    add_test(NAME Run_${TEST_NAME}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                     $<TARGET_FILE:vvm> ${TEST_DIR}/configs/${TEST_NAME}.json
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    separate_arguments(VARS_LIST UNIX_COMMAND "${VARS_TO_TEST}")

    add_test(NAME Verify_${TEST_NAME}
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/verify_output.py
             --baseline ${TEST_DIR}/baselines/${TEST_NAME}.h5
             --test_out ${CMAKE_BINARY_DIR}/testing_output_${TEST_NAME}/vvm_output_000001.h5
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
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS}
                     ${GPU_WRAP} $<TARGET_FILE:vvm> ${CFG}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    add_test(NAME Verify_${TAG}
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/check_output.py
                     --output ${ODIR}/vvm_output_000001.h5
                     --reference ${TEST_DIR}/references/${CASE_NAME}.json)

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
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                     ${GPU_WRAP} $<TARGET_FILE:vvm> ${CFG1}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    add_test(NAME Run_${TAG}_r${RANKS}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS}
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
