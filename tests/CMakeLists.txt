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
# backend; cross-backend agreement is a separate question, and one that
# -DVVM_DETERMINISTIC_FP=ON settles for the dry cases -- see
# docs/developer-guides/reproducibility.md.
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
#   default                always on   advection_u/v/w, stretching, twisting,
#                                      2dbubble, mountain, plus the unit tests
#   VVM_TEST_BP5           opt-in      direct BP5 output: unit, MPI, model
#   VVM_TEST_SST           opt-in      SST relay precision (needs a working SST)
#   VVM_TEST_PHYSICS       GPU only    P3 / RRTMGP / Noah land cases
#   VVM_TEST_MULTIRANK     4 ranks     rank invariance (2-D decomposition)
#
# On a GPU build the default tier needs exactly ONE device. Every test that
# touches a GPU runs at one rank and holds RESOURCE_LOCK gpu, so `ctest -j`
# never puts two of them on the same device. The one multi-rank test in the
# tier, test_restart_metadata_io_r2, links no Kokkos and touches no device.
# Only VVM_TEST_MULTIRANK (4 GPUs) and VVM_TEST_SST raise that.
#
# Pick which device with VVM_TEST_GPUS, a comma-separated list read by
# scripts/one_gpu_per_rank.sh -- useful when the low-numbered GPUs are busy:
#
#   VVM_TEST_GPUS=6 ctest --test-dir build
#
# Enable one or several at configure time:
#
#   cmake --preset f1-cpu -DBUILD_TESTS=ON -DVVM_TEST_BP5=ON
#   cmake --preset blaze  -DBUILD_TESTS=ON -DVVM_TEST_PHYSICS=ON -DVVM_TEST_MULTIRANK=ON
#
# Opt-in at CONFIGURE time rather than label-filtered, because `ctest` with no
# arguments runs every registered test -- `-L` filters down to a label, it does
# not hold anything back. Registering a tier only when asked is what actually
# keeps it off a default run. Once enabled, each tier still carries a label, so
# `ctest -L bp5` or `ctest -L multirank` selects it.
#
# Regression digests are bit-for-bit by default. Random perturbations are
# disabled unless a case explicitly enables them; RCEMIP does, because its
# fixed seed and global-coordinate mapping are deterministic. RCEMIP's stored
# reference is the one portability exception: it hashes every value after a
# canonical float32 narrowing so harmless low FP64 bits may differ across GPUs.
#
# On a GPU build the multi-rank tiers need that many *physical* GPUs. Several
# NCCL ranks sharing one device is unsupported and silently changes results, so
# they cannot be faked on a smaller machine -- see scripts/one_gpu_per_rank.sh.
# That makes rank count an expensive axis, so it is swept where the answer can
# actually depend on it (decomposition, halo exchange, global reductions,
# per-rank output slabs) and pinned at one value everywhere else.
# ---------------------------------------------------------------------------
option(VVM_TEST_BP5       "Register direct BP5 output tests (CPU and GPU builds)"     OFF)
option(VVM_TEST_SST       "Register the SST relay precision test (2 ranks, network)"  OFF)
option(VVM_TEST_PHYSICS   "Register physics-tier tests (GPU builds only, ~80 s)"      OFF)
option(VVM_TEST_MULTIRANK "Register rank-invariance tests (4 ranks, ~30 s)"           OFF)

# ---------------------------------------------------------------------------
# What a test occupies while it runs.
#
# On a GPU build the scarce resource is the device. Several NCCL ranks sharing
# one GPU is unsupported and silently changes results, so anything that touches
# a device takes a hard lock and the suite runs one model at a time.
#
# On a CPU build the scarce resource is cores, and there is no reason two small
# cases cannot run side by side: they are separate processes writing separate
# output directories, and thread count does not change results. So instead of a
# lock, each test declares how many cores it wants (ranks x threads) and
# `ctest -j <cores>` packs them without oversubscribing. The default cases are
# 32x32x33 and 512x16x74 grids -- far too small to keep 224 cores busy one at a
# time, which is what the lock forced.
# ---------------------------------------------------------------------------
set(VVM_TEST_CPU_THREADS 64 CACHE STRING
    "OMP_NUM_THREADS per rank for CPU tests")

function(_vvm_set_test_resources TEST_NAME RANKS)
    if(VVM_ENABLE_GPU)
        set_tests_properties(${TEST_NAME} PROPERTIES RESOURCE_LOCK gpu)
    else()
        math(EXPR _vvm_procs "${RANKS} * ${VVM_TEST_CPU_THREADS}")
        set_tests_properties(${TEST_NAME} PROPERTIES PROCESSORS ${_vvm_procs})
    endif()
endfunction()

# A tier the current backend cannot run is switched off here, with the reason.
# Silently registering nothing would look like the tier passed.
if(VVM_TEST_PHYSICS AND NOT VVM_ENABLE_GPU)
    message(WARNING
        "VVM_TEST_PHYSICS=ON ignored: no CPU reference data exists under "
        "tests/references_cpu/ for the physics cases.")
    set(VVM_TEST_PHYSICS OFF)
endif()

# Report the tiers the way the rest of the configure output reports its options,
# including which backend each one needs and how to switch it on.
function(_report_test_tier LABEL STATE NEEDS OPT_NAME)
    if(STATE)
        message(STATUS "  Test tier ${LABEL}: ON   (${NEEDS})")
    else()
        message(STATUS "  Test tier ${LABEL}: OFF  (${NEEDS}) -- enable with -D${OPT_NAME}=ON")
    endif()
endfunction()

message(STATUS "Test tiers (native digests are bit-for-bit; RCEMIP uses canonical float32; simulated seconds per case: ${VVM_TEST_SECONDS}):")
if(VVM_ENABLE_GPU)
    message(STATUS "  Test tier default  : ON   (advection x3, stretching, twisting, 2dbubble, mountain, unit tests) -- 1 GPU")
else()
    message(STATUS "  Test tier default  : ON   (advection x3, stretching, twisting, 2dbubble, mountain, unit tests)")
endif()
_report_test_tier("bp5      " ${VVM_TEST_BP5}       "CPU or GPU build"  VVM_TEST_BP5)
_report_test_tier("sst      " ${VVM_TEST_SST}       "2 ranks, SST"      VVM_TEST_SST)
_report_test_tier("physics  " ${VVM_TEST_PHYSICS}   "GPU build only"    VVM_TEST_PHYSICS)
_report_test_tier("multirank" ${VVM_TEST_MULTIRANK} "4 ranks / 4 GPUs"  VVM_TEST_MULTIRANK)
message(STATUS "  Select the test device with VVM_TEST_GPUS (e.g. VVM_TEST_GPUS=6 ctest --test-dir build)")

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

    # Through GPU_WRAP like every other device test: at one rank it only pins
    # CUDA_VISIBLE_DEVICES to a single device, which is what lets VVM_TEST_GPUS
    # move the whole default tier off a busy GPU.
    add_test(NAME Run_${TEST_NAME}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                     ${GPU_WRAP} $<TARGET_FILE:vvm> ${CFG}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    separate_arguments(VARS_LIST UNIX_COMMAND "${VARS_TO_TEST}")

    add_test(NAME Verify_${TEST_NAME}
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/verify_output.py
             --baseline ${VVM_BASELINE_DIR}/${TEST_NAME}.h5
             --test_out ${ODIR}/vvm_output_000001.h5
             --vars ${VARS_LIST}
             --tol 1e-6)

    # A hung MPI run has no other way to end: without a timeout ctest waits for
    # it forever, which turns one wedged rank into a stuck unattended suite.
    # Generous against the slowest measured CPU run of this tier.
    set_tests_properties(Run_${TEST_NAME}    PROPERTIES TIMEOUT 1800)
    _vvm_set_test_resources(Run_${TEST_NAME} 1)
    set_tests_properties(Verify_${TEST_NAME} PROPERTIES DEPENDS Run_${TEST_NAME} TIMEOUT 300)
endfunction()

# A default_cases case, checked against a stored per-variable digest. Full
# baselines do not scale here (rcemip output is ~1 GB), so check_output.py
# compares SHA-256 digests instead. A reference can self-declare canonical
# float32 narrowing when cross-machine FP64 low bits are not portable.
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

    # As above: mountain is 409 s on a CPU build, so the bound is generous, but
    # an unbounded MPI run is what makes an unattended suite hang.
    set_tests_properties(Run_${TAG}    PROPERTIES DEPENDS Prep_${TAG} TIMEOUT 1800)
    _vvm_set_test_resources(Run_${TAG} ${RANKS})
    set_tests_properties(Verify_${TAG} PROPERTIES DEPENDS Run_${TAG} TIMEOUT 300)
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

    # These are the only tests in the tree with no CTest timeout of their own,
    # and on a CPU build they are also the slowest by a wide margin, so a run
    # that wedges blocks the whole suite indefinitely rather than failing.
    # 3600 s is roughly 3x the slowest measured run (twisting at 4 CPU ranks,
    # 1237 s) -- long enough not to fire on a loaded machine, short enough that
    # an unattended suite still terminates.
    set_tests_properties(Run_${TAG}_r1       PROPERTIES DEPENDS Prep_${TAG} TIMEOUT 3600)
    _vvm_set_test_resources(Run_${TAG}_r1 1)
    set_tests_properties(Run_${TAG}_r${RANKS} PROPERTIES DEPENDS Run_${TAG}_r1 TIMEOUT 3600)
    _vvm_set_test_resources(Run_${TAG}_r${RANKS} ${RANKS})
    set_tests_properties(Verify_${TAG}       PROPERTIES DEPENDS Run_${TAG}_r${RANKS}
                         TIMEOUT 300)
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
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS unit TIMEOUT 60)
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

add_test(NAME test_check_output_digest
         COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/test_check_output.py)
set_tests_properties(test_check_output_digest PROPERTIES LABELS unit TIMEOUT 60)

add_executable(test_deterministic_perturbation
               ${TEST_DIR}/unit/test_deterministic_perturbation.cpp)
target_include_directories(test_deterministic_perturbation PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_deterministic_perturbation PRIVATE Kokkos::kokkos)
add_test(NAME test_deterministic_perturbation
         COMMAND ${GPU_WRAP} $<TARGET_FILE:test_deterministic_perturbation>)
set_tests_properties(test_deterministic_perturbation PROPERTIES LABELS unit TIMEOUT 60)
_vvm_set_test_resources(test_deterministic_perturbation 1)

# The GrADS descriptor both output engines emit: the header lines that differ
# per engine, and the name mangling GrADS forces (lowercase, 15 characters).
# Links vvm_io rather than being header-only, so it is registered by hand.
add_executable(test_grads_ctl ${TEST_DIR}/unit/test_grads_ctl.cpp)
target_include_directories(test_grads_ctl PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_grads_ctl PRIVATE vvm_io Kokkos::kokkos MPI::MPI_CXX)
add_test(NAME test_grads_ctl COMMAND test_grads_ctl)
set_tests_properties(test_grads_ctl PROPERTIES LABELS unit TIMEOUT 60)

# Output precision: the engine-neutral output.precision key, its aliases, and
# how the BP5 block overrides it. Launched through mpirun because
# ConfigurationManager reads MPI_COMM_WORLD.
add_executable(test_output_precision ${TEST_DIR}/unit/test_output_precision.cpp)
target_include_directories(test_output_precision PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_output_precision PRIVATE vvm_io vvm_utils Kokkos::kokkos MPI::MPI_CXX)
add_test(NAME test_output_precision
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 $<TARGET_FILE:test_output_precision>)
set_tests_properties(test_output_precision PROPERTIES LABELS unit TIMEOUT 60)

add_executable(test_numerical_configuration
               ${TEST_DIR}/unit/test_numerical_configuration.cpp)
target_include_directories(test_numerical_configuration PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_numerical_configuration PRIVATE vvm_utils MPI::MPI_CXX)
add_test(NAME test_numerical_configuration
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 $<TARGET_FILE:test_numerical_configuration>)
set_tests_properties(test_numerical_configuration PROPERTIES LABELS unit TIMEOUT 60)

# --- output precision on the legacy writer ---------------------------------
# output.precision decides the on-disk field type for HDF5 and SST as it does
# for BP5. The HDF5 case writes a real file and reads its dataset types back;
# 'unset' is the pre-existing behaviour and guards against a silent change to
# the default path. One rank, one GPU, no network.
#
# An explicit "native" is not run here: it resolves to the same on-disk type as
# 'unset', so the file it writes and every assertion made about it are identical.
# That the key itself parses to Native is test_output_precision's job.
set(OUTPUT_PRECISION_WORKDIR "${CMAKE_BINARY_DIR}/output_precision")
add_executable(test_hdf5_precision integration/test_hdf5_precision.cpp)
target_include_directories(test_hdf5_precision PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_hdf5_precision PRIVATE
                      vvm_io vvm_core vvm_utils HDF5::HL Kokkos::kokkos MPI::MPI_CXX)
foreach(PRECISION unset float32 float64)
    add_test(NAME test_hdf5_precision_${PRECISION}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                     ${GPU_WRAP} $<TARGET_FILE:test_hdf5_precision>
                     ${TEST_DIR}/configs/advection_u.json
                     ${OUTPUT_PRECISION_WORKDIR} ${PRECISION}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(test_hdf5_precision_${PRECISION} PROPERTIES
                         LABELS "integration;precision" TIMEOUT 120)
    _vvm_set_test_resources(test_hdf5_precision_${PRECISION} 1)
endforeach()

# The SST relay carries the same precision through the I/O server. Opt-in
# because it opens a real SST stream, which depends on the machine's network
# rather than on this code.
if(VVM_TEST_SST)
    add_executable(test_sst_precision integration/test_sst_precision.cpp)
    target_include_directories(test_sst_precision PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_sst_precision PRIVATE
                          vvm_io vvm_core vvm_utils HDF5::HL Kokkos::kokkos MPI::MPI_CXX)
    foreach(PRECISION unset float32)
        add_test(NAME test_sst_precision_${PRECISION}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${VVM_MPI_BIND_ARGS}
                         ${GPU_WRAP} $<TARGET_FILE:test_sst_precision>
                         ${TEST_DIR}/configs/advection_u.json
                         ${OUTPUT_PRECISION_WORKDIR} ${PRECISION}
                 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(test_sst_precision_${PRECISION} PROPERTIES
                             LABELS "integration;precision;sst" TIMEOUT 300)
        _vvm_set_test_resources(test_sst_precision_${PRECISION} 2)
    endforeach()
endif()

# --- bp5 tier: opt-in with -DVVM_TEST_BP5=ON -------------------------------
# Everything below writes and reads real BP5 datasets under
# ${CMAKE_BINARY_DIR}, so it is off by default rather than paid for on every
# `ctest`.
if(VVM_TEST_BP5)
    # Direct BP5 pure logic and CPU staging. These link the isolated I/O
    # subsystem but do not open a dataset or require multiple ranks.
    foreach(BP5_UNIT_TEST
            test_bp5_output_config
            test_bp5_field_schema
            test_bp5_path_policy
            test_cpu_field_source)
        add_executable(${BP5_UNIT_TEST} ${TEST_DIR}/unit/${BP5_UNIT_TEST}.cpp)
        target_include_directories(${BP5_UNIT_TEST} PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(${BP5_UNIT_TEST} PRIVATE vvm_io Kokkos::kokkos MPI::MPI_CXX)
        add_test(NAME ${BP5_UNIT_TEST} COMMAND ${BP5_UNIT_TEST})
        set_tests_properties(${BP5_UNIT_TEST} PROPERTIES LABELS "unit;bp5" TIMEOUT 60)
    endforeach()

    add_executable(test_bp5_history_mpi integration/test_bp5_history_mpi.cpp)
    target_include_directories(test_bp5_history_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bp5_history_mpi PRIVATE
                          vvm_io vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

    add_executable(test_bp5_collective_config integration/test_bp5_collective_config.cpp)
    target_include_directories(test_bp5_collective_config PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bp5_collective_config PRIVATE vvm_io MPI::MPI_CXX)
    add_test(NAME test_bp5_collective_config_r2
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} $<TARGET_FILE:test_bp5_collective_config>)
    set_tests_properties(test_bp5_collective_config_r2 PROPERTIES
                         LABELS "bp5;integration;failure" TIMEOUT 30)
    _vvm_set_test_resources(test_bp5_collective_config_r2 2)

    # One-rank readback after 1/2/4-rank writes proves reader/writer process
    # counts need not match. The four-rank selection deliberately leaves ranks
    # with no selected cells.
    #
    # Rank count is swept on the default path only (direct + sync), because it
    # is the writer's own axis: r1 writes an undecomposed domain, r2 splits it,
    # and r4 is the only case that hands some ranks an empty selection. The
    # other axes -- buffer mode, async, precision -- are per-rank-local choices
    # that meet the decomposition through the same Selection object, so running
    # each of them at all three counts re-tested the same slab arithmetic for
    # four more GPUs. They are pinned at r2, the smallest count that still
    # decomposes. 'pack' keeps its r4 run: staging an empty selection is a
    # zero-length buffer, and that is the one case where the two modes can
    # genuinely differ.
    set(BP5_INTEGRATION_WORKDIR "${CMAKE_BINARY_DIR}/bp5_integration")
    foreach(BP5_CASE direct:1 direct:2 direct:4 pack:2 pack:4)
        string(REPLACE ":" ";" BP5_CASE_PARTS ${BP5_CASE})
        list(GET BP5_CASE_PARTS 0 BP5_MODE)
        list(GET BP5_CASE_PARTS 1 BP5_RANKS)
        set(BP5_TEST_NAME test_bp5_history_${BP5_MODE}_sync_r${BP5_RANKS})
        add_test(NAME ${BP5_TEST_NAME}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${BP5_RANKS}
                         ${VVM_MPI_BIND_ARGS} ${GPU_WRAP}
                         $<TARGET_FILE:test_bp5_history_mpi>
                         ${TEST_DIR}/configs/advection_u.json
                         ${BP5_INTEGRATION_WORKDIR} ${BP5_MODE} sync ${BP5_RANKS}
                 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(${BP5_TEST_NAME} PROPERTIES
                             LABELS "bp5;integration;multirank" TIMEOUT 90)
        _vvm_set_test_resources(${BP5_TEST_NAME} ${BP5_RANKS})
    endforeach()

    add_test(NAME test_bp5_history_direct_async_r2
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP}
                     $<TARGET_FILE:test_bp5_history_mpi>
                     ${TEST_DIR}/configs/advection_u.json
                     ${BP5_INTEGRATION_WORKDIR} direct async 2
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(test_bp5_history_direct_async_r2 PROPERTIES
                         LABELS "bp5;integration;async;multirank" TIMEOUT 90)
    _vvm_set_test_resources(test_bp5_history_direct_async_r2 2)

    # Output precision is independent of VVM::Real, so both on-disk types are
    # exercised whichever way this build was configured. The test asserts the
    # variables really are the requested type, so on a double build float32 is
    # the converting case and float64 is the pass-through -- and vice versa.
    # 'direct' is requested deliberately: the converting case must resolve
    # itself to packing rather than fail.
    foreach(BP5_PRECISION float32 float64)
        set(BP5_TEST_NAME test_bp5_history_${BP5_PRECISION}_r2)
        add_test(NAME ${BP5_TEST_NAME}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                         ${VVM_MPI_BIND_ARGS} ${GPU_WRAP}
                         $<TARGET_FILE:test_bp5_history_mpi>
                         ${TEST_DIR}/configs/advection_u.json
                         ${BP5_INTEGRATION_WORKDIR} direct sync 2
                         ${BP5_PRECISION}
                 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(${BP5_TEST_NAME} PROPERTIES
                             LABELS "bp5;integration;precision;multirank" TIMEOUT 90)
        _vvm_set_test_resources(${BP5_TEST_NAME} 2)
    endforeach()

    # Reduced precision has to survive the async path too: it is the combination
    # where a staging buffer is both converted and handed to a background
    # writer thread.
    add_test(NAME test_bp5_history_float32_async_r2
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP}
                     $<TARGET_FILE:test_bp5_history_mpi>
                     ${TEST_DIR}/configs/advection_u.json
                     ${BP5_INTEGRATION_WORKDIR} pack async 2 float32
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(test_bp5_history_float32_async_r2 PROPERTIES
                         LABELS "bp5;integration;precision;async;multirank" TIMEOUT 90)
    _vvm_set_test_resources(test_bp5_history_float32_async_r2 2)

    # Keep model-smoke output in this backend's binary tree. The checked-in
    # configs use ./build for convenience, which would make a build_cpu run
    # overwrite or verify the GPU tree instead of testing backend coexistence.
    foreach(BP5_MODEL_CONFIG
            bp5_model_smoke
            bp5_model_async_smoke
            hdf5_model_smoke
            sst_model_smoke
            bp5_restart_source
            bp5_restart_append
            bp5_restart_resume)
        file(READ ${TEST_DIR}/configs/${BP5_MODEL_CONFIG}.json VVM_BP5_MODEL_CFG)
        set(VVM_BP5_MODEL_OUTPUT_NAME ${BP5_MODEL_CONFIG})
        if(BP5_MODEL_CONFIG STREQUAL "bp5_restart_append")
            set(VVM_BP5_MODEL_OUTPUT_NAME bp5_restart_source)
        endif()
        string(REGEX REPLACE "\"output_dir\"[ \t]*:[ \t]*\"[^\"]*\""
                             "\"output_dir\": \"${CMAKE_BINARY_DIR}/testing_output_${VVM_BP5_MODEL_OUTPUT_NAME}\""
                             VVM_BP5_MODEL_CFG "${VVM_BP5_MODEL_CFG}")
        # The restart source has to follow the same tree, for the same reason.
        string(REGEX REPLACE "\"source_file\"[ \t]*:[ \t]*\"[^\"]*history.bp\""
                             "\"source_file\": \"${CMAKE_BINARY_DIR}/testing_output_bp5_restart_source/history.bp\""
                             VVM_BP5_MODEL_CFG "${VVM_BP5_MODEL_CFG}")
        set(${BP5_MODEL_CONFIG}_CONFIG
            ${CMAKE_BINARY_DIR}/test_configs/${BP5_MODEL_CONFIG}.json)
        file(WRITE ${${BP5_MODEL_CONFIG}_CONFIG} "${VVM_BP5_MODEL_CFG}")
    endforeach()
    add_executable(test_bp5_model_output integration/test_bp5_model_output.cpp)
    target_include_directories(test_bp5_model_output PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bp5_model_output PRIVATE
                          adios2::adios2 Kokkos::kokkos MPI::MPI_CXX)
    add_test(NAME Run_bp5_model_smoke
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP} $<TARGET_FILE:vvm>
                     ${bp5_model_smoke_CONFIG}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    add_test(NAME Verify_bp5_model_smoke
             COMMAND $<TARGET_FILE:test_bp5_model_output>
                     ${CMAKE_BINARY_DIR}/testing_output_bp5_model_smoke/history.bp
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(Run_bp5_model_smoke PROPERTIES
                         LABELS "bp5;integration;model" TIMEOUT 240)
    _vvm_set_test_resources(Run_bp5_model_smoke 2)
    set_tests_properties(Verify_bp5_model_smoke PROPERTIES
                         DEPENDS Run_bp5_model_smoke LABELS "bp5;integration;model" TIMEOUT 30)

    add_test(NAME Run_bp5_model_async_smoke
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP} $<TARGET_FILE:vvm>
                     ${bp5_model_async_smoke_CONFIG}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    add_test(NAME Verify_bp5_model_async_smoke
             COMMAND $<TARGET_FILE:test_bp5_model_output>
                     ${CMAKE_BINARY_DIR}/testing_output_bp5_model_async_smoke/history.bp
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(Run_bp5_model_async_smoke PROPERTIES
                         LABELS "bp5;integration;model;async" TIMEOUT 240)
    _vvm_set_test_resources(Run_bp5_model_async_smoke 2)
    set_tests_properties(Verify_bp5_model_async_smoke PROPERTIES
                         DEPENDS Run_bp5_model_async_smoke
                         LABELS "bp5;integration;model;async" TIMEOUT 30)

    # Restart from a .bp dataset. The resumed run is configured to take no
    # further steps, so its first output is the loaded state itself and can be
    # compared against the source step -- fields, per-rank slabs, and clock.
    add_executable(test_bp5_restart integration/test_bp5_restart.cpp)
    target_include_directories(test_bp5_restart PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bp5_restart PRIVATE
                          adios2::adios2 HDF5::HL Kokkos::kokkos MPI::MPI_CXX)
    add_test(NAME Run_bp5_restart_source
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP} $<TARGET_FILE:vvm>
                     ${bp5_restart_source_CONFIG}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    add_test(NAME Run_bp5_restart_resume
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP} $<TARGET_FILE:vvm>
                     ${bp5_restart_resume_CONFIG}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    add_test(NAME Verify_bp5_restart
             COMMAND $<TARGET_FILE:test_bp5_restart>
                     ${CMAKE_BINARY_DIR}/testing_output_bp5_restart_source/history.bp 10
                     ${CMAKE_BINARY_DIR}/testing_output_bp5_restart_resume/history_000010.h5
                     thbar rhobar topo u v w th qv xi eta zeta
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(Run_bp5_restart_source PROPERTIES
                         LABELS "bp5;integration;restart" TIMEOUT 240)
    _vvm_set_test_resources(Run_bp5_restart_source 2)
    set_tests_properties(Run_bp5_restart_resume PROPERTIES
                         DEPENDS Run_bp5_restart_source
                         LABELS "bp5;integration;restart" TIMEOUT 240)
    _vvm_set_test_resources(Run_bp5_restart_resume 2)
    set_tests_properties(Verify_bp5_restart PROPERTIES
                         DEPENDS Run_bp5_restart_resume
                         LABELS "bp5;integration;restart" TIMEOUT 60)

    add_test(NAME Run_bp5_restart_append
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP} $<TARGET_FILE:vvm>
                     ${bp5_restart_append_CONFIG}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    add_test(NAME Verify_bp5_restart_append
             COMMAND $<TARGET_FILE:test_bp5_model_output>
                     ${CMAKE_BINARY_DIR}/testing_output_bp5_restart_source/history.bp 12
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(Run_bp5_restart_append PROPERTIES
                         DEPENDS Verify_bp5_restart
                         LABELS "bp5;integration;restart;append" TIMEOUT 240)
    _vvm_set_test_resources(Run_bp5_restart_append 2)
    set_tests_properties(Verify_bp5_restart_append PROPERTIES
                         DEPENDS Run_bp5_restart_append
                         LABELS "bp5;integration;restart;append" TIMEOUT 60)

    # The three engines must agree bit-for-bit on the same run: same case, same
    # two compute ranks (SST adds one I/O rank on top), compared through each
    # format's own reader. Values and metadata both -- the SST relay once
    # delivered every number correctly while dropping attributes.
    add_executable(test_engine_output_compat integration/test_engine_output_compat.cpp)
    target_include_directories(test_engine_output_compat PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_engine_output_compat PRIVATE
                          adios2::adios2 HDF5::HL Kokkos::kokkos MPI::MPI_CXX)
    add_test(NAME Run_hdf5_model_smoke
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP} $<TARGET_FILE:vvm>
                     ${hdf5_model_smoke_CONFIG}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    add_test(NAME Run_sst_model_smoke
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP} $<TARGET_FILE:vvm>
                     ${sst_model_smoke_CONFIG} --io-tasks 1
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    add_test(NAME Compare_engine_outputs_model_smoke
             COMMAND $<TARGET_FILE:test_engine_output_compat>
                     ${CMAKE_BINARY_DIR}/testing_output_bp5_model_smoke/history.bp
                     ${CMAKE_BINARY_DIR}/testing_output_hdf5_model_smoke
                     ${CMAKE_BINARY_DIR}/testing_output_sst_model_smoke
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(Run_hdf5_model_smoke PROPERTIES
                         LABELS "bp5;integration;compatibility" TIMEOUT 240)
    _vvm_set_test_resources(Run_hdf5_model_smoke 2)
    set_tests_properties(Run_sst_model_smoke PROPERTIES
                         LABELS "bp5;integration;compatibility;sst" TIMEOUT 300)
    _vvm_set_test_resources(Run_sst_model_smoke 3)
    set_tests_properties(Compare_engine_outputs_model_smoke PROPERTIES
                         DEPENDS "Run_bp5_model_smoke;Run_hdf5_model_smoke;Run_sst_model_smoke"
                         LABELS "bp5;integration;compatibility" TIMEOUT 60)
endif() # VVM_TEST_BP5

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
        set_tests_properties(${TEST_NAME}_r${N} PROPERTIES LABELS unit TIMEOUT 120)
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
    target_link_libraries(${TEST_NAME} PRIVATE vvm_core vvm_io vvm_utils Kokkos::kokkos MPI::MPI_CXX)

    add_test(NAME ${TEST_NAME}_r1
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                     ${GPU_WRAP} $<TARGET_FILE:${TEST_NAME}> ${CONFIG}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
    set_tests_properties(${TEST_NAME}_r1 PROPERTIES LABELS unit TIMEOUT 300)
    _vvm_set_test_resources(${TEST_NAME}_r1 1)

    if(VVM_TEST_MULTIRANK)
        add_test(NAME ${TEST_NAME}_r${RANKS}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS} ${VVM_MPI_BIND_ARGS}
                         ${GPU_WRAP} $<TARGET_FILE:${TEST_NAME}> ${CONFIG}
                 WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
        set_tests_properties(${TEST_NAME}_r${RANKS}
                             PROPERTIES LABELS "unit;multirank" TIMEOUT 300)
        _vvm_set_test_resources(${TEST_NAME}_r${RANKS} ${RANKS})
    endif()
endfunction()

# State::calculate_horizontal_mean(): the API shared by the NCCL and the
# standard-MPI backend. The 2-rank run is what pins the global reduction.
add_vvm_device_unit_test(test_horizontal_mean ${TEST_DIR}/configs/2dbubble.json 2)

# HaloExchanger: the NCCL and the standard-MPI implementation are separate code,
# and only a value-level check keeps them exchanging the same cells.
add_vvm_device_unit_test(test_halo_exchange ${TEST_DIR}/configs/2dbubble.json 2)

# Generate two shapes from the full test configuration so the same value-level
# contract runs with each horizontal axis reduced away. At two ranks, the
# remaining active axis is decomposed and still exercises MPI/NCCL.
function(add_reduced_halo_tests AXIS NX NY RANKS)
    file(READ ${TEST_DIR}/configs/2dbubble.json REDUCED_HALO_CONFIG)
    string(REGEX REPLACE
           "\"nx\"[ 	]*:[ 	]*[0-9]+"
           "\"nx\": ${NX}"
           REDUCED_HALO_CONFIG "${REDUCED_HALO_CONFIG}")
    string(REGEX REPLACE
           "\"ny\"[ 	]*:[ 	]*[0-9]+"
           "\"ny\": ${NY}"
           REDUCED_HALO_CONFIG "${REDUCED_HALO_CONFIG}")
    set(REDUCED_CONFIG
        ${CMAKE_BINARY_DIR}/test_configs/halo_exchange_${AXIS}.json)
    file(WRITE ${REDUCED_CONFIG} "${REDUCED_HALO_CONFIG}")

    add_test(NAME test_halo_exchange_${AXIS}_r1
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                     ${VVM_MPI_BIND_ARGS} ${GPU_WRAP}
                     $<TARGET_FILE:test_halo_exchange> ${REDUCED_CONFIG}
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
    set_tests_properties(test_halo_exchange_${AXIS}_r1
                         PROPERTIES LABELS unit TIMEOUT 300)
    _vvm_set_test_resources(test_halo_exchange_${AXIS}_r1 1)

    if(VVM_TEST_MULTIRANK)
        add_test(NAME test_halo_exchange_${AXIS}_r${RANKS}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS}
                         ${VVM_MPI_BIND_ARGS} ${GPU_WRAP}
                         $<TARGET_FILE:test_halo_exchange> ${REDUCED_CONFIG}
                 WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
        set_tests_properties(test_halo_exchange_${AXIS}_r${RANKS}
                             PROPERTIES LABELS "unit;multirank" TIMEOUT 300)
        _vvm_set_test_resources(test_halo_exchange_${AXIS}_r${RANKS} ${RANKS})
    endif()
endfunction()

add_reduced_halo_tests(nx1 1 32 2)
add_reduced_halo_tests(ny1 32 1 2)

add_executable(test_p3_multistage_history
               ${TEST_DIR}/unit/test_p3_multistage_history.cpp)
target_include_directories(test_p3_multistage_history
                           PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_p3_multistage_history
                      PRIVATE vvm_dynamics vvm_core vvm_io vvm_utils
                              Kokkos::kokkos MPI::MPI_CXX)
add_test(NAME test_p3_multistage_history_r1
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${VVM_MPI_BIND_ARGS} ${GPU_WRAP}
                 $<TARGET_FILE:test_p3_multistage_history>
                 ${TEST_DIR}/configs/p3_multistage_history.json
         WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
set_tests_properties(test_p3_multistage_history_r1
                     PROPERTIES LABELS unit TIMEOUT 300)
_vvm_set_test_resources(test_p3_multistage_history_r1 1)

# --- default: 1 GPU, ~40 s -------------------------------------------------
add_vvm_test(advection_u "u th xi eta zeta")
add_vvm_test(advection_v "v th xi eta zeta")
add_vvm_test(advection_w "w th xi eta")
add_vvm_test(stretching "u v w xi eta zeta")
add_vvm_test(twisting "u v w xi eta zeta")
add_vvm_test(2dbubble "u v w xi eta zeta th")
# Terrain + the only physics-off default_case beyond the 32^2 set. 7 s.
add_case_test(mountain 1 "")

# --- physics tier: GPU only, ~80 s. P3, RRTMGP and the Noah land model -----
# Forced OFF on a CPU build by the backend guard above: tests/references_cpu/
# has no digests for these cases, so they would fail for the wrong reason.
if(VVM_TEST_PHYSICS)
    add_case_test(evergreen          1 physics)
    add_case_test(grass              1 physics)
    add_case_test(urban              1 physics)
    add_case_test(sea_grass_mountain 1 physics)
    add_case_test(sea_urban_mountain 1 physics)
    add_case_test(rcemip             1 physics --enable-random-perturbation)
endif()

# --- multirank: needs 4 real GPUs, ~30 s ----------------------------------
# 4 ranks gives a 2x2 decomposition. This is the tier that would have caught the
# initialiser bug where each rank stamped its own perturbation from local indices;
# 2 ranks splits one direction only and misses it.
#
# Two cases, not three: every rank-invariance failure is a halo or a global-index
# bug, and what decides whether a case can expose one is which terms touch the
# halo, not which baseline it came from. twisting carries all three vorticity
# components and their cross-derivatives; 2dbubble adds buoyancy, thermodynamics
# and the full elliptic wind solve, the widest halo dependence in the model. The
# x-only advection of advection_u is a strict subset of both, so its pair of
# 4-GPU runs bought no coverage the other two did not already have.
#
# On a CPU build these are the slowest tests in the tree by a wide margin: the
# wind solver exchanges halos many times per step, and at 4 ranks over MPI on a
# 32^2 grid that is pure latency with no work to hide it (measured on blaze:
# twisting 1 s at 1 rank, 20 min at 4). Keep them out of any per-commit gate.
if(VVM_TEST_MULTIRANK)
    add_rank_invariance_test(twisting 4)
    add_rank_invariance_test(2dbubble 4)
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
