function(_vvm_set_test_resources TEST_NAME RANKS)
    if(VVM_ENABLE_GPU)
        set_tests_properties(${TEST_NAME} PROPERTIES RESOURCE_LOCK gpu)
    else()
        math(EXPR _vvm_procs "${RANKS} * ${VVM_TEST_CPU_THREADS}")
        set_tests_properties(${TEST_NAME} PROPERTIES PROCESSORS ${_vvm_procs})
    endif()
endfunction()

# Original regression cases compare against small stored baseline files.
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

# A single C++ test. DEVICE applies the GPU wrapper / CPU resource budget;
# MPI launches one rank. For configuration variants, build with
# vvm_add_test_executable() and register each invocation separately.
function(add_vvm_unit_test TEST_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 TEST "DEVICE;MPI" "TIMEOUT" "LIBRARIES;LABELS")
    if(TEST_UNPARSED_ARGUMENTS OR TEST_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Invalid arguments for unit test ${TEST_NAME}")
    endif()
    vvm_add_test_executable(${TEST_NAME} LIBRARIES ${TEST_LIBRARIES})
    set(command "")
    if(TEST_MPI)
        list(APPEND command ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS})
    endif()
    if(TEST_DEVICE)
        list(APPEND command ${GPU_WRAP})
    endif()
    list(APPEND command $<TARGET_FILE:${TEST_NAME}>)
    if(TEST_MPI)
        list(APPEND command ${MPIEXEC_POSTFLAGS})
    endif()
    if(NOT TEST_TIMEOUT)
        set(TEST_TIMEOUT 60)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${command})
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "unit;${TEST_LABELS}" TIMEOUT ${TEST_TIMEOUT})
    if(TEST_DEVICE)
        _vvm_set_test_resources(${TEST_NAME} 1)
    endif()
endfunction()

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

# Build once, then register any number of configurations with add_test().
function(vvm_add_test_executable NAME)
    cmake_parse_arguments(PARSE_ARGV 1 TEST "" "" "LIBRARIES")
    if(TEST_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments for ${NAME}: ${TEST_UNPARSED_ARGUMENTS}")
    endif()
    add_executable(${NAME} ${TEST_DIR}/unit/${NAME}.cpp)
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${NAME} PRIVATE ${TEST_LIBRARIES})
endfunction()

# Python model checks share the launcher, device wrapper and resource policy.
function(vvm_add_model_test NAME)
    cmake_parse_arguments(PARSE_ARGV 1 TEST "" "SCRIPT;OUTPUT;TIMEOUT" "LABELS;ARGS")
    if(TEST_UNPARSED_ARGUMENTS OR TEST_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Invalid arguments for model test ${NAME}")
    endif()
    if(NOT TEST_SCRIPT OR NOT TEST_OUTPUT)
        message(FATAL_ERROR "${NAME} requires SCRIPT and OUTPUT")
    endif()
    if(NOT TEST_TIMEOUT)
        set(TEST_TIMEOUT 300)
    endif()
    add_test(NAME ${NAME}
        COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/${TEST_SCRIPT}
            $<TARGET_FILE:vvm> ${CMAKE_BINARY_DIR}/${TEST_OUTPUT}
            --wrapper "${GPU_WRAP}" --launcher ${MPIEXEC_EXECUTABLE} ${TEST_ARGS})
    set_tests_properties(${NAME} PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        LABELS "integration;${TEST_LABELS}" TIMEOUT ${TEST_TIMEOUT})
    _vvm_set_test_resources(${NAME} 1)
endfunction()
