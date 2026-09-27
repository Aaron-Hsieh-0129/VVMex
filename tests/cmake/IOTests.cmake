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
