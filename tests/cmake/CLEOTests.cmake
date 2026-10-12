# --- cleo tier: superdroplet microphysics coupling (physics.microphysics.cleo)
# Every case but test_cleo_interface needs CLEO input binaries, which
# tools/cleo/gen_cleo_inputs.py makes with cleopy, so the tier is opt-in.
# Labels: cleo, plus unit or integration.
if(VVM_TEST_CLEO)
    # gen_cleo_inputs.py needs CLEO's own cleoconstants.hpp, which only the
    # checkout this build compiles against is guaranteed to agree with.
    # Without it the generator falls back to its own search.
    set(_vvm_cleo_source_args "")
    set(_vvm_cleo_model_args "")
    if(CLEO_SOURCE_DIR)
        set(_vvm_cleo_source_args --path2cleo ${CLEO_SOURCE_DIR})
        # vvm_add_model_test rejects an ARGS keyword with no values.
        set(_vvm_cleo_model_args ARGS ${_vvm_cleo_source_args})
    endif()
    set(_vvm_cleo_test_libraries
        vvm_cleo vvm_core vvm_io vvm_utils Kokkos::kokkos MPI::MPI_CXX)

    # The constructor's field contract, and that a regular_latlon grid is
    # refused. Reads no CLEO input.
    vvm_add_test_executable(test_cleo_interface LIBRARIES ${_vvm_cleo_test_libraries})
    add_test(NAME test_cleo_interface
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                     ${GPU_WRAP} $<TARGET_FILE:test_cleo_interface>
                     ${TEST_DIR}/configs/cleo_coupling.json
                     ${TEST_DIR}/configs/cleo_regular_latlon.json
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
    set_tests_properties(test_cleo_interface PROPERTIES LABELS "unit;cleo" TIMEOUT 120)
    _vvm_set_test_resources(test_cleo_interface 1)

    # One executable, one scenario per invocation; see the header of
    # unit/test_cleo_coupling.cpp. Each scenario gets its own generated config
    # and binaries, made by a fixture so `ctest -R` on one scenario still
    # prepares it. Extra arguments go to make_cleo_test_case.py.
    vvm_add_test_executable(test_cleo_coupling LIBRARIES ${_vvm_cleo_test_libraries})
    function(_vvm_add_cleo_prep SCENARIO)
        add_test(NAME Prep_cleo_${SCENARIO}
                 COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/make_cleo_test_case.py
                         --template ${TEST_DIR}/configs/cleo_coupling.json
                         --scenario ${SCENARIO} --out-dir ${CMAKE_BINARY_DIR}/cleo_tests/${SCENARIO}
                         ${_vvm_cleo_source_args} ${ARGN}
                 WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
        set_tests_properties(Prep_cleo_${SCENARIO} PROPERTIES
            LABELS "unit;cleo" TIMEOUT 300 FIXTURES_SETUP cleo_${SCENARIO})
    endfunction()
    # Runs <scenario> on RANKS ranks with the config named CONFIG in its case
    # directory; the test is test_cleo_coupling_<scenario>, plus _r<ranks>
    # above one rank.
    function(_vvm_add_cleo_run SCENARIO RANKS CONFIG)
        set(_name test_cleo_coupling_${SCENARIO})
        set(_labels "unit;cleo")
        if(RANKS GREATER 1)
            set(_name ${_name}_r${RANKS})
            set(_labels "unit;cleo;multirank")
        endif()
        add_test(NAME ${_name}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS} ${VVM_MPI_BIND_ARGS}
                         ${GPU_WRAP} $<TARGET_FILE:test_cleo_coupling>
                         ${CMAKE_BINARY_DIR}/cleo_tests/${SCENARIO}/${CONFIG} ${SCENARIO}
                 WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
        set_tests_properties(${_name} PROPERTIES
            LABELS "${_labels}" TIMEOUT 300 FIXTURES_REQUIRED cleo_${SCENARIO})
        _vvm_set_test_resources(${_name} ${RANKS})
    endfunction()

    foreach(SCENARIO transport condensation average_y rain rain_collision)
        _vvm_add_cleo_prep(${SCENARIO})
        _vvm_add_cleo_run(${SCENARIO} 1 config.json)
    endforeach()

    # Advection writes CLEO's zarr; check_cleo_superdroplets.py follows every
    # superdroplet through it. config_r4.json shares the binaries and writes
    # to its own output directory, for the 4-rank comparison below.
    set(_advection_dir ${CMAKE_BINARY_DIR}/cleo_tests/advection)
    _vvm_add_cleo_prep(advection --extra-output r4)
    _vvm_add_cleo_run(advection 1 config.json)
    set_tests_properties(test_cleo_coupling_advection PROPERTIES FIXTURES_SETUP cleo_advection_r1)
    add_test(NAME Verify_cleo_advection
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/check_cleo_superdroplets.py
                     --config ${_advection_dir}/config.json
             WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
    set_tests_properties(Verify_cleo_advection PROPERTIES
        LABELS "unit;cleo" TIMEOUT 300 FIXTURES_REQUIRED cleo_advection_r1)

    # The full model with CLEO: Model wiring, output, and CLEO's own zarr
    # record of what it was sent, compared cell by cell with the model's.
    vvm_add_model_test(test_cleo_model
        SCRIPT check_cleo_model.py OUTPUT cleo_model_tests
        TIMEOUT 900 LABELS cleo ${_vvm_cleo_model_args})

    # The same at 2 ranks against 1: CollectiveDataset's placement of each
    # rank's gridboxes and superdroplets is where the original coupling went
    # wrong without any error. Needs 2 GPUs, like the rest of the multirank tier.
    if(VVM_TEST_MULTIRANK)
        # 4 ranks split 2 x 2, and the wind carries every superdroplet across
        # rank boundaries, corners included, many times: lost or misrouted
        # superdroplets (the stripe bug) show up as a droplet-by-droplet
        # difference from the 1-rank run.
        _vvm_add_cleo_run(advection 4 config_r4.json)
        set_tests_properties(test_cleo_coupling_advection_r4 PROPERTIES
            FIXTURES_SETUP cleo_advection_r4)
        add_test(NAME Verify_cleo_advection_r4
                 COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/check_cleo_superdroplets.py
                         --config ${_advection_dir}/config_r4.json
                         --compare-to ${_advection_dir}/config.json
                 WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
        set_tests_properties(Verify_cleo_advection_r4 PROPERTIES
            LABELS "unit;cleo;multirank" TIMEOUT 300
            FIXTURES_REQUIRED "cleo_advection_r1;cleo_advection_r4")

        # 2 ranks with ny = 4 split y; average_over_y must then be refused.
        _vvm_add_cleo_prep(average_y_split)
        _vvm_add_cleo_run(average_y_split 2 config.json)

        vvm_add_model_test(test_cleo_model_r2
            SCRIPT check_cleo_model.py OUTPUT cleo_model_r2_tests
            TIMEOUT 1200 LABELS "cleo;multirank" ARGS --ranks 2 ${_vvm_cleo_source_args})
        _vvm_set_test_resources(test_cleo_model_r2 2)
    endif()
endif()
