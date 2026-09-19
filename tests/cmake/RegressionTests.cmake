# --- default: 1 GPU, ~40 s -------------------------------------------------
add_vvm_test(advection_u "u th xi eta zeta")
add_vvm_test(advection_v "v th xi eta zeta")
add_vvm_test(advection_w "w th xi eta")
add_vvm_test(stretching "u v w xi eta zeta")
add_vvm_test(twisting "u v w xi eta zeta")
add_vvm_test(2dbubble "u v w xi eta zeta th")
# Terrain + the only physics-off default_case beyond the 32^2 set. 7 s.
add_case_test(mountain 1 "")

set(VVM_GRID_EQUIVALENCE_STEPS 12 CACHE STRING
    "Number of one-second steps for Cartesian configuration equivalence"
)

set(VVM_GRID_EQUIVALENCE_WORK
    "${CMAKE_BINARY_DIR}/testing_grid_configuration_equivalence"
)

set(VVM_GRID_EQUIVALENCE_SCRIPT
    "${TEST_DIR}/scripts/test_grid_configuration_equivalence.py"
)

add_test(
    NAME Prep_grid_configuration_equivalence
    COMMAND ${Python3_EXECUTABLE} "${VVM_GRID_EQUIVALENCE_SCRIPT}" prepare
            --case "${TEST_DIR}/configs/2dbubble.json"
            --work "${VVM_GRID_EQUIVALENCE_WORK}"
            --steps "${VVM_GRID_EQUIVALENCE_STEPS}"
)

set_tests_properties(Prep_grid_configuration_equivalence PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    FIXTURES_SETUP grid_configuration_prepared
    LABELS "integration"
    TIMEOUT 60
)

set(VVM_GRID_EQUIVALENCE_OUTPUT_FIXTURES "")

foreach(VARIANT IN ITEMS legacy horizontal vertical structured mixed)
    set(RUN_TEST "Run_grid_configuration_equivalence_${VARIANT}")
    set(OUTPUT_FIXTURE "grid_configuration_output_${VARIANT}")

    add_test(
        NAME ${RUN_TEST}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
                ${GPU_WRAP} $<TARGET_FILE:vvm> ${MPIEXEC_POSTFLAGS}
                "${VVM_GRID_EQUIVALENCE_WORK}/${VARIANT}.json"
                --io-tasks 0
    )

    set_tests_properties(${RUN_TEST} PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        FIXTURES_REQUIRED grid_configuration_prepared
        FIXTURES_SETUP "${OUTPUT_FIXTURE}"
        LABELS "integration"
        TIMEOUT 600
    )

    _vvm_set_test_resources(${RUN_TEST} 1)

    list(APPEND VVM_GRID_EQUIVALENCE_OUTPUT_FIXTURES "${OUTPUT_FIXTURE}")
endforeach()

add_test(
    NAME Verify_grid_configuration_equivalence
    COMMAND ${Python3_EXECUTABLE} "${VVM_GRID_EQUIVALENCE_SCRIPT}" verify
            --work "${VVM_GRID_EQUIVALENCE_WORK}"
)

set_tests_properties(Verify_grid_configuration_equivalence PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    FIXTURES_REQUIRED "${VVM_GRID_EQUIVALENCE_OUTPUT_FIXTURES}"
    LABELS "integration"
    TIMEOUT 120
)
