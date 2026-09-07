add_executable(test_vertical_elliptic_solver
    ${TEST_DIR}/unit/test_vertical_elliptic_solver.cpp
)

target_link_libraries(test_vertical_elliptic_solver PRIVATE
    vvm_dynamics
    vvm_core
    vvm_utils
    Kokkos::kokkos
    MPI::MPI_CXX
)

add_executable(test_regular_latlon_wind_diagnostic
    ${TEST_DIR}/unit/test_regular_latlon_wind_diagnostic.cpp
)

target_link_libraries(test_regular_latlon_wind_diagnostic PRIVATE
    vvm_dynamics
    vvm_core
    vvm_utils
    Kokkos::kokkos
    MPI::MPI_CXX
)

set(_vertical_ranks 1)
if(VVM_TEST_MULTIRANK)
    list(APPEND _vertical_ranks 2 4)
endif()

foreach(_geometry IN ITEMS cartesian rll)
    foreach(_ranks IN LISTS _vertical_ranks)
        set(_test
            test_vertical_elliptic_solver_${_geometry}_r${_ranks})

        add_test(NAME ${_test}
            COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${_ranks}
                    ${VVM_MPI_BIND_ARGS}
                    ${MPIEXEC_PREFLAGS}
                    ${GPU_WRAP}
                    $<TARGET_FILE:test_vertical_elliptic_solver>
                    ${TEST_DIR}/configs/vertical_elliptic_${_geometry}.json
                    ${MPIEXEC_POSTFLAGS}
        )

        set_tests_properties(${_test} PROPERTIES
            WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
            LABELS "unit"
            TIMEOUT 300
        )

        _vvm_set_test_resources(
            ${_test}
            ${_ranks})
    endforeach()
endforeach()

foreach(_ranks IN LISTS _vertical_ranks)
    set(_test
        test_regular_latlon_wind_diagnostic_r${_ranks})

    add_test(NAME ${_test}
        COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} ${_ranks}
                ${VVM_MPI_BIND_ARGS}
                ${MPIEXEC_PREFLAGS}
                ${GPU_WRAP}
                $<TARGET_FILE:test_regular_latlon_wind_diagnostic>
                ${TEST_DIR}/configs/vertical_elliptic_rll.json
                ${MPIEXEC_POSTFLAGS}
    )

    set_tests_properties(${_test} PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        LABELS "unit"
        TIMEOUT 600
    )

    _vvm_set_test_resources(
        ${_test}
        ${_ranks})
endforeach()
