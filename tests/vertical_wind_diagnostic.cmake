add_executable(test_vertical_wind_diagnostic ${TEST_DIR}/unit/test_vertical_wind_diagnostic.cpp)
target_include_directories(test_vertical_wind_diagnostic PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_vertical_wind_diagnostic PRIVATE vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(NAME test_vertical_wind_diagnostic
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_vertical_wind_diagnostic>
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(test_vertical_wind_diagnostic PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    LABELS "unit"
    TIMEOUT 120
)

_vvm_set_test_resources(test_vertical_wind_diagnostic 1)
