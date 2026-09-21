vvm_add_test_executable(test_vertical_wind_diagnostic
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

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

# Full nonorthogonal CVVM RELAX_3D metric terms using the same manufactured
# 2-D chart as the generalized wind-recovery tests. This is a local operator
# test; panel topology and halo transforms are intentionally outside scope.
vvm_add_test_executable(test_generalized_vertical_wind_diagnostic
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(NAME test_generalized_vertical_wind_diagnostic
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_generalized_vertical_wind_diagnostic>
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(test_generalized_vertical_wind_diagnostic PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    LABELS "unit"
    TIMEOUT 120
)

_vvm_set_test_resources(test_generalized_vertical_wind_diagnostic 1)

