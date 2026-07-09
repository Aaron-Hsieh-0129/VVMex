find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Use VVM_ROOT as the runtime working directory for CTest tests.
# Root CMakeLists.txt already requires VVM_ROOT, so this should normally exist.
set(VVM_TEST_WORKDIR "$ENV{VVM_ROOT}")
if(VVM_TEST_WORKDIR STREQUAL "")
  message(FATAL_ERROR "VVM_ROOT is not set. Please export VVM_ROOT=/path/to/VVMex before configuring tests.")
endif()

file(TO_CMAKE_PATH "${VVM_TEST_WORKDIR}" VVM_TEST_WORKDIR)

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

    set_tests_properties(Verify_${TEST_NAME} PROPERTIES DEPENDS Run_${TEST_NAME})
endfunction()

add_vvm_test(advection_u "u th xi eta zeta")
add_vvm_test(advection_v "v th xi eta zeta")
add_vvm_test(advection_w "w th xi eta")
add_vvm_test(stretching "u v w xi eta zeta")
add_vvm_test(twisting "u v w xi eta zeta")
add_vvm_test(2dbubble "u v w xi eta zeta th")
