# Included from tests/CMakeLists.txt so existing test helpers and backend
# settings remain shared. Keep the CTest name stable after relocation.
get_filename_component(_jung_build_name "${CMAKE_BINARY_DIR}" NAME)
add_test(NAME test_jung2019_shared_model
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check_jung2019_model.py
        $<TARGET_FILE:vvm> ${CMAKE_SOURCE_DIR}/experiments/jung2019/testing_output/${_jung_build_name}
        --wrapper "${GPU_WRAP}" --launcher ${MPIEXEC_EXECUTABLE})
set_tests_properties(test_jung2019_shared_model PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}" LABELS "integration" RESOURCE_LOCK gpu TIMEOUT 300)
_vvm_set_test_resources(test_jung2019_shared_model 1)
unset(_jung_build_name)
