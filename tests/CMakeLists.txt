# Test selection and registration API: see tests/README.md.
include(cmake/TestOptions.cmake)
include(cmake/TestHelpers.cmake)

include(cmake/UnitTests.cmake)
include(cmake/IOTests.cmake)
if(VVM_TEST_REGRESSION)
    include(cmake/RegressionTests.cmake)
endif()
include(cmake/RLLTests.cmake)
include(cmake/OptionalTests.cmake)

# Apply runtime policy after every suite has registered its tests.
include(cmake/TestEnvironment.cmake)
