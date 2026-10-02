message(STATUS "Fetching CLEO library...")

include(FetchContent)
FetchContent_Declare(
  cleo
  GIT_REPOSITORY https://github.com/Lance798/CLEO.git
  GIT_TAG        b92eb77
  FIND_PACKAGE_ARGS 0.69 NAMES CLEO
)

set(CLEO_COUPLED_DYNAMICS "null")
set(CLEO_DOMAIN "cartesian")
FetchContent_MakeAvailable(cleo)

