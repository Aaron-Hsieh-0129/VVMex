message(STATUS "Fetching CLEO library...")

include(FetchContent)
FetchContent_Declare(
  cleo
  GIT_REPOSITORY https://github.com/Lance798/CLEO.git
  GIT_TAG        649d4bcc654153b4184eca735739761762de3783
  FIND_PACKAGE_ARGS 0.69 NAMES CLEO
)

set(CLEO_COUPLED_DYNAMICS "null")
set(CLEO_DOMAIN "cartesian")
FetchContent_MakeAvailable(cleo)

