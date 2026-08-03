add_library (eamxx_physics_share
  eamxx_trcmix.cpp
)

# Add ETI source files if not on CUDA/HIP. Without EAMXX_ENABLE_GPU,
# physics_functions.hpp stops including physics_saturation_impl.hpp, so the
# instantiations have to come from this object instead.
if (NOT EAMXX_ENABLE_GPU)
  target_sources(eamxx_physics_share PRIVATE
    physics_saturation.cpp)
  # EkatConfigFile leaves SCREAM_SMALL_PACK_SIZE commented out in eamxx_config.h,
  # so Spack (Pack<S,SCREAM_SMALL_PACK_SIZE>) will not resolve. P3 works around
  # this by defining the pack sizes on its own target; this ETI unit needs the
  # same. PRIVATE, and only on this branch, so the GPU build is unaffected.
  target_compile_definitions(eamxx_physics_share PRIVATE
    SCREAM_PACK_SIZE=${SCREAM_PACK_SIZE}
    SCREAM_SMALL_PACK_SIZE=${SCREAM_SMALL_PACK_SIZE}
    SCREAM_POSSIBLY_NO_PACK_SIZE=1)
endif()

set_target_properties(eamxx_physics_share PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
)
target_include_directories(eamxx_physics_share PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/modules
)
target_link_libraries(eamxx_physics_share PUBLIC
  eamxx_core
  eamxx_utils
  ekat::Pack)
