add_library(vvm_cleo VVM_cleo_process_interface.cpp)

set_target_properties(vvm_cleo PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

target_link_libraries(vvm_cleo
    PUBLIC
        vvm_core
        vvm_utils
        Kokkos::kokkos
    PRIVATE
        CLEO::runcleo
        CLEO::cartesiandomain
        CLEO::movement
        CLEO::gridboxes
        CLEO::superdrops
        CLEO::initialise
        CLEO::observers
        CLEO::zarr
        CLEO::configuration
)

option(VVM_CLEO_DEBUG "Build the CLEO interface at -O0 -g for debugging" OFF)
if(VVM_CLEO_DEBUG)
    message(STATUS "VVM CLEO debug build: cleo_interface.cpp at -O0 -g")
    target_compile_options(vvm_cleo PRIVATE -O0 -g)
endif()

target_include_directories(vvm_cleo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_NCCL)
    target_include_directories(vvm_cleo PUBLIC ${NCCL_INCLUDE_DIR})
endif()
