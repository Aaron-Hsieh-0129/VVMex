add_library(vvm_turbulence
    TurbulenceProcess.cpp
)

target_link_libraries(vvm_turbulence
    PUBLIC
    vvm_core
    vvm_utils
    Kokkos::kokkos
)

target_include_directories(vvm_turbulence
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_NCCL)
    target_include_directories(vvm_turbulence PUBLIC ${NCCL_INCLUDE_DIR})
endif()
