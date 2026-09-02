set(LAND_FORTRAN_SRCS
    mod_rank.f90
    mod_index.f90
    machine.f90
    mod_param.f90
    mod_const.f90
    physcons.f90
    namelist_soilveg.f90
    set_soilveg.f90
    fpvs_gpu.f90
    sfc_diff_gpu.f90
    sfc_ocean_gpu.f90
    sfc_sice_gpu.f90
    sfc_diag_gpu.f90
    sflx_gpu.f90
    sfc_drv_gpu.f90
    vvm_land_interface.f90
)

add_library(vvm_land_fortran SHARED ${LAND_FORTRAN_SRCS})

set_target_properties(vvm_land_fortran PROPERTIES LINKER_LANGUAGE Fortran)

set_target_properties(vvm_land_fortran PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    # Without -acc the !$acc directives degrade to comments and the Noah routines
    # run serially on the host; LandProcess then skips the acc_map_data calls that
    # only make sense when device memory is distinct from host memory.
    if(VVM_ENABLE_GPU)
        target_compile_options(vvm_land_fortran PRIVATE -fast -acc)
    else()
        target_compile_options(vvm_land_fortran PRIVATE -fast)
    endif()

    target_compile_options(vvm_land_fortran PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-Mallocatable=03;-Mfreeform;-Mextend;-Mpreprocess>
    )

    if(VVM_USE_DOUBLE_PRECISION)
        target_compile_options(vvm_land_fortran PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-r8>)
    else()
        target_compile_definitions(vvm_land_fortran PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:SP>)
    endif()

    if(VVM_ENABLE_GPU)
        target_link_options(vvm_land_fortran PRIVATE -acc -fortranlibs -c++libs)
    else()
        target_link_options(vvm_land_fortran PRIVATE -fortranlibs -c++libs)
    endif()
endif()

add_custom_command(TARGET vvm_land_fortran PRE_LINK
    COMMAND rm -f $<TARGET_FILE:vvm_land_fortran>
    COMMENT "Auto-removing stale shared library to prevent pgacclnk incremental link crash..."
)

# LandProcess.cpp itself: STATIC, like every other Kokkos/CUDA physics module.
add_library(vvm_land STATIC LandProcess.cpp)

set_target_properties(vvm_land PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (VVM_ENABLE_GPU AND CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options(vvm_land PRIVATE -acc)
    target_link_options(vvm_land PRIVATE -acc)
endif()

target_link_libraries(vvm_land PUBLIC Kokkos::kokkos vvm_core vvm_land_fortran)
target_link_libraries(eamxx_physics INTERFACE vvm_land)
