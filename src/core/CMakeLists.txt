set(VVM_CORE_SOURCES
    State.cpp
    Grid.cpp
    GridSpecification.cpp
    BoundaryConditionManager.cpp
    Parameters.cpp

    geometry/CartesianGeometry.cpp
    geometry/RegularLatLonGeometry.cpp
    geometry/HorizontalGeometryFactory.cpp

    boundary/HorizontalBoundaryStencils.cpp
)

set(VVM_CORE_HEADERS
    State.hpp
    Grid.hpp
    GridSpecification.hpp
    BoundaryConditionManager.hpp
    Parameters.hpp
    Field.hpp

    haloexchange/HaloExchanger.hpp
    haloexchange/HaloExchangeDegenerate.hpp
    haloexchange/HaloExchangerMPI.hpp
    haloexchange/HaloExchangerNCCL.hpp

    geometry/GeometryKind.hpp
    geometry/HorizontalLocation.hpp
    geometry/HorizontalStaggering.hpp
    geometry/GeometryField2D.hpp
    geometry/HorizontalGridSpec.hpp
    geometry/HorizontalGeometry.hpp
    geometry/HorizontalBoundaryDistances.hpp
    StartupCollectives.hpp
    geometry/AffineGeometry.hpp
    geometry/StaggeredVectorTransform.hpp
    geometry/CartesianGeometry.hpp
    geometry/RegularLatLonGeometry.hpp
    geometry/HorizontalGeometryFactory.hpp

    boundary/HorizontalBoundaryStencils.hpp
)

add_library(vvm_core STATIC
    ${VVM_CORE_SOURCES}
    ${VVM_CORE_HEADERS}
)

target_include_directories(vvm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# NCCL is only a dependency of the NCCL communication backend.
if(ENABLE_NCCL)
  target_include_directories(vvm_core PUBLIC ${NCCL_INCLUDE_DIR})
  target_link_libraries(vvm_core PUBLIC ${NCCL_LIBRARY})
endif()


target_link_libraries(vvm_core
    PUBLIC
        vvm_utils
        MPI::MPI_CXX
        ${NVCPUMATH_LIBRARY}
        ${PNETCDF_LIBRARY}
    PRIVATE
        # Initializer constructs readers implemented in vvm_io. Both libraries
        # are static; CMake repeats their link group to resolve the cycle.
        vvm_io
        Kokkos::kokkos
        ${OpenMP_LIBRARIES}
        ${NVTOOLSEXT_LIBRARY}
        PnetCDF::pnetcdf
)

add_library(vvm_initializer STATIC
    Initializer.cpp
    InitializeBarotropic.cpp
    Initializer.hpp
)

target_link_libraries(vvm_initializer
    PUBLIC
        vvm_core
        vvm_io
    PRIVATE
        Kokkos::kokkos
)
