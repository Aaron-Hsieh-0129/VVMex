# When radiation and the surface process recompute their coefficients.
add_vvm_unit_test(test_process_scheduling)

# What the stale-SST cleanup in main() is allowed to delete, and that it treats
# every character in output_dir / output_filename_prefix as a filename character
# rather than shell syntax.
add_vvm_unit_test(test_sst_path)

# How a restart run recovers its clock: the metadata stored inside the file
# decides, a stored time and step must agree, and a file with neither is only
# usable through an explicit opt-in. Pure decision logic.
add_vvm_unit_test(test_restart_time)

add_test(NAME test_check_output_digest
         COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/test_check_output.py)
set_tests_properties(test_check_output_digest PROPERTIES LABELS unit TIMEOUT 60)

add_vvm_unit_test(test_deterministic_perturbation DEVICE
    LIBRARIES Kokkos::kokkos)

# The GrADS descriptor both output engines emit: the header lines that differ
# per engine, and the name mangling GrADS forces (lowercase, 15 characters).
# Links vvm_io to exercise the shared descriptor writer.
add_vvm_unit_test(test_grads_ctl
    LIBRARIES vvm_io Kokkos::kokkos MPI::MPI_CXX)

# Output precision: the engine-neutral output.precision key, its aliases, and
# how the BP5 block overrides it. Launched through mpirun because
# ConfigurationManager reads MPI_COMM_WORLD.
vvm_add_test_executable(test_output_precision
    LIBRARIES vvm_io vvm_utils Kokkos::kokkos MPI::MPI_CXX)
add_test(NAME test_output_precision
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 $<TARGET_FILE:test_output_precision>)
set_tests_properties(test_output_precision PROPERTIES LABELS unit TIMEOUT 60)

vvm_add_test_executable(test_numerical_configuration
    LIBRARIES vvm_utils MPI::MPI_CXX)
add_test(NAME test_numerical_configuration
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 $<TARGET_FILE:test_numerical_configuration>)
set_tests_properties(test_numerical_configuration PROPERTIES LABELS unit TIMEOUT 60)

# Compact generalized-coordinate field accessor test.
vvm_add_test_executable(test_geometry_field_2d
    LIBRARIES Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_geometry_field_2d
    COMMAND test_geometry_field_2d
)

set_tests_properties(
    test_geometry_field_2d
    PROPERTIES
        LABELS unit
        TIMEOUT 60
)

_vvm_set_test_resources(
    test_geometry_field_2d
    1
)

# Cartesian test
vvm_add_test_executable(test_cartesian_geometry
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_cartesian_geometry
    COMMAND test_cartesian_geometry
)

set_tests_properties(
    test_cartesian_geometry
    PROPERTIES
        LABELS unit
        TIMEOUT 60
)

_vvm_set_test_resources(
    test_cartesian_geometry
    1
)

# Conservative generalized-coordinate flux divergence in the Cartesian limit.
add_vvm_unit_test(test_horizontal_flux_divergence DEVICE
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

# Metric-aware scalar gradient from T points to U/V faces.
add_vvm_unit_test(test_horizontal_scalar_gradient DEVICE
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

# Laplace-Beltrami composed from the scalar gradient and conservative divergence.
add_vvm_unit_test(test_horizontal_laplace_beltrami DEVICE
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

# Metric-aware vertical curl from covariant U/V components to Z.
add_vvm_unit_test(test_horizontal_curl DEVICE
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

# Stagger-aware lowering of contravariant U/V components to covariant U/V components.
add_vvm_unit_test(test_horizontal_vector_lowering DEVICE
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

# Vertical vorticity composed from stagger-aware vector lowering and covariant curl.
add_vvm_unit_test(test_vertical_vorticity DEVICE
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

# Regular latitude-longitude coordinates, metrics, and vector transformations.
# Runtime factory selection remains disabled until the configuration contract
# is added in the following step.
vvm_add_test_executable(test_regular_latlon_geometry
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_regular_latlon_geometry
    COMMAND test_regular_latlon_geometry
)

set_tests_properties(
    test_regular_latlon_geometry
    PROPERTIES
        LABELS unit
        TIMEOUT 60
)

_vvm_set_test_resources(
    test_regular_latlon_geometry
    1
)

# Canonical coordinate-component fields in State.
#
# The same binary is run once with Cartesian geometry and once with regular
# latitude-longitude geometry.  This pins the migration contract:
#
#   var_con -> contravariant dynamical-core representation
#   var     -> physical representation
#
# VVM retains its historical vorticity orientation:
#
#   xi_con   =  omega^1
#   eta_con  = -omega^2
#   zeta_con =  omega^3
vvm_add_test_executable(test_state_coordinate_fields
    LIBRARIES vvm_core vvm_io vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_state_coordinate_fields_cartesian
    COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG}
        1
        ${VVM_MPI_BIND_ARGS}
        ${MPIEXEC_PREFLAGS}
        ${GPU_WRAP}
        $<TARGET_FILE:test_state_coordinate_fields>
        ${TEST_DIR}/configs/grid_structured_cartesian.json
        ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(
    test_state_coordinate_fields_cartesian
    PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        LABELS "unit"
        TIMEOUT 120
)

_vvm_set_test_resources(
    test_state_coordinate_fields_cartesian
    1
)

add_test(
    NAME test_state_coordinate_fields_rll
    COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG}
        1
        ${VVM_MPI_BIND_ARGS}
        ${MPIEXEC_PREFLAGS}
        ${GPU_WRAP}
        $<TARGET_FILE:test_state_coordinate_fields>
        ${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json
        ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(
    test_state_coordinate_fields_rll
    PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        LABELS "unit"
        TIMEOUT 120
)

_vvm_set_test_resources(
    test_state_coordinate_fields_rll
    1
)

# Normalization of legacy and structured horizontal/vertical grid configuration.
vvm_add_test_executable(test_grid_specification
    LIBRARIES vvm_core vvm_utils MPI::MPI_CXX)
add_test(NAME test_grid_specification
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 $<TARGET_FILE:test_grid_specification>)
set_tests_properties(test_grid_specification PROPERTIES LABELS unit TIMEOUT 60)

# The structured Cartesian configuration must produce the same Grid contract
# as the legacy flat configuration.
add_test(NAME test_grid_geometry_structured_r1
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 ${GPU_WRAP} $<TARGET_FILE:test_grid_geometry>
                 ${TEST_DIR}/configs/grid_structured_cartesian.json
         WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
set_tests_properties(test_grid_geometry_structured_r1 PROPERTIES LABELS unit TIMEOUT 300)
_vvm_set_test_resources(test_grid_geometry_structured_r1 1)

add_test(NAME test_grid_geometry_structured_bounded_r1
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 ${GPU_WRAP} $<TARGET_FILE:test_grid_geometry>
                 ${TEST_DIR}/configs/grid_structured_bounded_cartesian.json
         WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
set_tests_properties(test_grid_geometry_structured_bounded_r1
                     PROPERTIES LABELS unit TIMEOUT 300)
_vvm_set_test_resources(test_grid_geometry_structured_bounded_r1 1)

# RLL configuration is parsed, but it must not reach the legacy Cartesian
# topology or dynamical core until those paths have been migrated.
vvm_add_test_executable(test_grid_rll_guard
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)
add_test(NAME test_grid_rll_guard
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 ${GPU_WRAP} $<TARGET_FILE:test_grid_rll_guard>
                 ${TEST_DIR}/configs/grid_structured_rll.json
         WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
set_tests_properties(test_grid_rll_guard PROPERTIES LABELS unit TIMEOUT 300)
_vvm_set_test_resources(test_grid_rll_guard 1)

# Boundary filling
vvm_add_test_executable(test_horizontal_boundary_stencils
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_horizontal_boundary_stencils
    COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG}
    1
    ${VVM_MPI_BIND_ARGS}
    ${GPU_WRAP}
    $<TARGET_FILE:test_horizontal_boundary_stencils>
    ${TEST_DIR}/configs/grid_structured_bounded_cartesian.json
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

set_tests_properties(
    test_horizontal_boundary_stencils
    PROPERTIES LABELS unit TIMEOUT 300)

_vvm_set_test_resources(
    test_horizontal_boundary_stencils
    1)

# q2 poisson test
vvm_add_test_executable(test_bounded_q2_poisson_relaxation
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_bounded_q2_poisson_relaxation
    COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG}
    1
    ${VVM_MPI_BIND_ARGS}
    ${GPU_WRAP}
    $<TARGET_FILE:test_bounded_q2_poisson_relaxation>
    ${TEST_DIR}/configs/grid_structured_bounded_cartesian.json
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

set_tests_properties(
    test_bounded_q2_poisson_relaxation
    PROPERTIES LABELS unit TIMEOUT 300)

_vvm_set_test_resources(
    test_bounded_q2_poisson_relaxation
    1)

# CVVM-compatible shifted-Jacobi solve of the generalized horizontal
# Laplace-Beltrami equation at T and Z points.
vvm_add_test_executable(test_horizontal_elliptic_solver
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_horizontal_elliptic_solver_cartesian
    COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG}
    1
    ${VVM_MPI_BIND_ARGS}
    ${GPU_WRAP}
    $<TARGET_FILE:test_horizontal_elliptic_solver>
    ${TEST_DIR}/configs/grid_structured_cartesian.json
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
)

set_tests_properties(
    test_horizontal_elliptic_solver_cartesian
    PROPERTIES
    LABELS unit
    TIMEOUT 300
)

_vvm_set_test_resources(
    test_horizontal_elliptic_solver_cartesian
    1
)

add_test(
    NAME test_horizontal_elliptic_solver_rll
    COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG}
    1
    ${VVM_MPI_BIND_ARGS}
    ${GPU_WRAP}
    $<TARGET_FILE:test_horizontal_elliptic_solver>
    ${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
)

set_tests_properties(
    test_horizontal_elliptic_solver_rll
    PROPERTIES
    LABELS unit
    TIMEOUT 300
)

_vvm_set_test_resources(
    test_horizontal_elliptic_solver_rll
    1
)

# wind reconstruction test
vvm_add_test_executable(test_horizontal_wind_reconstruction
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

add_test(NAME test_horizontal_wind_reconstruction
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                 ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_reconstruction>
         WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

set_tests_properties(test_horizontal_wind_reconstruction PROPERTIES LABELS unit TIMEOUT 60)
_vvm_set_test_resources(test_horizontal_wind_reconstruction 1)

# wind consistency test
vvm_add_test_executable(test_horizontal_wind_consistency
    LIBRARIES vvm_core Kokkos::kokkos MPI::MPI_CXX)

add_test(NAME test_horizontal_wind_consistency
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_consistency>
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

set_tests_properties(test_horizontal_wind_consistency PROPERTIES LABELS unit TIMEOUT 60)
_vvm_set_test_resources(test_horizontal_wind_consistency 1)

# wind recovery
vvm_add_test_executable(test_horizontal_wind_recovery
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(NAME test_horizontal_wind_recovery_cartesian
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_recovery>
            ${TEST_DIR}/configs/grid_structured_cartesian.json
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

add_test(NAME test_horizontal_wind_recovery_rll
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_recovery>
            ${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

set_tests_properties(
    test_horizontal_wind_recovery_cartesian
    test_horizontal_wind_recovery_rll
    PROPERTIES LABELS unit TIMEOUT 120)

_vvm_set_test_resources(test_horizontal_wind_recovery_cartesian 1)
_vvm_set_test_resources(test_horizontal_wind_recovery_rll 1)

vvm_add_test_executable(test_parameters_resolved_grid
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

foreach(case_name IN ITEMS legacy structured mixed)
    add_test(NAME test_parameters_resolved_grid_${case_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
                ${GPU_WRAP} $<TARGET_FILE:test_parameters_resolved_grid>
                ${TEST_DIR}/configs/parameters_resolved_${case_name}.json
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

    set_tests_properties(test_parameters_resolved_grid_${case_name} PROPERTIES LABELS unit TIMEOUT 60)
    _vvm_set_test_resources(test_parameters_resolved_grid_${case_name} 1)
endforeach()

add_test(NAME test_parameters_resolved_grid_rll_guard
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
            ${GPU_WRAP} $<TARGET_FILE:test_parameters_resolved_grid>
            ${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

set_tests_properties(test_parameters_resolved_grid_rll_guard PROPERTIES LABELS unit TIMEOUT 60)
_vvm_set_test_resources(test_parameters_resolved_grid_rll_guard 1)

vvm_add_test_executable(test_vertical_grid_configuration
    LIBRARIES vvm_core vvm_io vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(NAME test_vertical_grid_configuration
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
            ${GPU_WRAP} $<TARGET_FILE:test_vertical_grid_configuration>
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

set_tests_properties(test_vertical_grid_configuration PROPERTIES LABELS unit TIMEOUT 300)
_vvm_set_test_resources(test_vertical_grid_configuration 1)

vvm_add_test_executable(test_model_configuration_validation
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(NAME test_model_configuration_validation
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${VVM_MPI_BIND_ARGS}
            $<TARGET_FILE:test_model_configuration_validation>
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")

set_tests_properties(test_model_configuration_validation PROPERTIES LABELS unit TIMEOUT 60)

vvm_add_test_executable(test_surface_vertical_configuration
    LIBRARIES vvm_surface vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_surface_vertical_configuration
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_surface_vertical_configuration>
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(test_surface_vertical_configuration PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    LABELS "unit"
    RESOURCE_LOCK gpu
    TIMEOUT 120
)

vvm_add_test_executable(test_horizontal_configuration_consumers
    LIBRARIES vvm_core vvm_io vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_horizontal_configuration_consumers
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_configuration_consumers>
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(test_horizontal_configuration_consumers PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    LABELS "unit"
    RESOURCE_LOCK gpu
    TIMEOUT 120
)

# Restart metadata in the files themselves, including the regression this all
# exists for: write a restart file, rename it, load it, and the recovered time
# and step do not move.
add_vvm_file_unit_test(test_restart_metadata_io 2)
# Grid owns a valid Cartesian geometry after decomposition. The multirank run
# also verifies that each rank receives the correct global coordinate origin.
add_vvm_device_unit_test(test_grid_geometry ${TEST_DIR}/configs/2dbubble.json 2)

# State::calculate_horizontal_mean(): the API shared by the NCCL and the
# standard-MPI backend. The 2-rank run is what pins the global reduction.
add_vvm_device_unit_test(test_horizontal_mean ${TEST_DIR}/configs/2dbubble.json 2)

# Geometry-aware horizontal mean. Cartesian J is uniform, so this pins
# compatibility with the existing point mean before RegularLatLonGeometry adds
# nonuniform area weights.
add_vvm_device_unit_test(test_horizontal_area_weighted_mean ${TEST_DIR}/configs/2dbubble.json 2)

# HaloExchanger: the NCCL and the standard-MPI implementation are separate code,
# and only a value-level check keeps them exchanging the same cells.
add_vvm_device_unit_test(test_halo_exchange ${TEST_DIR}/configs/2dbubble.json 2)
add_reduced_halo_tests(nx1 1 32 2)
add_reduced_halo_tests(ny1 32 1 2)

vvm_add_test_executable(test_p3_multistage_history
    LIBRARIES vvm_dynamics vvm_core vvm_io vvm_utils Kokkos::kokkos MPI::MPI_CXX)
add_test(NAME test_p3_multistage_history_r1
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${VVM_MPI_BIND_ARGS} ${GPU_WRAP}
                 $<TARGET_FILE:test_p3_multistage_history>
                 ${TEST_DIR}/configs/p3_multistage_history.json
         WORKING_DIRECTORY "${VVM_TEST_WORKDIR}")
set_tests_properties(test_p3_multistage_history_r1
                     PROPERTIES LABELS unit TIMEOUT 300)
_vvm_set_test_resources(test_p3_multistage_history_r1 1)

vvm_add_test_executable(test_horizontal_vorticity
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_horizontal_vorticity
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_vorticity>
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(test_horizontal_vorticity PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    LABELS "unit"
    TIMEOUT 60
)

_vvm_set_test_resources(test_horizontal_vorticity 1)

# wind vertical integration
vvm_add_test_executable(test_wind_vertical_integration
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_wind_vertical_integration
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_wind_vertical_integration>
            ${MPIEXEC_POSTFLAGS} "${TEST_DIR}/configs/2dbubble.json"
)

set_tests_properties(test_wind_vertical_integration PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    LABELS "unit"
    TIMEOUT 120
)

_vvm_set_test_resources(test_wind_vertical_integration 1)

# wind column recovery 
vvm_add_test_executable(test_horizontal_wind_column_recovery
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_horizontal_wind_column_recovery
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_column_recovery>
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(test_horizontal_wind_column_recovery PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    LABELS "unit"
    TIMEOUT 60
)

_vvm_set_test_resources(test_horizontal_wind_column_recovery 1)

# Wind-column execution and CUDA graph replay
vvm_add_test_executable(test_horizontal_wind_column_replay
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_horizontal_wind_column_replay
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_column_replay>
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(test_horizontal_wind_column_replay PROPERTIES
    WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
    LABELS "unit"
    TIMEOUT 120
)

_vvm_set_test_resources(test_horizontal_wind_column_replay 1)

# Fixed-iteration horizontal solver followed by full-column wind recovery
vvm_add_test_executable(test_horizontal_wind_solver_column
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_horizontal_wind_solver_column_cartesian
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_solver_column>
            ${TEST_DIR}/configs/grid_structured_cartesian.json
            ${MPIEXEC_POSTFLAGS}
)

add_test(
    NAME test_horizontal_wind_solver_column_rll
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_solver_column>
            ${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(
    test_horizontal_wind_solver_column_cartesian
    test_horizontal_wind_solver_column_rll
    PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        LABELS "unit"
        TIMEOUT 120
)

_vvm_set_test_resources(test_horizontal_wind_solver_column_cartesian 1)
_vvm_set_test_resources(test_horizontal_wind_solver_column_rll 1)

# Capture/replay of extrapolation, paired solve and wind-column recovery
vvm_add_test_executable(test_horizontal_wind_solver_column_replay
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_horizontal_wind_solver_column_replay_cartesian
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_solver_column_replay>
            ${TEST_DIR}/configs/grid_structured_cartesian.json
            ${MPIEXEC_POSTFLAGS}
)

add_test(
    NAME test_horizontal_wind_solver_column_replay_rll
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
            ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_solver_column_replay>
            ${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json
            ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(
    test_horizontal_wind_solver_column_replay_cartesian
    test_horizontal_wind_solver_column_replay_rll
    PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        LABELS "unit"
        TIMEOUT 120
)

_vvm_set_test_resources(test_horizontal_wind_solver_column_replay_cartesian 1)
_vvm_set_test_resources(test_horizontal_wind_solver_column_replay_rll 1)

if(VVM_TEST_MULTIRANK)
    foreach(ranks IN ITEMS 2 4)
        foreach(geometry IN ITEMS cartesian rll)
            if(geometry STREQUAL "cartesian")
                set(replay_config "${TEST_DIR}/configs/grid_structured_cartesian.json")
            else()
                set(replay_config "${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json")
            endif()

            set(replay_test "test_horizontal_wind_solver_column_replay_${geometry}_r${ranks}")

            add_test(
                NAME ${replay_test}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                        ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
                        ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_solver_column_replay>
                        "${replay_config}" ${MPIEXEC_POSTFLAGS}
            )

            set_tests_properties(${replay_test} PROPERTIES
                WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
                LABELS "unit;multirank"
                TIMEOUT 300
            )

            _vvm_set_test_resources(${replay_test} ${ranks})
        endforeach()
    endforeach()
endif()

# Decomposition consistency of the fixed solver and wind-column recovery
vvm_add_test_executable(test_horizontal_wind_solver_column_mpi
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

if(VVM_TEST_MULTIRANK)
    foreach(_column_ranks IN ITEMS 2 4)
        foreach(_column_geometry IN ITEMS cartesian rll)
            if(_column_geometry STREQUAL "cartesian")
                set(_column_config "${TEST_DIR}/configs/grid_structured_cartesian.json")
            else()
                set(_column_config "${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json")
            endif()

            set(_column_test "test_horizontal_wind_solver_column_mpi_${_column_geometry}_r${_column_ranks}")

            add_test(
                NAME ${_column_test}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_column_ranks}
                        ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
                        ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_solver_column_mpi>
                        ${_column_config}
                        ${MPIEXEC_POSTFLAGS}
            )

            set_tests_properties(${_column_test} PROPERTIES
                WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
                LABELS "unit;multirank"
                TIMEOUT 300
            )

            _vvm_set_test_resources(${_column_test} ${_column_ranks})
        endforeach()
    endforeach()
endif()

# wind adapter 
vvm_add_test_executable(test_horizontal_wind_state_adapter
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

set(wind_adapter_ranks 1)
if(VVM_TEST_MULTIRANK)
    list(APPEND wind_adapter_ranks 2 4)
endif()

foreach(ranks IN LISTS wind_adapter_ranks)
    foreach(geometry IN ITEMS cartesian rll)
        if(geometry STREQUAL "cartesian")
            set(adapter_config "${TEST_DIR}/configs/grid_structured_cartesian.json")
        else()
            set(adapter_config "${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json")
        endif()

        set(adapter_test "test_horizontal_wind_state_adapter_${geometry}_r${ranks}")

        add_test(
            NAME ${adapter_test}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
                    ${GPU_WRAP} $<TARGET_FILE:test_horizontal_wind_state_adapter>
                    "${adapter_config}" ${MPIEXEC_POSTFLAGS}
        )

        set_tests_properties(${adapter_test} PROPERTIES
            WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
            LABELS "unit"
            TIMEOUT 300
        )

        _vvm_set_test_resources(${adapter_test} ${ranks})
    endforeach()
endforeach()

vvm_add_test_executable(test_wind_solver_horizontal_diagnostic
    LIBRARIES vvm_dynamics vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

set(horizontal_diagnostic_ranks 1)
if(VVM_TEST_MULTIRANK)
    list(APPEND horizontal_diagnostic_ranks 2 4)
endif()

foreach(ranks IN LISTS horizontal_diagnostic_ranks)
    foreach(geometry IN ITEMS cartesian rll)
        if(geometry STREQUAL "cartesian")
            set(diagnostic_config "${TEST_DIR}/configs/grid_structured_cartesian.json")
        else()
            set(diagnostic_config "${TEST_DIR}/configs/horizontal_elliptic_regular_latlon.json")
        endif()

        set(diagnostic_test "test_wind_solver_horizontal_diagnostic_${geometry}_r${ranks}")

        add_test(
            NAME ${diagnostic_test}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${VVM_MPI_BIND_ARGS} ${MPIEXEC_PREFLAGS}
                    ${GPU_WRAP} $<TARGET_FILE:test_wind_solver_horizontal_diagnostic>
                    "${diagnostic_config}" ${MPIEXEC_POSTFLAGS}
        )

        set_tests_properties(${diagnostic_test} PROPERTIES
            WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
            LABELS "unit"
            TIMEOUT 300
        )

        _vvm_set_test_resources(${diagnostic_test} ${ranks})
    endforeach()
endforeach()

# Field-level regular latitude-longitude Takacs scalar transport.
add_vvm_unit_test(test_regular_latlon_scalar_transport DEVICE
    LIBRARIES vvm_dynamics)

add_vvm_unit_test(test_regular_latlon_top_transport DEVICE
    LIBRARIES vvm_core Kokkos::kokkos)

# Dry baroclinic horizontal-vorticity source on RLL geometry.
add_vvm_unit_test(test_regular_latlon_dry_buoyancy DEVICE
    LIBRARIES vvm_dynamics)

# Shared RLL moist interfaces; full-model capability remains guarded.
vvm_add_test_executable(test_regular_latlon_moist_interfaces
    LIBRARIES vvm_dynamics vvm_core vvm_io)
add_test(NAME test_regular_latlon_moist_interfaces
         COMMAND ${MPIEXEC_EXECUTABLE} ${VVM_MPI_BIND_ARGS} -n 1 ${GPU_WRAP}
                 $<TARGET_FILE:test_regular_latlon_moist_interfaces>)
set_tests_properties(test_regular_latlon_moist_interfaces PROPERTIES LABELS integration TIMEOUT 120)
_vvm_set_test_resources(test_regular_latlon_moist_interfaces 1)

# Regular latitude-longitude Shutts-Gray turbulence:
# metric-aware deformation, Km/Kh, scalar diffusion,
# horizontal-vorticity diffusion, and top-zeta diffusion.
vvm_add_test_executable(test_regular_latlon_turbulence
    LIBRARIES vvm_turbulence vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_regular_latlon_turbulence
    COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG}
        1
        ${VVM_MPI_BIND_ARGS}
        ${MPIEXEC_PREFLAGS}
        ${GPU_WRAP}
        $<TARGET_FILE:test_regular_latlon_turbulence>
        ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(
    test_regular_latlon_turbulence
    PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        LABELS "unit"
        TIMEOUT 120
)

_vvm_set_test_resources(
    test_regular_latlon_turbulence
    1
)

# Regular latitude-longitude surface physics:
# physical-wind exchange, latitude independence,
# momentum-flux orientation, and surface tendency coupling.
vvm_add_test_executable(test_regular_latlon_surface
    LIBRARIES vvm_surface vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)

add_test(
    NAME test_regular_latlon_surface
    COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG}
        1
        ${VVM_MPI_BIND_ARGS}
        ${MPIEXEC_PREFLAGS}
        ${GPU_WRAP}
        $<TARGET_FILE:test_regular_latlon_surface>
        ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(
    test_regular_latlon_surface
    PROPERTIES
        WORKING_DIRECTORY "${VVM_TEST_WORKDIR}"
        LABELS "unit"
        TIMEOUT 120
)

_vvm_set_test_resources(
    test_regular_latlon_surface
    1
)

include(${CMAKE_CURRENT_LIST_DIR}/vertical_wind_diagnostic.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/vertical_elliptic_solver.cmake)
