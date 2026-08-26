# Physics implementation

Use this page to see which component owns each parameterization and where it
runs in `Model::run_step`. The dynamical core always participates; optional
physics and forcing components are constructed from the case JSON.

## Components and toggles

| Component | Config flag | Implementation (typical entry) |
| ----------- | ------------ | ------------------------------ |
| P3 microphysics | `physics.p3.enable_p3` | `Physics::VVM_P3_Interface` |
| RRTMGP radiation | `physics.rrtmgp.enable_rrtmgp` | `Physics::RRTMGP::RRTMGPRadiation` |
| Turbulence | `physics.turbulence.enable_turbulence` | `Physics::TurbulenceProcess` |
| Surface fluxes | `physics.surface.enable_surface` | `Physics::SurfaceProcess` |
| Noah land | `physics.land.enable_land` | `Physics::LandProcess` |
| Sponge layer | `dynamics.forcings.sponge_layer.enable` | `Dynamics::SpongeLayer` |
| Lateral nudging | `dynamics.forcings.lateral_boundary_nudging.enable` | `Dynamics::LateralBoundaryNudging` |
| Random perturbation | `dynamics.forcings.random_perturbation.enable` | `Dynamics::RandomForcing` |

## Time-step order (`Model::run_step`)

The following order is implemented in `Model::run_step` (abbreviated):

1. **Lateral boundary nudging** — update large-scale forcing if enabled.
2. **Thermodynamic tendencies** — `dycore_->calculate_thermo_tendencies()`.
3. **Radiation** — on steps divisible by `rad_frequency_step`, `radiation_->run`; then `calculate_tendencies` for `th`.
4. **Update thermodynamics** — `dycore_->update_thermodynamics(dt)`.
5. **Random forcing** — if enabled.
6. **P3** — `microphysics_->run` using updated thermodynamic state.
7. **Turbulence** — compute coefficients and tendency fields for configured thermodynamic variables.
8. **Surface** — periodic `compute_coefficients` and tendencies into `fe_tendency_*` fields.
9. **Land** — periodic `run` and `calculate_tendencies` for surface-coupled variables.
10. **Forward updates** — apply turbulence/surface/land tendency updates via `TimeIntegrator::apply_forward_update` where applicable.
11. **Sponge / nudging** — additional forward updates on selected variables.
12. **Halo exchange** for thermodynamic variables and boundary application.
13. **Vorticity** — `calculate_vorticity_tendencies`, `update_vorticity`, then turbulence/sponge updates on vorticity-related fields, halos, vertical structure for `zeta`, optional **wind diagnosis** (`diagnose_wind_fields`) unless in idealized modes that disable the wind solver.

Radiation, surface, and land frequencies are evaluated against the model step
index.

P3 needs both the state before advection (`var_prev`) and the state after
advection (`var_now`). It therefore runs after
`dycore_->update_thermodynamics(dt)`: that update moves the old current state
into `var_prev` and exposes the newly advected state as `var_now`.

## P3

The P3 implementation lives under `src/physics/p3/` with EAMxx-style pack-oriented kernels. VVMex modifies this EAMxx framework to follow Fortran P3 process coupling for vapor adjustment, droplet activation, rain size distribution, mass conservation, and lookup-table behavior; see [P3 modifications](p3-modifications.md) for the full comparison table. Lookup tables can be generated or read from `rundata/p3/` depending on `make_lookup_table` and paths in configuration.

## RRTMGP

RRTMGP interfaces are under `src/physics/rrtmgp/`, including `VVM_rrtmgp_process_interface` and Kokkos-oriented gas optics / RTE layers under `external/cpp/`. `column_chunk_size` and `rad_frequency_step` trade accuracy, performance, and coupling frequency.

## Land and surface

- **Surface** modes (e.g. `sflux_tc_2d`) provide lower-boundary flux tendencies into the same `fe_tendency_*` machinery as turbulence.
- **Land** (Noah) runs on a sub-step schedule and feeds tendencies for surface-coupled quantities; Fortran OpenACC code is integrated via the build system for GPU execution where configured.

## Idealized dynamics tests

When `simulation.idealized_test` is one of `advection_u`, `advection_v`, `advection_w`, `stretching`, or `twisting`, the constructor sets `wind_solver_` false so the vertical wind solver is skipped for those benchmarks. `2dbubble` keeps the solver unless the same list logic changes in the future—refer to `Model` constructor for the authoritative list in `no_solver_mode`.

## Further reading

- Regression tests: `tests/CMakeLists.txt` and `tests/configs/*.json`
