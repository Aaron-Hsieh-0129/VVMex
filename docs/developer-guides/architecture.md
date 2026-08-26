# System architecture

Use this page to find ownership boundaries and follow a run from process startup
to output shutdown. VVMex is a C++17 model built around Kokkos, MPI, and
optional NCCL on NVIDIA GPUs.

## Repository layout (application code)

| Path | Role |
| ---- | ---- |
| `src/main.cpp` | MPI init, Kokkos init, optional NCCL, config load, split communicators for I/O servers, `Grid` / `Parameters` / `State` / `HaloExchanger` / `Model` / `HistoryWriter`, time loop |
| `src/driver/` | `Model`: orchestrates dynamical core, physics, and tendencies; implements `init`, `run_step`, `finalize` |
| `src/core/` | `Grid`, `State`, `Field`, `Parameters`, `HaloExchanger`, initializer, boundary helpers |
| `src/dynamics/` | Vector-vorticity dynamical core, time integration, forcings (sponge, nudging, random), idealized tests |
| `src/physics/` | P3 (`p3/`), RRTMGP (`rrtmgp/`), turbulence, surface, land (Noah); CMake aggregates as `eamxx_physics` interface + static libs |
| `src/io/` | `OutputManager` (ADIOS2 HDF5/SST), `IOServer` (SST consumer to HDF5), `history/` (neutral `HistoryWriter` interface), `bp5/` (direct BP5 writer; see [BP5 output internals](bp5-output.md)) |
| `src/utils/` | `ConfigurationManager` (JSON via nlohmann), timing and timers |
| `src/share/` | Shared EAMxx-derived utilities, constants, physics helpers |
| `externals/ekat/` | EKAT submodule: logging, YAML, test utilities, and Kokkos integration used by EAMxx-derived components |
| `rundata/` | Default-case JSON configs, initial profiles, spatial initial fields, P3 lookup tables |

Fortran pieces (e.g. Noah OpenACC) are linked through the physics/land subtree as required by CMake.

## Execution model

1. `MPI_Init` starts every rank. Shared-memory communicator sizing determines
   the OpenMP threads available to each rank.
2. GPU builds select a device from the node-local rank, then initialize Kokkos.
3. `ConfigurationManager` loads the case JSON.
4. When `--io-tasks N` is nonzero, ranks split into simulation and SST I/O
   communicators. I/O ranks enter `run_io_server`; simulation ranks continue.
5. Simulation ranks construct NCCL when enabled, followed by `Grid`, `State`,
   `HaloExchanger`, and `Model`.
6. `Model::init` loads initial or restart state and initializes enabled physics.
7. The selected `HistoryWriter` writes the initial state when configured. The
   time loop calls `model.run_step(dt)` until `simulation.total_time_s`.
8. The writer closes before Kokkos and MPI finalize.

## Major libraries (CMake targets)

- `vvm_driver` — `Model`
- `vvm_core` — grid, state, halos, parameters
- `vvm_dynamics` — dynamical core and related forcings
- `vvm_io` — ADIOS2 output and I/O server
- `vvm_utils` — configuration and timing
- `scream_share` — shared EAMxx code
- `vvm_physics` — interface aggregating P3, RRTMGP, turbulence, surface, land

The executable links `MPI::MPI_CXX`, `Kokkos::kokkos`, and the targets above.

## Communication

- **Halo exchange:** `HaloExchanger` exchanges halos for listed fields; CUDA graph optimization is configurable via `optimization.cuda_graph_halo_exchange` in JSON.
- **NCCL:** Used when `ENABLE_NCCL` is defined; `HaloExchanger` and `State` constructors take `ncclComm_t` and a CUDA stream in that build.

## Configuration

JSON keys are resolved with dotted paths (e.g. `physics.p3.enable_p3`) in `ConfigurationManager::find_node`. No separate YAML runtime config is required for the main executable; EKAT may use YAML for its own tooling in subprojects.
