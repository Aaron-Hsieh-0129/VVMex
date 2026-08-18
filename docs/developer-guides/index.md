# Developer guides

These pages summarize how VVMex is structured in this repository and how major subsystems fit together.

| Guide | Contents |
| ------ | -------- |
| [System architecture](architecture.md) | Directories, libraries, main program flow, MPI and Kokkos |
| [Physics implementation](physics.md) | RRTMGP, turbulence, surface, land, and the driver time step |
| [P3 Microphysics Modifications](p3-modifications.md) | **Details on restoring Fortran P3 $q_v \leftrightarrow q_c$ conversions within the EAMxx C++ framework** |
| [BP5 output internals](bp5-output.md) | Direct BP5 writer: architecture, dataset schema, HDF5 compatibility contract, CPU staging, buffer lifetime |

**Build system:** The root `CMakeLists.txt` configures C++17, CUDA/Kokkos, MPI, EKAT, NetCDF, PnetCDF, HDF5, ADIOS2, and optional NCCL. The main executable target is `vvm`, built from `src/main.cpp` and linked against `vvm_driver`, `vvm_core`, `vvm_io`, `vvm_dynamics`, `scream_share`, and `eamxx_physics`.

**Tests:** With `-DBUILD_TESTS=ON`, CMake registers a default tier that runs `vvm` against JSON files under `tests/configs/` and compares output to stored baselines. Optional tiers are opt-in at configure time — `VVM_TEST_BP5` (CPU builds only), `VVM_TEST_PHYSICS` (GPU builds only), and `VVM_TEST_MULTIRANK` — and each carries a `ctest` label once enabled. See the tier block at the top of `tests/CMakeLists.txt`.

**Continuous integration:** [CI](ci.md) covers which tiers gate a pull request, which run nightly, and how to register the self-hosted runner the compile-and-test jobs need. VVMex cannot be built on a GitHub-hosted runner, so the hermetic checks and the build-and-test jobs are separate workflows.
