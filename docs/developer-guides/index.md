# Developer guides

These pages explain the implementation choices behind VVMex. Start with the
architecture and physics overviews, then open a subsystem guide when changing
that area.

| Guide | Read it when you need to… |
| --- | --- |
| [System architecture](architecture.md) | Follow startup, MPI/Kokkos ownership, major targets, and repository layout |
| [Physics implementation](physics.md) | Understand the model time-step order and optional physics components |
| [P3 modifications](p3-modifications.md) | Compare VVMex microphysics coupling with EAMxx and original Fortran P3 |
| [RRTMGP modifications](rrtmgp-modifications.md) | Understand the single-precision radiation fixes |
| [BP5 output internals](bp5-output.md) | Change the BP5 schema, writer, restart path, or staging behavior |
| [Reproducibility](reproducibility.md) | Diagnose repeatability or CPU/GPU floating-point differences |
| [Continuous integration](ci.md) | Change test tiers, workflows, runners, or branch protection |

## Build and test orientation

The root `CMakeLists.txt` builds the `vvm` executable from application libraries
under `src/`. With `-DBUILD_TESTS=ON`, CMake registers unit and integration
tests from `tests/CMakeLists.txt`. Optional BP5, physics, and multirank tiers are
selected at configure time.

For a practical build command, use the [Quick start](../quick-start.md). For
workflow and runner details, use the [CI guide](ci.md).
