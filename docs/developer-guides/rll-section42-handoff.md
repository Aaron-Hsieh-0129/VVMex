# RLL Section 4.2 handoff

Updated 2026-09-08, Asia/Taipei. Reproduction is **not established**.

## Checkout and authorization

- Repository: `/home/mog/CVVMex`.
- Branch: `feat/generalized-coordinate`.
- Inspected HEAD: `16e0355a8db89c817c1d4594e351514ddc7aceb4`, adding the horizontal-vorticity transport component after `5228fb3dd7fc1e24b53d043a04258929d40aa4ff`.
- Pre-existing tracked change: `.gitignore` adds `refs`. Preserve this change and all existing results.
- User authorizes direct edits, builds, tests and logical local commits; no push or merge. Maximum 224 CPU cores/ranks, GPUs 0–7, experiment outputs inside the project and below 10 GB.

## Governing-document blocker

The architecture and production integration documents in `refs/` were read. Their historical propose-only workflow is superseded by the current user instructions. `refs/VVMEX_COLLABORATION_WORKFLOW.md` is missing; searching `/home/mog` and this repository's Git history found no copy. No applicable `AGENTS.md` was found in the repository or ancestor directories. Asked the user for the missing workflow location or confirmation to proceed without it, because the user explicitly requires reading it before implementation decisions. Implementation is pending that answer; inspection may continue.

## Initial source evidence

- `src/core/ModelConfigurationValidation.hpp`, `src/core/Parameters.cpp`, and `src/core/BoundaryConditionManager.cpp` still reject full-model RLL. Keep these guards until supported paths are implemented and validated.
- `src/dynamics/spatial_schemes/RegularLatLonTakacs.*` supports scalar transport and optional dry buoyancy. The numerical factory lives in `src/dynamics/numerical_methods/NumericalMethodFactory.cpp`.
- RLL horizontal transport, horizontal deformation, and top deformation components exist under `src/dynamics/operators/`; their presence does not establish production wiring.
- `src/dynamics/solvers/WindSolverRegularLatLonDiagnostic.cpp` contains a composed diagnostic with explicit preparation and channel boundary support. Production ownership, dispatch, circulation evolution, and boundary qualification still require a complete audit.
- `DynamicalCore.cpp` divides xi/eta by `rhobar_up` and zeta by `rhobar` during tendency preparation, then multiplies density back. Integration must account for this temporary representation and avoid double normalization.
- Initializer, temporal integration, output, all tendency callers, and complete operator/solver implementations have not yet been fully audited. No source implementation decision has been made.

## Build environment evidence

- Existing `build/CMakeCache.txt`: Release, FP64, tests enabled, NVHPC 24.9, CUDA 12.6, HPC-X 2.20 MPI wrappers, NCCL enabled, GCC toolchain `/home/mog/gcc11`.
- Kokkos: `/home/mog/libs_GPUVVM/lib/cmake/Kokkos`.
- Eight NVIDIA H200 GPUs, each reporting 143771 MiB.
- Presets: `blaze`, `blaze-float`, `blaze-cpu`. CPU preset points to `/raid/mog/libs_VVM_cpu`; its `lib/cmake/Kokkos/KokkosConfig.cmake` exists, but the complete CPU build environment has not been tested. `/home/mog/libs_CPU` is a separate dependency prefix.
- Only one configured build was found under this repository's `build/`; binaries have not been verified against current HEAD.
- Sandbox startup fails with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`; repository inspection succeeded using reviewed escalated commands.

## Scientific reference evidence and open questions

- Jung PDF exists under `refs/`; it has not yet been read. Paper-based experiment constants and acceptance tolerances are not established.
- `refs/RUN_BAR/RUN/a.setup` documents Y_TST4 RLL: 400×100, DT=600 s, ITTADD=1152, NITERW=10, NITERXY=200, NOMAP=T. It also documents C_TST1 RLL: 8000×2000, DT=20 s, ITTADD=30240, NITERW=10, NITERXY=50. These resolutions must not be confused with the two physical cases.
- The active setup selects A_TST5 cubed sphere with NOMAP=F. Do not execute its destructive setup scripts or treat its active selection as the RLL configuration.
- `CODE_BAR/ini_3d_module.F` includes an 80 m/s jet with active TEST1 latitude limits and commented alternative tests, plus Gaussian vertical-vorticity perturbation. Align these alternatives against paper Section 4.2 before selecting either physical case.
- No `.nc`, `.ctl`, `.dat`, or tar archive files were found under `refs`; DATA_BAR is absent there. The quantitative output archive location remains unresolved. `refs` currently occupies about 65 MB.

## Validation and next step

No builds, tests, scientific runs, comparison plots, or numerical acceptance decisions have been performed in this inspection. No existing output was changed. Next: resolve the required missing workflow, read the paper and locate reference outputs, finish the production interface audit, then implement one coherent shared-architecture RLL experiment path and validate progressively. Preserve fixed iteration counts, Cartesian arithmetic, graph capture/replay, field staggering/signs, buoyancy declaration and vertical range, and channel circulation.

Section 4.3 remains out of scope. Its subsequent milestone requires separate verification of the baroclinic initialization and thermodynamic balance, complete divergent three-dimensional dynamics and planetary terms, vertical-grid and solver qualification, and its own reference settings and quantitative validation. This is a preliminary scope note, not a completed Section 4.3 requirements audit.
