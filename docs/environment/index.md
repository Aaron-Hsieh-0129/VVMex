# Environment installation

VVMex needs a dependency stack built from source. There are two guides, one per
execution backend:

<div class="grid cards" markdown>

- **[GPU environment](gpu.md)**

    Kokkos on CUDA, NCCL for halo exchange, the Noah land model with OpenACC.
    This is the validated production path.

- **[CPU-only environment](cpu.md)**

    Kokkos on OpenMP, standard MPI in place of NCCL, no device at run time.
    Self-contained: it does not require the GPU stack to be built first.

</div>

## Which one do I need?

| | GPU | CPU-only |
|---|---|---|
| Kokkos backend | CUDA | OpenMP |
| Halo exchange | NCCL | MPI |
| Noah land model | OpenACC offload | directives compile as comments |
| NVIDIA driver at run time | required | not required |
| Validated output engines | HDF5, SST, BP5 | HDF5, BP5 |
| Reference test data | `tests/baselines/`, `tests/references/` | `tests/baselines_cpu/`, `tests/references_cpu/` |

Both are complete on their own — pick the one matching the machine you will run
on. If you need both on the same machine, read the warning below first.

## Shared requirements

Both stacks need the same things, and neither guide will work without them:

- **NVHPC is mandatory in both cases.** VVMex links `libnvcpumath`
  unconditionally, and the Noah land model is written for `nvfortran` flags
  (`-Mallocatable=03`, `-Mfreeform`, `-Mextend`, `-r8`). Configuration fails
  without it. In a CPU build the SDK is used only as a compiler and MPI stack;
  its CUDA backend stays unused. A pure GCC/gfortran build is **not** supported.
- **GCC 11** supplies the C++ standard library NVHPC compiles against. Pinning
  it avoids a class of ABI failures where the binary builds but will not start.
- **`VVM_ROOT`** must point at the repository before configuring or running.
- **A `CMakePresets.json` entry** describing your machine. `submit.py` reads
  the same preset that built the code, so the launcher can never disagree with
  the binary about which backend it is.

## Keep prefixes and build directories separate

Install each stack into its own prefix, and keep one build directory per
backend: every GPU preset must use `${sourceDir}/build`, while every CPU preset
must use `${sourceDir}/build_cpu`. `submit.py` reads the selected preset's
`binaryDir`, so the same choice also routes the launcher to the matching
binary. Both trees may remain configured and built at the same time; never
reconfigure one tree for the other backend.

This is not tidiness — CPU and CUDA
Kokkos ship `libkokkoscore.so` with the *same* SONAME, and so do the ADIOS2
libraries. If both prefixes end up on `LD_LIBRARY_PATH`, the loader picks
whichever comes first, and a CPU binary will happily load the CUDA Kokkos and
open a GPU context.

The symptoms are confusing rather than obvious: tests occupy a GPU on a machine
you thought was CPU-only, or a build fails with
`#error ... __CUDACC__ macro as expected` because a CUDA Kokkos header directory
reached the include path. Both guides have troubleshooting tables covering this.

## After the stack is built

Continue with the [Quick Start](../quick-start.md) for configuring, building,
and running the model, then [Output](../user-guides/output.md) for choosing an
output engine — the choice is backend-dependent, so it is worth reading before
your first production run.
