# Environment installation

VVMex has separate dependency stacks for GPU and CPU-only execution. Choose one
guide and keep its install prefix and build directory isolated from the other.

## Choose a backend

| | [GPU](gpu.md) | [CPU-only](cpu.md) |
| --- | --- | --- |
| Use when | NVIDIA GPUs are available | No GPU should be used at run time |
| Kokkos backend | CUDA | OpenMP |
| Halo exchange | NCCL | MPI |
| Build directory | `build/` | `build_cpu/` |
| Reference data | `tests/baselines/` | `tests/baselines_cpu/` |
| Main output paths | HDF5, SST, BP5 | HDF5, BP5 |

The GPU stack is the production path for large simulations. The CPU-only stack
is useful for development, portability checks, and systems without accessible
GPUs. Each stack is complete; neither must be built before the other.

## Requirements shared by both guides

- **NVHPC is required.** VVMex links `libnvcpumath`, and the Noah land model
  uses `nvfortran` flags. A pure GCC/gfortran build is not currently supported.
- **GCC 11** provides the C++ standard library used by NVHPC.
- **`VVM_ROOT`** must point to the repository.
- **A machine-specific CMake preset** must describe the dependency prefixes and
  backend. `submit.py` reads the same preset when launching the model.

## Keep the stacks isolated

Use a different install prefix for each backend and never place both prefixes
on `LD_LIBRARY_PATH` at the same time. CPU and CUDA builds of Kokkos and ADIOS2
use the same library names. If both are visible, the loader may silently select
the wrong backend.

Typical symptoms include:

- a CPU-only test unexpectedly opening a GPU context;
- a CUDA build finding CPU Kokkos headers;
- startup or finalization failures caused by libraries from different prefixes.

The [GPU guide](gpu.md) and [CPU-only guide](cpu.md) provide copy-ready build
commands and troubleshooting checks for their respective stacks.

## After installation

Continue with the [Quick start](../quick-start.md) to configure and build VVMex.
Before a production run, read [Output](../user-guides/output.md) and select an
engine appropriate for the chosen backend.
