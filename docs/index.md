# VVMex

VVMex is a C++17 cloud-resolving model based on the Vector Vorticity Model
(VVM). It uses Kokkos for CPU and NVIDIA GPU parallelism, MPI for domain
decomposition, and ADIOS2 for model output.

## Start here

| I want to… | Go to |
| --- | --- |
| Build and run a small example | [Quick start](quick-start.md) |
| Build the dependency stack | [Environment installation](environment/index.md) |
| Choose a ready-made case | [Default cases](examples/default-cases.md) |
| Edit a case JSON | [Model configuration](user-guides/configuration.md) |
| Run locally or submit through SLURM | [Job submission](user-guides/job-submission.md) |
| Choose HDF5, SST, or BP5 output | [Output](user-guides/output.md) |
| Understand or modify the code | [Developer guides](developer-guides/index.md) |

For routine runs, use `submit.py`. It reads the selected CMake preset, prepares
the runtime environment, assigns compute and I/O ranks, and keeps CPU/GPU
resources consistent with the binary that was built.

## What VVMex includes

- Three-dimensional vector-vorticity dynamics with configurable tendencies,
  forcings, and idealized benchmarks.
- Modified P3 microphysics based on E3SM EAMxx and the original Fortran P3
  process coupling.
- Kokkos-enabled RRTMGP radiation.
- Turbulence, surface fluxes, and the Noah land-surface model.
- HDF5, SST, and BP5 output, including restart support.
- Taiwan-oriented workflows using NetCDF topography and land fields.

## Project links

- Source: [VVMex on GitHub](https://github.com/Aaron-Hsieh-0129/VVMex)
- License: [Apache License 2.0](https://opensource.org/licenses/Apache-2.0)
- Questions: **B08209006@ntu.edu.tw**

For bugs and feature requests, open an issue in the source repository.
