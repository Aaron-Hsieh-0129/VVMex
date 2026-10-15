# VVMex

[![Checks](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/checks.yml/badge.svg)](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/checks.yml)
[![CPU tests](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/cpu-tests.yml)
[![GPU tests](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/gpu-tests.yml/badge.svg)](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/gpu-tests.yml)
[![Nightly](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/nightly.yml/badge.svg)](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/nightly.yml)
[![C++](https://img.shields.io/badge/C++-17%2B-blue.svg)](https://isocpp.org/)
[![Kokkos](https://img.shields.io/badge/Kokkos-Performance_Portability-blueviolet.svg)](https://kokkos.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21318987.svg)](https://doi.org/10.5281/zenodo.21318987)

VVMex is a GPU-capable, object-oriented C++ implementation of the Vector Vorticity cloud-resolving Model (VVM) for large-eddy simulations on heterogeneous high-performance computing systems.


## Table of Contents

- [Features](#features)
- [Documentation](#documentation)
- [Repository contents](#repository-contents)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Output and restarts](#output-and-restarts)
- [Testing and CI](#testing-and-ci)
- [Reproducing the paper experiments](#reproducing-the-paper-experiments)
- [Acknowledgments & References](#acknowledgments--references)
- [Citation](#citation)
- [License](#license)
- [Contact & Support](#contact--support)


## Features

- **3D Cloud-Resolving Dynamics**: High-performance atmospheric dynamics utilizing the Vector Vorticity formulation.
- **CPU and GPU backends**: Kokkos CUDA with NCCL halo exchange on NVIDIA GPUs, or Kokkos OpenMP with MPI halo exchange on CPUs.
- **Advanced Physics Schemes**:
  - **Microphysics**: P3 microphysics scheme adapted from E3SM EAMxx, with restored vapor-cloud water (qv $\leftrightarrow$ qc) conversion processes based on the original Fortran P3 formulation.
  - **Super-droplet microphysics**: Lagrangian super-droplet microphysics from CLEO, selected with `physics.microphysics.engine = "cleo"` (Cartesian grids).
  - **Radiation**: RRTMGP radiation scheme adapted from E3SM EAMxx.
  - **Land Surface Model**: Noah land surface model with GPU acceleration (Fortran OpenACC), provided by the Central Weather Administration (CWA) of Taiwan.
- **TaiwanVVM Support**: Capable of simulating high-resolution Taiwan topography using generated terrain datasets (example scripts provided).
- **Output and restarts**: HDF5, asynchronous SST-to-HDF5 output, and direct multi-step BP5 datasets, with configurable output precision and restart support.
- **Reproducibility**: Backend-specific regression baselines and reference digests check model results. Optional `VVM_DETERMINISTIC_FP=ON` aligns the CPU and GPU dry-core cases described in the [reproducibility guide](docs/developer-guides/reproducibility.md).


## Documentation

Full documentation is at **<https://aaron-hsieh-0129.github.io/VVMex/>**. The
pages most people need first:

| Page | Contents |
|---|---|
| [Environment Installation](https://aaron-hsieh-0129.github.io/VVMex/environment/) | Building the GPU and CPU-only dependency stacks from source. |
| [Quick Start](https://aaron-hsieh-0129.github.io/VVMex/quick-start/) | Dependencies, building, and running a first job. |
| [Model Configuration](https://aaron-hsieh-0129.github.io/VVMex/user-guides/configuration/) | Every configuration key the model reads, with types and defaults. |
| [Job Submission](https://aaron-hsieh-0129.github.io/VVMex/user-guides/job-submission/) | `submit.py`, SLURM, and rank/GPU layout. |
| [Output](https://aaron-hsieh-0129.github.io/VVMex/user-guides/output/) | Output engines, precision, restarts, and GrADS descriptors. |
| [CLEO Super-Droplet Microphysics](https://aaron-hsieh-0129.github.io/VVMex/user-guides/cleo/) | Building, input generation, coupling, and limitations of the CLEO option. |
| [Default Cases](https://aaron-hsieh-0129.github.io/VVMex/examples/default-cases/) | Runnable dynamics, microphysics, radiation, land, and Taiwan examples. |
| [Continuous Integration](https://aaron-hsieh-0129.github.io/VVMex/developer-guides/ci/) | Test tiers, workflows, self-hosted runners, and approval policy. |
| [Developer Guides](https://aaron-hsieh-0129.github.io/VVMex/developer-guides/) | Architecture, physics implementation, and reproducibility rules. |


## Repository contents

| Path | Description |
|---|---|
| `src/` | Main VVMex source code. |
| `tests/` | Verification and regression tests. |
| `rundata/` | Case configurations, sounding profiles, spatial inputs, and physics data. |
| `tools/` | Job-launching helpers, preprocessing, CLEO input generation, and CI runner setup. |
| `docs/` | User and developer documentation. |
| `externals/` | External dependency integration, including Git submodules and the fetched CLEO fork. |


## Requirements

VVMex is tested on NVIDIA GPU systems and with the NVHPC/Kokkos OpenMP
CPU-only backend. Other GPU backends are not yet part of the validated workflow.

| Component | Tested version / requirement | Notes |
|---|---:|---|
| Compiler toolchain | NVHPC and GCC 11+ | NVHPC is required on both backends for Noah Fortran and `libnvcpumath`; GCC supplies its C++ standard library. |
| GPU compiler / CUDA | NVHPC ≥ 25.3 with CUDA ≥ 12.8 | The always-built CLEO interface requires C++20. CUDA is only required for GPU builds. |
| MPI | Open MPI or NVHPC/HPC-X MPI | Keep `mpicc`, `mpic++`, and `mpifort` in the same toolchain. |
| CMake | ≥ 3.20 | Required for CMake presets. |
| Kokkos | 4.7+ | CUDA for GPU builds; OpenMP for CPU builds. GPU CLEO needs relocatable device code and static Kokkos libraries. |
| HDF5 | ≥ 1.14.5 | Required for NetCDF/HDF5 I/O. |
| NetCDF-C | ≥ 4.4.1.1 | Required for NetCDF input/output. |
| NetCDF-Fortran | ≥ 4.4.1 | Required by Fortran-based components. |
| PnetCDF | ≥ 1.14.1 | Used for parallel NetCDF support. |
| ADIOS2 | ≥ 2.11.0 | Used for scalable and asynchronous output. |
| Python | Python 3 | Used by `submit.py`, test tools, and input generators; CLEO input generation also needs NumPy and cleopy's dependencies. |

Use the [GPU](docs/environment/gpu.md) or [CPU-only](docs/environment/cpu.md)
installation guide for a validated dependency stack. Keep backend-specific
install prefixes and build directories separate. A pure GCC/gfortran build is
not currently supported. Both guides build ADIOS2 without Kokkos support so I/O
ranks remain host-only.


## Quick Start

Install the dependencies first, then select or create a machine entry in
`CMakePresets.json`. A preset names the backend, dependency prefixes, and build
directory; `submit.py` reads the same preset when launching the model.

```bash
git clone --recursive https://github.com/Aaron-Hsieh-0129/VVMex.git
cd VVMex
export VVM_ROOT=$PWD                      # required before configuring or running

# GPU preset: normally builds into build/
cmake --preset <gpu-preset> -DBUILD_TESTS=ON
cmake --build build -j 64
```

For a CPU-only preset, use its separate build tree:

```bash
cmake --preset <cpu-preset> -DBUILD_TESTS=ON
cmake --build build_cpu -j 64
```

Launch a small case using the preset you built:

```bash
./submit.py --local --preset <preset> \
    -c rundata/input_configs/default_cases/advection_u.json --compute 1
```

Check the case's input paths and `output.output_dir` before running. Spatial
NetCDF inputs can be regenerated with `tools/generate_init_nc.py`; see
[Default cases](docs/examples/default-cases.md). CLEO cases also need inputs
generated with `tools/cleo/gen_cleo_inputs.py` as described in the
[CLEO guide](docs/user-guides/cleo.md).

Run `./submit.py` without arguments for interactive setup. Use `--local` for
local execution, or omit it for SLURM submission. For local GPU runs,
`VVM_GPU_LIST` selects physical device IDs; for CPU runs, `--cpus` and
`--omp-threads` control cores and threads per rank. See the
[Quick Start](docs/quick-start.md) and [Job Submission](docs/user-guides/job-submission.md)
guides for environment preparation and resource sizing.


## Output and restarts

Set `output.engine` in the case JSON:

| Engine | Output | Dedicated I/O ranks |
|---|---|---|
| `HDF5` | One `.h5` file per output time; useful for small runs and reference output. | None |
| `SST` | Streams to host-only I/O ranks that write HDF5; GPU production path. | Required (`--io N`) |
| `BP5` | One multi-step `.bp` dataset; validated on CPU and host-staged on GPU. | None |

HDF5 and BP5 reject nonzero `--io`. All three engines support restart workflows;
SST produces HDF5 restart files. Keep `output.precision` at `native` when a
restart must preserve the model's working precision. See [Output](docs/user-guides/output.md)
for engine settings, restart configuration, and GrADS descriptors.


## Testing and CI

Configure with `-DBUILD_TESTS=ON`. Before running GPU regressions, fetch their
Git LFS baselines:

```bash
git lfs pull origin --include="tests/baselines/*.h5" --exclude=""
python3 tests/scripts/check_hdf5_baselines.py
```

Run CTest in the preset's build directory with the dependency and MPI
environment from the installation guide:

```bash
ctest --test-dir build --output-on-failure       # GPU build
ctest --test-dir build_cpu --output-on-failure   # CPU build
```

`-L '^unit$'` selects unit tests; some still use MPI or a GPU.

Optional BP5, physics, CLEO, multi-rank, and RLL suites are enabled at configure
time. GPU tests use `VVM_TEST_GPUS` to select devices, and multi-rank tests need
distinct physical GPUs. CPU tests use `VVM_TEST_CPU_THREADS` and CTest's `-j`
processor budget. See [tests/README.md](tests/README.md) for flags and examples.

| CI workflow | Coverage |
|---|---|
| Checks | Python, shell, configurations, test registration, and documentation checks on GitHub-hosted runners. |
| CPU tests | Clean CPU build, default tests, CLEO coupling, and multi-rank tests. |
| GPU tests | Clean GPU build, default tests, physics, CLEO coupling, and multi-rank tests. |
| Nightly | Adds BP5 to both backend suites; scheduled daily at 19:00 UTC or run manually. |

CPU and GPU jobs use self-hosted runners. Administrators run automatically;
other contributors require approval. Nightly waits up to 30 minutes for its
selected GPUs to become idle before building. Details are in the
[CI guide](docs/developer-guides/ci.md).


## Reproducing the paper experiments

The configuration files used for the experiments in the VVMex v1.0 model
description paper ([Hsieh et al., 2026](https://doi.org/10.5194/egusphere-2026-4205))
are provided in [`rundata/input_configs/default_cases/`](rundata/input_configs/default_cases/).

The exact code archived for the paper is
[VVMex v1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.21319556), corresponding
to tag [`v1.0.0`](https://github.com/Aaron-Hsieh-0129/VVMex/releases/tag/v1.0.0)
in this repository. Simulation output is archived separately at
[zenodo.org/records/21308460](https://zenodo.org/records/21308460).


## Acknowledgments & References

- **E3SM EAMxx**: The base implementations of the P3 microphysics and RRTMGP
  radiation schemes were adapted from the
  [E3SM](https://github.com/E3SM-Project/E3SM) project. If your work relies on
  those components, please also acknowledge E3SM following their
  [guidelines](https://e3sm.org/resources/policies/acknowledge-e3sm/):

  ```bibtex
  @misc{e3sm-model,
    title  = {{Energy Exascale Earth System Model (E3SM)}},
    author = {{E3SM Project}},
    doi    = {10.11578/E3SM/dc.20240301.3},
    url    = {https://dx.doi.org/10.11578/E3SM/dc.20240301.3},
    year   = 2024,
    month  = mar,
  }
  ```

- **P3 Microphysics**: The explicit condensation and evaporation processes between water vapor and cloud water, which are absent in the EAMxx version, have been re-implemented according to the original P3 formulation (e.g., *Morrison and Milbrandt, 2015*).
- **CWA Noah LSM**: The GPU-accelerated Noah land surface model is generously provided by the Central Weather Administration (CWA) of Taiwan.
- **CLEO**: The super-droplet microphysics is provided by
  [CLEO](https://github.com/yoctoyotta1024/CLEO), developed by Clara J. A. Bayley
  at the Max Planck Institute for Meteorology, which implements the
  Super-Droplet Method (*Shima et al., 2009*). VVMex builds a fork of it
  (`externals/cleo/CMakeLists.txt`). If your work uses it, please also cite:

  ```bibtex
  @software{bayley_cleo,
    author = {Bayley, Clara},
    title  = {{Cleo, an implementation of a Super-Droplet Model}},
    year   = {2025},
    url    = {https://github.com/yoctoyotta1024/CLEO},
  }

  @article{shima2009,
    author  = {Shima, S. and Kusano, K. and Kawano, A. and Sugiyama, T. and Kawahara, S.},
    title   = {{The super-droplet method for the numerical simulation of clouds and
               precipitation: a particle-based and probabilistic microphysics model
               coupled with a non-hydrostatic model}},
    journal = {Quarterly Journal of the Royal Meteorological Society},
    volume  = {135},
    number  = {642},
    pages   = {1307--1320},
    year    = {2009},
    doi     = {10.1002/qj.441},
  }
  ```


## Citation

If you use VVMex in published work, please cite **both** the model description
paper and the archived software release.

**Model description paper** (preprint, under review for *Geoscientific Model Development*):

```bibtex
@article{egusphere-2026-4205,
  author  = {Hsieh, Chin-Wei and Tseng, Shao-Yu and Wu, Chien-Ming},
  title   = {{VVMex v1.0: a modular GPU-capable refactoring of the Vector Vorticity
             cloud-resolving Model (VVM) with LES-based hierarchical validation and
             performance analysis}},
  journal = {EGUsphere [preprint]},
  year    = {2026},
  pages   = {1--44},
  doi     = {10.5194/egusphere-2026-4205},
  url     = {https://doi.org/10.5194/egusphere-2026-4205}
}
```

**Software release** (the exact code archived for the paper):

```bibtex
@software{hsieh_vvmex_2026,
  author    = {Hsieh, Chin-Wei},
  title     = {{VVMex v1.0.0: GPU-capable refactoring of the Vector Vorticity
               cloud-resolving Model}},
  year      = {2026},
  month     = {7},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21319556},
  url       = {https://doi.org/10.5281/zenodo.21319556}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Contact & Support

GitHub Issues: For bugs, feature requests, or code contributions, please open an issue on the GitHub repository.

Email: Users can contact us for more questions regarding the model or its usage at B08209006@ntu.edu.tw.
