# VVMex

[![Checks](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/checks.yml/badge.svg)](https://github.com/Aaron-Hsieh-0129/VVMex/actions/workflows/checks.yml)
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
- [Reproducing the paper experiments](#reproducing-the-paper-experiments)
- [Acknowledgments & References](#acknowledgments--references)
- [Citation](#citation)
- [License](#license)
- [Contact & Support](#contact--support)


## Features

- **3D Cloud-Resolving Dynamics**: High-performance atmospheric dynamics utilizing the Vector Vorticity formulation.
- **Advanced Physics Schemes**:
  - **Microphysics**: P3 microphysics scheme adapted from E3SM EAMxx, with restored vapor-cloud water (qv $\leftrightarrow$ qc) conversion processes based on the original Fortran P3 formulation.
  - **Radiation**: RRTMGP radiation scheme adapted from E3SM EAMxx.
  - **Land Surface Model**: Noah land surface model with GPU acceleration (Fortran OpenACC), provided by the Central Weather Administration (CWA) of Taiwan.
- **TaiwanVVM Support**: Capable of simulating high-resolution Taiwan topography using generated terrain datasets (example scripts provided).
- **Reproducibility**: Results are held bit-for-bit against the v1.0.0 reference, and an optional deterministic mode makes the dry dynamical core identical between the GPU and CPU backends.


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
| [Developer Guides](https://aaron-hsieh-0129.github.io/VVMex/developer-guides/) | Architecture, physics implementation, and reproducibility rules. |


## Repository contents

| Path | Description |
|---|---|
| `src/` | Main VVMex source code. |
| `tests/` | Verification and regression tests. |
| `rundata/` | Example configurations and input files. |
| `tools/` | Preprocessing, utility, and data-generation scripts. |
| `docs/` | User and developer documentation. |
| `externals/` | External dependencies included as Git submodules. |


## Requirements

VVMex is tested on NVIDIA GPU systems and with the NVHPC/Kokkos OpenMP
CPU-only backend. Other GPU backends are not yet part of the validated workflow.

| Component | Tested version / requirement | Notes |
|---|---:|---|
| C++ compiler | GCC ≥ 11 and NVHPC ≥ 24.9 | NVHPC is recommended on NVIDIA GPU systems. |
| CUDA Toolkit | ≥ 12.6 | Currently tested on NVIDIA GPUs. |
| MPI | System MPI or NVHPC/HPC-X MPI | Use the MPI implementation provided by the target HPC system when possible. |
| CMake | ≥ 3.20 | Required for CMake presets. |
| Kokkos | ≥ 4.7.01 | Used for performance-portable C++ kernels. |
| HDF5 | ≥ 1.14.5 | Required for NetCDF/HDF5 I/O. |
| NetCDF-C | ≥ 4.4.1.1 | Required for NetCDF input/output. |
| NetCDF-Fortran | ≥ 4.4.1 | Required by Fortran-based components. |
| PnetCDF | ≥ 1.14.1 | Used for parallel NetCDF support. |
| ADIOS2 | ≥ 2.11.0 | Used for scalable and asynchronous output. |

Lower versions may work but have not been systematically tested. Building the stack from source is covered in the
[Environment Installation](https://aaron-hsieh-0129.github.io/VVMex/environment/) guide.


## Quick Start

Enough to confirm the model builds and runs on your machine. `<preset>` is an entry in `CMakePresets.json`; the shipped presets target specific clusters, so add one for yours with the paths to your dependency stack.

```bash
git clone --recursive https://github.com/Aaron-Hsieh-0129/VVMex.git
cd VVMex
export VVM_ROOT=$PWD                      # required before configuring or running

cmake --preset <preset> -DBUILD_TESTS=ON
cmake --build build -j 64

./submit.py --local --preset <preset> \
    -c rundata/input_configs/default_cases/advection_u.json --compute 1
ctest --test-dir build -L unit            # optional: fast, no GPU needed
```

`submit.py` derives the runtime environment from the preset, so there is no separate environment script to source. For dependencies, cluster presets, designing an experiment, SLURM submission, and choosing an output engine, see the [Quick Start guide](https://aaron-hsieh-0129.github.io/VVMex/quick-start/).


## Reproducing the paper experiments

The configuration files used for the experiments in the VVMex v1.0 model
description paper ([Hsieh et al., 2026](https://doi.org/10.5194/egusphere-2026-4205))
are provided in [`rundata/input_configs/default_cases/`](...).

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
