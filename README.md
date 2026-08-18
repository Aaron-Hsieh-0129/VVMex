# VVMex

[![Checks](https://github.com/Aaron-Hsieh-0129/VVM_GPU_CPP/actions/workflows/checks.yml/badge.svg)](https://github.com/Aaron-Hsieh-0129/VVM_GPU_CPP/actions/workflows/checks.yml)
[![C++](https://img.shields.io/badge/C++-17%2B-blue.svg)](https://isocpp.org/)
[![Kokkos](https://img.shields.io/badge/Kokkos-Performance_Portability-blueviolet.svg)](https://kokkos.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

VVMex is a GPU-capable, object-oriented C++ design of the Vector Vorticity cloud-resolving Model (VVM) for large-eddy simulations on heterogeneous high-performance computing systems. 

The model is built with Kokkos to support accelerator-resident time stepping, modular dynamical and physical components, and extensible development toward exascale-oriented atmospheric modeling.

The name VVMex preserves the connection to VVM while leaving “ex” intentionally open, reflecting extensibility, exascale-oriented development, and modern C++-based implementation.


## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [License](#license)
- [Contact & Support](#contact--support)


## Features

- **3D Cloud-Resolving Dynamics**: High-performance atmospheric dynamics utilizing the Vector Vorticity formulation.
- **Advanced Physics Schemes**:
  - **Microphysics**: P3 microphysics scheme adapted from E3SM EAMxx, with restored vapor-cloud water (qv $\leftrightarrow$ qc) conversion processes based on the original Fortran P3 formulation.
  - **Radiation**: RRTMGP radiation scheme adapted from E3SM EAMxx.
  - **Land Surface Model**: Noah land surface model with GPU acceleration (Fortran OpenACC), provided by the Central Weather Administration (CWA) of Taiwan.
- **TaiwanVVM Support**: Capable of simulating high-resolution Taiwan topography using generated terrain datasets (example scripts provided).

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

### Tested software environment

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

Lower versions may work but have not been systematically tested.

For detailed installation instructions, see the [installation tutorial](https://aaron-hsieh-0129.github.io/VVMex/).

## Quick Start

### Step 1: Clone the Repository
After installing the required libraries, clone the project from GitHub:
```bash
git clone --recursive https://github.com/Aaron-Hsieh-0129/VVMex.git
cd VVMex
```

### Step 2: Environment Setup (Required)
You must define the project root directory using the `VVM_ROOT` environment variable before compiling or running the model. Add this to your session or `~/.bashrc`:

```bash
export VVM_ROOT=/absolute/path/to/your/VVMex
```

### Step 3: Configure CMake Presets

Open `CMakePresets.json` and configure or add a preset matching your machine cluster. Update the environment paths (`NVHPC_DIR`, `CMAKE_CXX_COMPILER`, `HDF5_DIR`, etc.) to match your build prefix.

At runtime, the submission wrapper (`submit.py`) will automatically scan this file, extract the parameters, and set up your execution environment (including `LD_LIBRARY_PATH` and `hpcx-init.sh`) dynamically. No separate environment script or manual variable export is required.


### Step 4: Build the Project

Compile the project from the root directory. Replace `<your_preset_name>` with your configured preset and `<core_number>` with the number of CPU cores for parallel building:

```bash
cd $VVM_ROOT
cmake --preset <your_preset_name> -DBUILD_TESTS=ON
cmake --build build -j<core_number>
```

### Step 5: Configure the Experiment

- **Default cases**: Start from a sample JSON under `rundata/input_configs/default_cases/`. These are ready-to-run VVMex cases with matching profiles and spatial inputs.

- **Main Settings**: Copy one of the default-case JSON files and modify it to design your experiment. Each physical process has its own toggle switch.
    
- **Initial Conditions**:
    
    - Default-case profiles are under `rundata/initial_conditions/profiles/default_cases/`.
        
    - Default-case spatial NetCDF files are under `rundata/initial_conditions/spatial/default_cases/`.
        
    - The spatial NetCDF files can be generated with `tools/generate_init_nc.py`; the script writes the path configured in `netcdf_reader.source_file`.

    - Initialization keeps the original split design: `initial_conditions.source_file`
      supplies the vertical text profile, while `netcdf_reader.source_file` supplies
      spatial, surface, and tracer fields.

    - Nonstandard cases (`simulation.idealized_test: "none"`) require both inputs.
      Missing profile columns, configured NetCDF variables, and incompatible spatial
      dimensions fail with diagnostics that name the input file and field.

- **ERA5 nudging/large-scale forcing** (only needed for nudging case):

    Edit the user-settings section near the top of
    `tools/generate_ls_forcing.py`: select the case JSON, local ERA5 dynamics
    and humidity files, UTC start/end times, target geographic bounds, and the
    separate `WIND_MEAN_*` bounds used for the U/V profile average. Then run:

    ```bash
    python tools/generate_ls_forcing.py
    ```

    `VVM_CONFIG_PATH` may override the configured JSON path. The script reads
    `nx`, `ny`, output naming, and forcing cadence from that JSON. It also reads
    the external `initial_conditions.source_file` profile and reproduces
    VVMex's hydrostatic initialization to obtain the actual physical-layer
    pressure grid; target pressure levels are not hard-coded.

    The tool uses `numpy`, `xarray`, and `netCDF4`. Reading GRIB additionally
    requires the `cfgrib` backend and ecCodes; NetCDF inputs do not. ERA5
    downloading, inspection plots, and profile generation are intentionally
    outside this script. Timestamp generation uses `PROCESS_COUNT = 16` worker
    processes by default; set it to `1` to run serially.

    Like `generate_init_nc.py`, the forcing generator also retains an
    idealized mode. Set `USE_ERA5_FORCING = False` to skip ERA5 input and call
    `define_idealized_forcing(...)`. Its default fields are horizontally
    uniform profiles derived from `initial_conditions.source_file`; edit that
    helper to define idealized spatial structure or time dependence.


### Step 6: Execute

We provide a user-friendly wrapper script (submit.py) located in the root directory to handle both local execution and SLURM job submission. It automatically manages MPI tasks, GPU allocations, and directory creation.

#### Option A: Using the Submission Wrapper (Recommended)

**Interactive Mode:**

If you do not know which inputs to provide, simply run the script without any arguments and follow the guided prompts step by step:

```bash
$VVM_ROOT/submit.py
```

**Command-Line Mode (Quick Start)**

For automated workflows or quick executions, you can pass arguments directly.

- Local Execution (HDF5 Engine):

```bash
cd $VVM_ROOT
./submit.py -c ./rundata/input_configs/default_cases/advection_u.json --compute 4 --local
```

- Local execution on specific GPUs:

```bash
cd $VVM_ROOT
VVM_GPU_LIST=0,1,2,3,4,5,6,7 ./submit.py --local \
  -c "rundata/input_configs/default_cases/taiwanvvm_2048.json" \
  --preset blaze \
  --compute 8 \
  --nodes 1
```

- SLURM Submission (SST Engine with Asynchronous I/O):

```bash
cd $VVM_ROOT
./submit.py -c ./rundata/input_configs/default_cases/sea_grass_mountain.json --compute 16 --io 4 --nodes 4 --gpus 5 -t 24:00:00
```

- SLURM Submission (BP5 Engine, CPU build, no I/O ranks):

```bash
cd $VVM_ROOT
./submit.py --preset f1-cpu -c ./rundata/input_configs/default_cases/taiwanvvm_2048_bp5.json \
  --compute 512 --nodes 10 --cpus 2 --omp-threads 2 -t 24:00:00
```


#### Option B: Manual Execution (Advanced)
Run the model from the `build` directory:

```bash
cd $VVM_ROOT
mpirun -np 1 ./build/vvm
```

### Step 7: Choose an Output Engine

The `output.engine` key in the case JSON selects how results are written. The
three engines are not interchangeable — the right one depends on your build:

| Engine | Writes | I/O ranks | GPU build | CPU build |
|---|---|---|---|---|
| `HDF5` | one file per output time | none | yes | yes |
| `SST` | streams to I/O ranks, which write HDF5 | required (`--io N`) | yes (production path) | yes |
| `BP5` | one multi-step `.bp` dataset, written directly by the compute ranks | none | yes (host-staged) | yes (production path) |

`BP5` stages CUDA fields through host memory before writing; CPU builds can
write compatible layouts directly.

Every engine supports reduced-precision history (`"output": {"precision":
"float32"}`), which roughly halves output size without changing how the model
computes. Clocks and coordinates keep the model's precision. Note that HDF5
output is also the restart source, so narrowing it narrows what a restart
recovers.

See [Output](https://aaron-hsieh-0129.github.io/VVMex/user-guides/output/) for
the full option reference, engine comparison, and dataset-sizing guidance.

##### Asynchronous I/O (Optional)

For SST, specify the engine in your case JSON and allocate dedicated tasks for I/O.

For example, to use **1 GPU/CPU for the model** and **1 CPU for I/O**:

```bash
cd $VVM_ROOT
mpirun -np 2 ./build/vvm --io-tasks 1
```

To use **2 GPUs/CPUs for the model** and **2 CPUs for I/O**:

```bash
cd $VVM_ROOT
mpirun -np 4 ./build/vvm --io-tasks 2
```

For BP5, asynchronous writing is a JSON option (`"async_write": true`) rather
than a rank allocation — there are no I/O ranks to request.


## Reproducing the paper experiments

The configuration files used for the experiments in the VVMex v1.0 paper are provided in [`rundata/input_configs/default_cases/`](rundata/input_configs/default_cases/).

A case-by-case summary is available in [`rundata/input_configs/default_cases/README.md`](rundata/input_configs/default_cases/README.md). The directory includes the configuration files for the verification, validation, and performance experiments reported in the paper.




## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Acknowledgments & References

- **E3SM EAMxx**: The base implementation of the P3 microphysics and RRTMGP radiation schemes were adapted from the E3SM project.
- **P3 Microphysics**: The explicit condensation and evaporation processes between water vapor and cloud water, which are absent in the EAMxx version, have been re-implemented according to the original P3 formulation (e.g., *Morrison and Milbrandt, 2015*).
- **CWA Noah LSM**: The GPU-accelerated Noah land surface model is generously provided by the Central Weather Administration (CWA) of Taiwan.

## Citation

If you use VVMex, please cite the archived software release:

```bibtex
@software{vvmex_v1_0_0,
  title   = {{VVMex v1.0.0: GPU-capable refactoring of the Vector Vorticity cloud-resolving Model}},
  author  = {{Chin-Wei Hsieh}},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.21318987},
  url     = {https://doi.org/10.5281/zenodo.21318987}
}
```


## Contact & Support
GitHub Issues: For bugs, feature requests, or code contributions, please open an issue on the GitHub repository.

Email: Users can contact us for more questions regarding the model or its usage at B08209006@ntu.edu.tw.
