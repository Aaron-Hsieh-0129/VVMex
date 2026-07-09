# Quick Start

This guide covers dependencies, building VVMex, and running jobs through the recommended `submit.py` wrapper.

## Requirements

### Compilers and runtime

| Component | Minimum | Notes |
| --------- | ------- | ----- |
| C++ compiler | GCC 11+ | C++17 |
| CUDA | 11.4+ | NVIDIA GPUs are the tested target |
| MPI | OpenMPI 4.x+ | Use `mpic++` / `mpicc` / `mpifort` consistent with your toolchain |

**NVHPC 24.9+** is recommended on NVIDIA systems: it bundles CUDA, OpenMPI, and math libraries that align with the CMake hints used in `CMakePresets.json`.

### Libraries

This guide covers dependencies, building VVMex, and running the `vvm` executable. For detailed instructions on building the full dependency stack from source, please refer to the **[Environment Installation Guide](user-guides/environment.md)**.

| Library | Minimum (tested) | Role |
| ------- | ------------------ | ---- |
| CMake | 3.20 | Build |
| Kokkos | 4.7+ | If not found, CMake may fetch Kokkos 4.5.x via `FetchContent` |
| HDF5 | 1.14.5+ | NetCDF / ADIOS2 stack |
| NetCDF-C | 4.4+ | I/O |
| NetCDF-Fortran | 4.4+ | Fortran interfaces |
| PnetCDF | 1.14+ | Parallel I/O |
| ADIOS2 | 2.11+ | Model output (`HDF5` / `SST` engines) |

The root `CMakeLists.txt` also expects **NVIDIA CPU Math Library** (`libnvcpumath`) and, when `ENABLE_NCCL` is ON (default), **NCCL** under `NVHPC_DIR`. Turn off NCCL with `-DENABLE_NCCL=OFF` only if you have a matching build and know the implications for halo exchange.

## Build

### 1. Clone the repository

```bash
git clone https://github.com/Aaron-Hsieh-0129/VVMex.git
cd VVMex
```

### 2. Environment setup

Define `VVM_ROOT` for convenient build and run commands:

```bash
export VVM_ROOT=/absolute/path/to/your/VVMex
cd $VVM_ROOT
```

`submit.py` also auto-detects the project root when launched from the repository.

### 3. Configure CMake presets

Edit `CMakePresets.json` (or pass cache variables on the command line) so that:

- `CMAKE_CXX_COMPILER`, `CMAKE_C_COMPILER`, `CMAKE_Fortran_COMPILER` point to MPI wrappers.
- `NVHPC_DIR` matches your installation.
- `HDF5_DIR`, `NETCDF_C_DIR`, `NETCDF_Fortran_DIR`, `PNETCDF_DIR` point to your dependency prefixes.

`find_package(ADIOS2 REQUIRED CXX MPI)` must succeed using your `CMAKE_PREFIX_PATH` or install layout.

### 4. Configure and compile

```bash
cmake --preset <your_preset_name> -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

The main binary is **`build/vvm`** (`RUNTIME_OUTPUT_DIRECTORY` is the build root).

## Configure a run

1. Choose a sample from **`rundata/input_configs/default_cases/`**. These JSON files are the recommended starting points for runnable VVMex examples.

2. Set **`output.output_dir`** to a directory you can write.

3. Check **`initial_conditions.source_file`**. Default cases point at profiles under `rundata/initial_conditions/profiles/default_cases/`.

4. Check **`netcdf_reader.source_file`**. Default cases point at spatial NetCDF inputs under `rundata/initial_conditions/spatial/default_cases/`. These NetCDF files can be regenerated with `tools/generate_init_nc.py`.

5. Optional: pass a different config path as the **first non-option argument**:

   ```bash
   ./submit.py --local --preset <your_preset_name> -c /path/to/my_config.json --compute 1
   ```

Full key reference: [Model configuration](user-guides/configuration.md).

## Run

Use `submit.py` from the project root. It safely handles SLURM resource allocation, local execution, MPI task counts, GPU assignment, directory creation, and asynchronous I/O task separation.

**Recommended for all normal runs:** Direct `mpirun` commands are advanced/debugging commands. Use `submit.py` for performance runs because CPU/GPU assignment and I/O-rank allocation strongly affect speed.

### Interactive setup

```bash
./submit.py
```

If you do not know which inputs to provide, run `./submit.py` with no arguments. The interactive phase prompts for the required values step by step, explains the run options, and prints an equivalent command-line invocation at the end so you can reuse it for future runs.

### Command-line execution

```bash
# Local test run without SLURM (4 MPI tasks)
./submit.py --local --preset <your_preset_name> -c ./rundata/input_configs/default_cases/advection_u.json --compute 4

# Local run on specific GPU IDs
VVM_GPU_LIST=0,1,2,3,4,5,6,7 ./submit.py --local \
  -c "rundata/input_configs/default_cases/taiwanvvm_2048.json" \
  --preset blaze \
  --compute 8 \
  --nodes 1

# Submit to SLURM (16 Compute tasks, 1 Node)
./submit.py --preset <your_preset_name> -c ./rundata/input_configs/default_cases/sea_grass_mountain.json --compute 16 --nodes 1 --gpus 16
```

### Asynchronous I/O with SST
If `output.engine` is `SST` in your JSON, `submit.py` sets the required I/O ranks automatically when `--io` is omitted. You can run the same style of command and let the wrapper choose the I/O task count:

```bash
./submit.py --preset <your_preset_name> -c my_config.json --compute 16 --nodes 4 --gpus 4
```

Use `--io N` only when you want to override the wrapper's inferred I/O rank count. More detail: [Job submission](user-guides/job-submission.md).

## Direct MPI (advanced)

Manual MPI is useful for small debug sessions after the environment has already been prepared. It bypasses the wrapper's resource checks, so verify rank placement, GPU visibility, CPU binding, and OpenMP settings yourself.

```bash
mpirun -np 1 ./build/vvm ./rundata/input_configs/default_cases/advection_u.json
```

### Asynchronous I/O (optional)

Reserve ranks for dedicated I/O servers that consume an ADIOS2 **SST** stream and write HDF5 (`output.engine` must be `SST`):

```bash
# 1 simulation rank + 1 I/O rank
mpirun -np 2 ./build/vvm ./rundata/input_configs/default_cases/advection_u.json --io-tasks 1

# 2 simulation ranks + 2 I/O ranks
mpirun -np 4 ./build/vvm ./rundata/input_configs/default_cases/advection_u.json --io-tasks 2
```

Details: [I/O management](user-guides/io-management.md).

## Documentation site

To preview this documentation locally (requires [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/)):

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Use `requirements-docs.txt` so your local MkDocs/Material versions match GitHub Actions. Then open the served URL in your browser.
