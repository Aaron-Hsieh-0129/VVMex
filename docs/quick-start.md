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

This guide covers dependencies, building VVMex, and running the `vvm` executable. For detailed instructions on building the full dependency stack from source, please refer to the **[Environment Installation Guide](environment/index.md)** — [GPU](environment/gpu.md) and [CPU-only](environment/cpu.md) stacks are documented separately.

| Library | Minimum (tested) | Role |
| ------- | ------------------ | ---- |
| CMake | 3.20 | Build |
| Kokkos | 4.7+ | If not found, CMake may fetch Kokkos 4.5.x via `FetchContent` |
| HDF5 | 1.14.5+ | NetCDF / ADIOS2 stack |
| NetCDF-C | 4.4+ | I/O |
| NetCDF-Fortran | 4.4+ | Fortran interfaces |
| PnetCDF | 1.14+ | Parallel I/O |
| ADIOS2 | 2.11+ | Model output (`HDF5` / `SST`; `BP5` needs 2.12+) |

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
# GPU presets (blaze, nano4, nano5, spark, twnia2, twnia3) build into build/
cmake --preset <your_preset_name> -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# CPU-only presets (blaze-cpu, f1-cpu) build into build_cpu/
cmake --preset <your_cpu_preset_name> -DBUILD_TESTS=ON
cmake --build build_cpu -j$(nproc)
```

The main binary is **`vvm`** in the preset's `binaryDir` (`RUNTIME_OUTPUT_DIRECTORY` is the build root), so `build/vvm` for GPU presets and `build_cpu/vvm` for CPU-only ones. `submit.py` reads the same preset and picks the matching binary, so a CPU build is never handed a GPU mapping.

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

Two things decide what a run command looks like, and they are set in two different places:

| Decides | Set by | Effect on the command |
| --- | --- | --- |
| **Backend** — GPU or CPU | `--preset` (the preset's `VVM_ENABLE_GPU`) | GPU runs use `--gpus` / `VVM_GPU_LIST`; CPU runs use `--cpus` / `--omp-threads` and ignore `--gpus` |
| **Output engine** — `HDF5`, `SST`, `BP5` | `output.engine` in the case JSON | `SST` needs I/O ranks (`--io`); `HDF5` and `BP5` reject a nonzero `--io` |

So pick the section below that matches your build, then the engine row that matches your JSON. Engine trade-offs are in [Output](user-guides/output.md).

### Interactive setup (either backend)

```bash
./submit.py
```

If you do not know which inputs to provide, run `./submit.py` with no arguments. The interactive phase detects your presets, reads the configured output engine, prompts for the required values step by step, and prints an equivalent command-line invocation at the end so you can reuse it for future runs.

### Run on GPU

GPU presets: `blaze`, `nano4`, `nano5`, `spark`, `twnia2`, `twnia3`. One MPI compute rank per GPU is the normal mapping. `--gpus` is GPUs **per node** and covers compute ranks only — I/O ranks are host-only. Omit it and the wrapper infers `ceil(compute / nodes)`.

| `output.engine` | I/O ranks | Extra flags | Behaviour |
| --- | --- | --- | --- |
| `HDF5` | none | *(none)* | One `.h5` file per output time, written synchronously by the compute ranks. Restart-capable; best for small runs and reference output. |
| `SST` | required | `--io N`, optionally `--io-cpus` | Compute ranks stream to dedicated host-only I/O ranks that write HDF5. Omit `--io` and the wrapper sets it for you. Production path on GPU. |
| `BP5` | none (`--io` rejected) | *(none)* | Compute ranks write one multi-step `.bp` dataset directly. Fields are host-staged from device memory. History only — no restart. |

```bash
# HDF5 -- local test run without SLURM, 4 ranks on 4 GPUs
./submit.py --local --preset blaze \
  -c ./rundata/input_configs/default_cases/advection_u.json \
  --compute 4

# HDF5 -- local run pinned to specific physical GPU IDs
VVM_GPU_LIST=0,1,2,3,4,5,6,7 ./submit.py --local --preset blaze \
  -c ./rundata/input_configs/default_cases/taiwanvvm_2048.json \
  --compute 8 --nodes 1

# HDF5 -- SLURM, 16 compute ranks on 1 node
./submit.py --preset <your_preset_name> \
  -c ./rundata/input_configs/default_cases/sea_grass_mountain.json \
  --compute 16 --nodes 1 --gpus 16 -t 24:00:00

# SST -- SLURM, 4 compute + 1 I/O rank per node; --io may be omitted
./submit.py --preset <your_preset_name> \
  -c ./rundata/input_configs/default_cases/sea_grass_mountain.json \
  --compute 16 --io 4 --nodes 4 --gpus 4 --io-cpus 1 -t 24:00:00

# BP5 -- SLURM, 8 compute ranks on 1 node, no I/O ranks
./submit.py --preset blaze -c <case>.json \
  --compute 8 --nodes 1 --gpus 8 -t 04:00:00
```

`VVM_GPU_LIST` selects physical GPU IDs for **local** runs; `--gpus` is the per-node count the wrapper requests from SLURM. They are not interchangeable.

### Run on CPU

CPU-only presets: `blaze-cpu`, `f1-cpu`. Kokkos runs on OpenMP, standard MPI replaces NCCL, and no device is touched at run time. `--gpus` is ignored with an `[Info]` message and no GPUs are requested from SLURM, so the job does not queue for resources it will never use. Instead, size the run with ranks x threads: `--cpus` is cores per rank and `--omp-threads` sets the OpenMP team when you want it smaller than the core count.

| `output.engine` | I/O ranks | Extra flags | Behaviour |
| --- | --- | --- | --- |
| `HDF5` | none | *(none)* | Same synchronous one-file-per-step path as GPU. Restart-capable; the reference output BP5 is validated against. |
| `SST` | required | `--io N`, optionally `--io-cpus` | Supported, but the extra ranks buy less here than on GPU since there is no device to keep busy. |
| `BP5` | none (`--io` rejected) | *(none)* | Validated CPU production path — one multi-step `.bp` dataset, no I/O ranks, no SST transport. |

```bash
# HDF5 -- local test run without SLURM, 4 ranks
./submit.py --local --preset blaze-cpu \
  -c ./rundata/input_configs/default_cases/advection_u.json \
  --compute 4

# BP5 -- SLURM, 1024 ranks on 20 nodes, 2 OpenMP threads each
./submit.py --preset f1-cpu -c <case>.json \
  --compute 1024 --nodes 20 --cpus 2 --omp-threads 2 \
  --partition ct2k --account <account> -t 64:00:00

# SST -- SLURM, 16 compute + 4 I/O ranks
./submit.py --preset f1-cpu -c <case>.json \
  --compute 16 --io 4 --nodes 4 --io-cpus 1 -t 24:00:00
```

More detail on rank and core sizing: [Job submission](user-guides/job-submission.md).

### A note on `--cpus`

Most commands above do not pass `--cpus`, and that is the recommended way to run
them. Left unset, the wrapper sizes it to fill the node and then pins each rank
to its own cores, with compute ranks placed on the NUMA node of their GPU.

`--cpus` maps to `--cpus-per-task`, so it decides how much of the node the job
holds: `--cpus x tasks per node`. Passing a small value such as `--cpus 1` is
valid but under-allocates badly on a many-core node -- one core per rank has to
carry the CUDA launch loop, MPI, NCCL and the driver threads at once, which
starves the GPU without producing any error. On a CPU run the same flag is what
sets the OpenMP team size, so it is normally given deliberately together with
`--omp-threads`. See
[CPU allocation](user-guides/job-submission.md#cpu-allocation).

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

Details: [Output](user-guides/output.md).

## Documentation site

To preview this documentation locally (requires [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/)):

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Use `requirements-docs.txt` so your local MkDocs/Material versions match GitHub Actions. Then open the served URL in your browser.
