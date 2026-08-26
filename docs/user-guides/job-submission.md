# Job submission

Use the root-level `submit.py` wrapper for normal runs. It selects the binary
from `CMakePresets.json`, prepares library paths, creates output directories,
assigns compute and I/O ranks, and requests matching CPU/GPU resources.

## Choose a launch mode

| Situation | Use |
| --- | --- |
| First run or unfamiliar machine | Interactive `./submit.py` |
| Repeatable local test | `./submit.py --local ...` |
| Production batch run | `./submit.py ...` without `--local` |
| Low-level launcher debugging | Direct MPI commands at the end of this page |

Direct `mpirun` is intentionally the last option: an incorrect mapping can make
ranks share GPUs, leave I/O ranks without CPU time, or load libraries from the
wrong preset.


If you do not know which inputs to provide, run the wrapper without arguments and answer the prompts step by step:

```bash
cd $VVM_ROOT
./submit.py
```

The wizard detects available CMake presets, checks the configured output engine, and recommends a GPU count so the usual mapping is one MPI task per GPU.

It also shows which fields you need to fill in, explains the important run options, and prints an equivalent command-line invocation at the end. Save that command for future runs when you want to skip the interactive phase.

## Command-line mode

Use command-line mode for repeatable local tests and batch submissions.

### Local HDF5 test

```bash
cd $VVM_ROOT
./submit.py \
  --local \
  --preset <your_preset_name> \
  -c ./rundata/input_configs/default_cases/advection_u.json \
  --compute 4
```

### Local run on specific GPUs

For local execution, use `VVM_GPU_LIST` to select the physical GPU IDs exposed to VVMex ranks. This is the correct way to pin a local run to specific GPUs; `--gpus` controls the per-node GPU count used by the wrapper, while `VVM_GPU_LIST` selects the IDs.

```bash
cd $VVM_ROOT
VVM_GPU_LIST=0,1,2,3,4,5,6,7 ./submit.py --local \
  -c "rundata/input_configs/default_cases/taiwanvvm_2048.json" \
  --preset blaze \
  --compute 8 \
  --nodes 1
```

### SLURM compute run

```bash
cd $VVM_ROOT
./submit.py \
  --preset <your_preset_name> \
  -c ./rundata/input_configs/default_cases/sea_grass_mountain.json \
  --compute 16 \
  --nodes 1 \
  --gpus 16 \
  -t 24:00:00
```

`--cpus` is omitted on purpose: left alone, the wrapper sizes it to fill the node.
See [CPU allocation](#cpu-allocation).

### SLURM with SST I/O

```bash
cd $VVM_ROOT
./submit.py \
  --preset <your_preset_name> \
  -c ./rundata/input_configs/default_cases/sea_grass_mountain.json \
  --compute 16 \
  --io 4 \
  --nodes 4 \
  --gpus 4 \
  --io-cpus 1 \
  -t 24:00:00
```

This is 4 compute ranks and 1 I/O rank per node. `--gpus 4` covers the compute
ranks only -- the I/O rank is host-only and needs no device, so the GPU count
follows `ceil(compute / nodes)` and ignores `--io` entirely. Omitting `--gpus`
infers the same number.

## Full option reference

Every `submit.py` flag, with the value used when it is omitted. `--config` is the
only one that is mandatory in command-line mode; run the wizard instead and it
fills the rest in for you.

### Run selection

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c`, `--config PATH` | *(required)* | Case JSON handed to the executable. Its `output.engine` decides whether I/O ranks are required, and its `output.output_dir` is created before launch. |
| `--preset NAME` | *(none)* | `CMakePresets.json` entry to take the environment from. It also selects the **backend** (`VVM_ENABLE_GPU`) and the **binary** (`binaryDir/vvm`), so the launcher can never hand a CPU build a GPU mapping. Without it, library paths are whatever your shell already has. |
| `--local` | off | Run `tools/core_run.sh` immediately in this shell instead of submitting it with `sbatch`. All SLURM-only flags below are ignored. |

### Rank and core sizing

| Flag | Default | Meaning |
| --- | --- | --- |
| `--compute N` | `1` | Simulation MPI ranks. |
| `--io N` | inferred | Dedicated I/O-server ranks. Inferred as `--compute` for `SST` and `0` otherwise. A nonzero value with `HDF5` or `BP5` is rejected before an allocation is requested. |
| `--io-cpus N` | `1` | Cores reserved per I/O rank. Does **not** create I/O ranks. |
| `--nodes N` | `1` | Nodes to spread the total rank count over. |
| `--gpus N` | `ceil(compute / nodes)` | GPUs **per node**, covering compute ranks only. Ignored on CPU-only presets, which request no GPUs at all. |
| `--cpus N` | fills the node | `--cpus-per-task`, and the base for `OMP_NUM_THREADS`. Left unset, the wrapper reads the partition's `CPUEfctv` and divides by tasks per node; it falls back to `1` where SLURM cannot answer. See [CPU allocation](#cpu-allocation). |
| `--omp-threads N` | `--cpus` | OpenMP/Kokkos threads per compute rank, held fixed while `--cpus` varies. Useful on GPU runs where a rank wants cores for the launch loop but not an OpenMP worker on each. |

### SLURM job

Ignored under `--local`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-t`, `--time HH:MM:SS` | `24:00:00` | Wall-time limit. |
| `-A`, `--account NAME` | `MST114418` | Charging account. Override this on any other system. |
| `-p`, `--partition NAME` | `normal` | Partition. Also the partition the wrapper queries when sizing `--cpus`. |
| `--job-name NAME` | `VVMex` | `--job-name`. |
| `--out PATH` | `<output.output_dir>/%j.out` | Standard output file; `%j` expands to the job ID. Defaults into the run's output directory. Its directory is created. |
| `--err PATH` | `<output.output_dir>/%j.err` | Standard error file, alongside the standard output file. |

### Advanced SLURM

Command-line only — the wizard does not prompt for these.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--exclusive` / `--no-exclusive` | `--exclusive` | Whether to hold whole nodes. Exclusive is the default because shared nodes make the CPU-affinity split unreliable. |
| `--export VALUE` | `ALL` | `sbatch --export`. |
| `--nodelist LIST` | *(none)* | Restrict to specific nodes. |
| `--exclude LIST` | *(none)* | Exclude specific nodes. |
| `--contiguous` | off | Request contiguous nodes. |
| `--slurm-arg ARG` | *(none)* | Append a raw `sbatch` argument. Repeatable: `--slurm-arg='--qos=debug' --slurm-arg='--mem=0'`. |

### Environment variables you set

| Variable | Applies to | Meaning |
| --- | --- | --- |
| `VVM_GPU_LIST` | local GPU runs | Comma-separated **physical** GPU IDs to expose, e.g. `0,1,2,3`. Ranks map onto this list in local-rank order. Unset, they map modulo `--gpus`. CPU presets ignore it. |
| `VVM_OMP_THREADS` | any run | Same effect as `--omp-threads`, honoured straight from the caller's environment: `VVM_OMP_THREADS=4 ./submit.py ...`. Reported as `[Info] OMP_NUM_THREADS overridden: 11 -> 4`. |
| `VVM_EXTRA_LD_LIBRARY_PATH` | any run | Prepended ahead of the preset's own library directories, so one library can be swapped — an ADIOS2 built without Kokkos, say — without editing `CMakePresets.json`. Reported as `[Info] Honouring caller VVM_EXTRA_LD_LIBRARY_PATH`. |

### Environment variables the wrapper sets

`submit.py` exports these for `tools/core_run.sh` and the executable. They are
listed so the launcher banner and `[GPUMap]` lines are readable — setting them by
hand does not change the allocation SLURM was asked for, and will desynchronise
the two.

`VVM_ROOT`, `VVM_BACKEND`, `VVM_BINARY`, `VVM_CONFIG_FILE`, `VVM_ARGS`,
`VVM_COMPUTE_TASKS`, `VVM_IO_TASKS`, `VVM_TOTAL_TASKS`, `VVM_COMPUTE_PER_NODE`,
`VVM_IO_PER_NODE`, `VVM_GPUS`, `VVM_IO_CPUS`, `VVM_IO_ENGINE`, `VVM_OUTPUT_DIR`,
`VVM_ENV_SCRIPT`, `OMP_NUM_THREADS`.

### Executable options

`submit.py` builds this command line for you. It matters only when running
`mpirun` by hand:

| Argument | Meaning |
| --- | --- |
| `path/to/config.json` | First non-flag argument selects the case JSON. |
| `--io-tasks N` | Reserve the last **N** ranks as I/O servers. Only meaningful with `output.engine = "SST"`. |

## Choosing resources

`--compute` is the number of simulation MPI ranks. `--io` is the number of dedicated I/O ranks used when `output.engine` is `SST`. The total MPI size is:

```text
total MPI tasks = compute tasks + I/O tasks
```

For GPU runs, request enough GPUs so compute ranks do not unexpectedly share devices. For local runs, set `VVM_GPU_LIST` when you need specific physical GPU IDs. A common starting point is:

```text
GPUs per node >= ceil(compute tasks / nodes)
```

I/O ranks are host-only. They read and write SST streams and never initialize Kokkos, so they do **not** consume a GPU and are not counted when the wrapper sizes the GPU request. Two consequences:

- **I/O ranks may outnumber the GPUs.** `--compute 1 --io 4` on a single GPU is a valid configuration. Only compute ranks are mapped onto devices.
- **I/O ranks get their own core count,** set with `--io-cpus` (default 1). Under SLURM the wrapper reserves `io ranks x --io-cpus` cores per node and gives the remainder to the compute ranks, rather than splitting the node evenly across both roles. The launcher banner reports the split:

```text
Total Cores Used/Node: 96 | Cores/compute rank: 11 | Cores/IO rank: 1
```

## CPU allocation

`--cpus` is passed straight through to `--cpus-per-task`, which makes it the
single knob that decides **how much of each node the job actually holds**. SLURM
allocates:

```text
CPUs per node = --cpus x tasks per node
```

Under-request it and the rest of the node is not merely unused, it is
unavailable: the cgroup confines every rank and every thread they spawn to the
CPUs that were asked for.

### Leave it unset

Omitted, the wrapper fills the node. It takes a node name from the partition with
`sinfo`, reads that node's effective CPU count with `scontrol`, and divides by
tasks per node:

```text
[Info] CPUs per task: 6  (104 usable CPUs/node / 16 tasks/node)
```

It reads `CPUEfctv` rather than `CPUTot` on purpose. A node can advertise more
CPUs than a job is permitted to hold, and requesting the difference gets the job
rejected rather than scheduled. Where SLURM cannot answer -- unknown partition,
`--local`, no SLURM at all -- it falls back to `1` and prints the reason.

### Why `--cpus 1` is slow

It is a valid request and it will run, but on a node with many cores it is a
severe under-allocation. A concrete case: 8 compute + 8 I/O ranks per node on a
104-CPU node.

| | `--cpus 1` | `--cpus 6` (the default here) |
| --- | --- | --- |
| CPUs allocated per node | 16 | 96 |
| Cores per compute rank | 1 | 11 |
| CPU affinity | disabled | NUMA-local |

At one core per rank, that core has to carry the CUDA kernel-launch loop, the
MPI halo exchanges, the NCCL host side, and the CUDA driver threads -- while
sharing itself with an I/O rank. The GPU goes idle waiting on its host thread,
which shows up as low GPU utilization rather than as any obvious error.

### CPU affinity

Affinity is automatic and needs no flags. Every rank derives the same partition
independently, from the CPU set the cgroup actually granted
(`/proc/self/status`) and the PCI/NUMA topology in `sysfs`:

- **I/O ranks** take a reserved slice at the end of the CPU list, `--io-cpus`
  wide each. Compute ranks never touch it.
- **Compute ranks** are placed on the NUMA node their GPU is attached to, so the
  launch loop and its memory sit on the right socket.
- **`OMP_NUM_THREADS` is clamped** to the cores a rank actually owns, so the
  OpenMP pool cannot exceed its share.

The `[GPUMap]` lines report the decision:

```text
[GPUMap] ... local_rank=0 CUDA_VISIBLE_DEVICES=0 cpus=0,1,...,11 bind=numa0
[GPUMap] ... local_rank=8 role=io                cpus=88        bind=io_reserved
```

`bind=numa<N>` is the intended result. `bind=contiguous` means the topology was
unreadable and the CPU pool was split in rank order instead. `bind=off:<reason>`
means affinity declined, and the reason says why:

| `bind=` | meaning |
| --- | --- |
| `numa<N>` | placed on the NUMA node of this rank's GPU |
| `contiguous` | no usable topology; pool split in local-rank order |
| `io_reserved` | I/O rank on its reserved slice |
| `off:too_few_cpus` | fewer than 2 cores per compute rank |
| `off:no_taskset`, `off:no_cpuset` | `taskset` or `/proc` unavailable |

`off:too_few_cpus` is deliberate, and it is what `--cpus 1` produces. Pinning 16
ranks onto 16 CPUs would stop the kernel lending a blocked I/O rank's core to a
busy compute rank and buy nothing back, so the wrapper leaves the ranks unbound
where binding would hurt.

### Separating cores from OpenMP threads

`Cores/compute rank` also becomes `OMP_NUM_THREADS`. On a GPU-resident run that
is not always wanted: a compute rank needs cores for the launch loop, the CUDA
driver, NCCL and MPI progress, but not necessarily an OpenMP worker on each one,
and idle Kokkos/OpenMP workers spin at barriers. `VVM_OMP_THREADS` holds the
thread count fixed while `--cpus` varies:

```bash
VVM_OMP_THREADS=4 ./submit.py --preset <your_preset_name> -c my_config.json \
  --compute 64 --io 64 --nodes 8 --gpus 8
```

```text
[Info] OMP_NUM_THREADS overridden: 11 -> 4 (VVM_OMP_THREADS)
```

The I/O server spends its time in ADIOS2 and HDF5 and is effectively single-threaded, so the default of one core per I/O rank is usually right. Raise it only if the I/O ranks are demonstrably the bottleneck.

This requires ADIOS2 to be built without Kokkos support; see [GPU environment](../environment/gpu.md). With a Kokkos-enabled ADIOS2, each I/O rank opens a CUDA context anyway and the wrapper's GPU accounting no longer matches reality. VVMex warns about this at configure time.

If you use fewer GPUs than *compute* tasks per node, the wrapper will warn that MPI ranks may share GPUs.

## What the wrapper does

- Sets `VVM_ROOT` from the script location.
- Loads compilers and library paths from the selected `CMakePresets.json` entry.
- Creates the configured output and log directories.
- Writes `submit_command.sh` and the `code_snapshot/` provenance into the output directory.
- Requires `--io` when the JSON uses `output.engine = "SST"`.
- Rejects nonzero `--io` for HDF5 and BP5 before an allocation is requested.
- Runs `tools/core_run.sh` locally or submits it through `sbatch`.
- Maps local MPI ranks to visible GPU IDs and exports runtime variables used by the executable.
- Hides all GPUs from I/O server ranks, and gives them `--io-cpus` cores each.
- Sizes `--cpus` to fill the node when it is not given.
- Pins each rank to its own cores, compute ranks NUMA-local to their GPU.
- Lets waiting I/O ranks yield the CPU while compute ranks keep spinning, so a
  blocked I/O rank does not take cycles from the compute ranks it is waiting on.

## What the wrapper writes into the output directory

Every run drops its provenance next to the model output, so a finished run can be
read back and repeated without guessing what produced it.

| Path | Contents |
| --- | --- |
| `%j.out` / `%j.err` | SLURM logs, unless `--out` / `--err` say otherwise. |
| `submit_command.sh` | Executable script holding the exact `submit.py` command that started the run, with `VVM_ROOT` exported and a `cd` into it. Run it to repeat the run. |
| `code_snapshot/` | Copy of the source tree as it was at submit time, plus the case JSON and any profile/spatial input files. |
| `code_snapshot/ORIGINAL_GIT_VERSION.txt` | Commit, branch, `git describe`, upstream, remote, author, date, subject, and the `git status --porcelain` listing of the source repository at submit time. |
| `code_snapshot/ORIGINAL_GIT_UNCOMMITTED.patch` | The uncommitted diff at submit time. Written only when the source tree was dirty. |

The snapshot is itself a git repository. Its single commit reproduces the source
repository's `HEAD` exactly, and any uncommitted local edits are left in the working
tree, so `git -C <output_dir>/code_snapshot diff` shows precisely what the run
carried on top of that commit.

## Direct MPI commands

Direct MPI remains useful for small debugging sessions after the environment has already been prepared.

**Advanced only:** These commands bypass the wrapper's allocation checks. On multi-node or multi-GPU runs, make sure rank placement, GPU visibility, CPU binding, OpenMP threads, and I/O ranks are correct before trusting performance numbers.

```bash
cd $VVM_ROOT
mpirun -np 1 ./build/vvm ./rundata/input_configs/default_cases/advection_u.json
```

For asynchronous I/O, reserve the final ranks for I/O servers:

```bash
cd $VVM_ROOT

# 1 simulation rank + 1 I/O rank
mpirun -np 2 ./build/vvm ./rundata/input_configs/default_cases/advection_u.json --io-tasks 1

# 2 simulation ranks + 2 I/O ranks
mpirun -np 4 ./build/vvm ./rundata/input_configs/default_cases/advection_u.json --io-tasks 2
```
