# Output

All VVMex output goes through **ADIOS2**. The `output` block of the case JSON
selects the engine, the destination, and which fields are written. Output cadence
comes from `simulation.output_interval_s`, not from the command line.

Three engines exist, and they are not interchangeable — the right one depends on
which backend you built.

## Choosing an engine

| | `HDF5` | `SST` | `BP5` |
|---|---|---|---|
| Writes | one file per output time | streams to I/O ranks, which write HDF5 | one multi-step `.bp` dataset |
| Dedicated I/O ranks | none | required (`--io N`) | none |
| CPU build | supported | supported | **supported, validated** |
| GPU build | supported | supported, production path | **not available — see below** |
| Restart source | yes | yes (writes HDF5) | no, history only |
| Best for | small runs, restarts, reference output | GPU production | CPU production |

If you are unsure: **HDF5** for anything small or when you need restarts,
**BP5** for CPU production, **SST** for GPU production.

## Options common to every engine

```json
"output": {
  "output_dir": "./output/my_case",
  "output_filename_prefix": "vvm_output",
  "output_initial_step": true,
  "fields_to_output": ["u", "v", "w", "th", "qv"],
  "output_grid": {
    "x_start": 0, "x_end": -1,
    "y_start": 0, "y_end": -1,
    "z_start": 0, "z_end": -1
  }
}
```

| Key | Meaning |
|---|---|
| `output_dir` | Destination directory. Created if missing. |
| `output_filename_prefix` | Base name for files or the dataset. |
| `output_initial_step` | Write step 0 before the first model step. Default `true`. |
| `fields_to_output` | Field names that must exist on the model state. A missing, disabled, or misspelled name is an error, not a warning. |
| `output_grid` | Inclusive index bounds per direction; `-1` means "to the end". Halo cells are never written. |

Cadence is `simulation.output_interval_s`. With `dt_s: 1.0` and
`output_interval_s: 600.0` you get one output every 600 model steps.

## HDF5

The default. Compute ranks write directly through ADIOS2's HDF5 engine, one file
per output time, named `<prefix>_NNNNNN.h5`.

```json
"output": { "engine": "HDF5" }
```

Synchronous, so the model waits for each write. This is the reference output
path — the BP5 compatibility tests compare against it bit-for-bit — and the
supported route for checkpoint/restart.

## SST

`output.engine: "SST"` streams data over an ADIOS2 SST stream to dedicated I/O
ranks, which relay it to HDF5. This overlaps compute and I/O at the cost of
extra ranks.

```bash
./submit.py -c <case>.json --compute 16 --io 4 --nodes 4 --gpus 5
```

Total MPI ranks = compute + I/O. The first `world_size - N` ranks run the model;
the last `N` run `VVM::IO::run_io_server()` (`src/io/IOServer.cpp`), open the
stream named from `output_dir` and `output_filename_prefix`, and write collective
HDF5. Submit with `submit.py --io N` so ranks, cores, and GPUs are requested
together.

Before starting, rank 0 removes a stale `output_dir/output_filename_prefix.sst`
directory so a previous run's stream is not reused.

!!! warning "Known scaling limit"

    SST has reproducible failures at 384–512 writers with a small reader cohort
    on this class of system — segfaults inside the data plane's read-request
    handler during the first output step, across WAN/sockets, UCX, and the stock
    MPI transport. This is what motivated the direct BP5 path. See the
    [SST large-scale report](../adios2-sst-large-scale-report.md) for the full
    evidence and the open questions put to the ADIOS2 team.

## BP5

Direct compute-rank output to a single multi-step `.bp` dataset. No I/O ranks,
no SST, no relay, and no patched ADIOS2. This is the CPU production path.

```json
"output": {
  "engine": "BP5",
  "output_dir": "./output/taiwanvvm_2048_bp5",
  "output_filename_prefix": "vvm_output",
  "bp5": {
    "aggregation_type": "TwoLevelShm",
    "num_subfiles": 20,
    "stats_level": 0,
    "async_write": false,
    "buffer_mode": "direct",
    "precision": "native",
    "overwrite": false
  }
}
```

Submit without `--io`; `submit.py` detects BP5 and assigns zero I/O ranks. A
nonzero `--io` is rejected.

```bash
./submit.py --preset f1-cpu -c <case>.json \
    --compute 1024 --nodes 20 --cpus 2 --omp-threads 2 \
    --partition ct2k --account <account> -t 64:00:00
```

### BP5 options

| Key | Meaning |
|---|---|
| `aggregation_type` | Currently must be `TwoLevelShm`. |
| `num_subfiles` | Data subfile count. Start with the number of nodes. |
| `stats_level` | `0` minimises statistics work; `1` enables BP5 statistics. |
| `async_write` | Background file writing. Default `false`. |
| `buffer_mode` | `direct` passes the model's own memory with a memory selection; `pack` stages into persistent contiguous buffers. |
| `precision` | On-disk float type for field data: `native`, `float32`, or `float64`. |
| `overwrite` | When `false`, refuse to replace an existing dataset. |

Unknown keys and invalid values are errors. Every resolved setting is compared
across all compute ranks before the dataset is opened, so ranks cannot disagree.

### Output precision

History precision is independent of `VVM::Real`. A double-precision build can
write `float32` history without changing how it computes — roughly halving the
dataset for a run dominated by 3-D fields.

| Value | On-disk field type |
|---|---|
| `native` (default) | `VVM::Real` — identical to the behaviour before this option existed |
| `float32`, `float`, `single` | `float` |
| `float64`, `double` | `double` |

Case-insensitive. **Field data only.** `time`, `model_time_s`, `model_step`, and
`coordinates/*` keep their types: they are kilobytes per step against tens of
gigabytes of field data, so narrowing them would save nothing while making a
timestamp or grid coordinate lossy.

Two dataset attributes record what happened — `vvm_field_precision` (the on-disk
field type) and `vvm_real_precision` (the model's working precision). Differing
means the history is a narrowed copy; matching means it is lossless.

Converting requires a staging buffer, so `buffer_mode: direct` resolves to
`pack` automatically when precision converts. This is reported, not silent:

```
[BP5] Field precision: float32 (requested 'float32', model VVM::Real is float64)
[BP5] CPU buffer mode: pack  (requested 'direct'; converting precision requires a staging buffer)
```

The conversion is a `static_cast` performed in the same pass that strips halo
cells, so it costs no copy beyond the pack that `pack` mode already does.

### Asynchronous writing

`async_write: true` lets ADIOS2 write to disk in the background while the model
continues. VVMex switches its `Put` calls to `Mode::Sync` when this is on, so the
data is copied into ADIOS2's buffer before `write()` returns and the model may
immediately modify its own fields — only ADIOS2-internal file work is
asynchronous.

Measured at 1024 ranks on 20 nodes, `EndStep` stayed flat at ~0.06 s across 14
output steps with no backlog growth. Note that `submit.py` binds each rank to
exactly `--cpus` cores and the ADIOS2 background thread inherits that mask, so it
shares cores with the OpenMP team. If async shows no benefit, that is the first
thing to check.

### Not available in GPU builds

Selecting `engine: "BP5"` in a CUDA build **throws at construction**:

```
Direct BP5 history output is currently enabled only for the CPU build.
Use the legacy output path for GPU runs until GpuFieldSource is implemented.
```

This is a deliberate guard, not an untested path — the field-source layer that
stages data for ADIOS2 has only a CPU implementation, and the BP5 test suite is
gated off entirely in GPU builds (`if(NOT VVM_ENABLE_GPU)` in
`tests/CMakeLists.txt`). A `GpuFieldSource` sitting behind the same interface is
future work; the schema and writer lifecycle are designed not to change when it
lands. **GPU runs should use SST or HDF5.**

## Sizing and quota

Output volume is easy to underestimate and is the most common way a long run
dies. Worked example — TaiwanVVM 2048², ~60 fields, 17 of them 3-D, `float64`:

| | |
|---|---|
| Per output step | **47.5 GB** |
| `output_interval_s: 600` over 24 h | 144 steps ≈ **6.8 TB** |
| Same run at `precision: float32` | ≈ **3.4 TB** |
| At `output_interval_s: 1800`, `float32` | ≈ **1.1 TB** |

The 3-D fields are ~97% of those bytes, so trimming `fields_to_output` saves
proportionally more than anything else.

BP5 prints its own estimate at startup — check it against your quota before
committing to a long job:

```
[BP5] Estimated logical bytes/step: 47513111264
```

!!! danger "Quota exhaustion aborts the job"

    Running out of quota surfaces as `errno = 122: Disk quota exceeded` from
    every aggregator at once, and the uncaught exception takes down all ranks.
    Automatic dataset rotation is not implemented, so one run writes one dataset
    however large it grows. Check headroom with `lfs quota -h -u $USER <path>`
    before submitting, and remember that `overwrite: false` will refuse to start
    if a previous dataset is still sitting in `output_dir`.

## Reading the output

HDF5 output is readable by any HDF5 or NetCDF4 tool.

BP5 datasets need ADIOS2. To inspect one:

```bash
<adios2-prefix>/bin/bpls -l output/my_case/vvm_output.bp
```

The hand-built ADIOS2 in the environment guides has no Python bindings. Because
BP5 is a portable format, a separate reader-only install works fine and does not
have to match the build that wrote the data:

```bash
conda install -c conda-forge adios2
```

```python
import adios2

with adios2.FileReader("output/my_case/vvm_output.bp") as f:
    print(f.num_steps())
    print(sorted(f.available_variables()))
    field = f.read("precip_liq_surf_mass", step_selection=[0, 1])
```

Reading is slower than the file size suggests: each 2-D field was written by
every compute rank as a separate block, so one field at 1024 ranks costs ~1024
small reads scattered across the subfiles. Read the steps you need rather than
looping over all of them.

## Where the code lives

| Path | Role |
|---|---|
| `src/io/OutputManager.*` | HDF5 and SST writer, schema, field packing |
| `src/io/IOServer.*` | SST reader and HDF5 relay |
| `src/io/history/HistoryWriter.hpp` | Neutral `write`/`close` interface used by `main.cpp` |
| `src/io/bp5/` | Direct BP5 writer: config, schema, CPU staging, lifecycle |

For BP5 internals — architecture, dataset layout, the compatibility contract
against HDF5 output, buffer lifetime, and the validation matrix — see
[BP5 output internals](../developer-guides/bp5-output.md).
