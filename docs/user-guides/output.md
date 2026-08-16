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
| GPU build | supported | supported, production path | **supported, host-staged** |
| Restart source | yes | yes (writes HDF5) | yes, any step of the dataset |
| Best for | small runs, restarts, reference output | GPU production with I/O ranks | direct multi-step output |

If you are unsure: **HDF5** for anything small or when you need restarts,
**BP5** for direct output without I/O ranks, **SST** for GPU production with I/O ranks.

The engine changes the container, not the numbers. For the same case and the
same compute-rank count, all three write **bit-for-bit identical** field values
and identical metadata; a test compares them through each format's own reader on
every output step (`ctest -R Compare_engine_outputs`). Choose an engine on I/O
cost and workflow, never on results.

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
| `fields_to_output` | Field names to write. A name the run never registered — a disabled microphysics or radiation output, say — is skipped with a message rather than written. On HDF5 and SST a name that is neither a known optional field nor a registered one is still an error, so typos are caught. |
| `output_grid` | Inclusive index bounds per direction; `-1` means "to the end". Halo cells are never written. |
| `precision` | On-disk float type for field data: `native` (default), `float32`, or `float64`. See below. |

Cadence is `simulation.output_interval_s`. With `dt_s: 1.0` and
`output_interval_s: 600.0` you get one output every 600 model steps.

### Output precision

History precision is independent of `VVM::Real`, on **every** engine. A
double-precision build can write `float32` history without changing how it
computes — roughly halving the output for a run dominated by 3-D fields.

```json
"output": { "precision": "float32" }
```

| Value | On-disk field type |
|---|---|
| `native` (default) | `VVM::Real` — identical to the behaviour before this option existed |
| `float32`, `float`, `single` | `float` |
| `float64`, `double` | `double` |

Case-insensitive. **Field data only.** `time`, `model_time_s`, `model_step`, and
`coordinates/*` keep their types: they are kilobytes per step against tens of
gigabytes of field data, so narrowing them would save nothing while making a
timestamp or grid coordinate lossy.

Two attributes record what happened — `vvm_field_precision` (the on-disk field
type) and `vvm_real_precision` (the model's working precision). Differing means
the history is a narrowed copy; matching means it is lossless.

Per engine:

| Engine | How the choice is applied |
|---|---|
| `HDF5` | Field datasets are created at the chosen type; the staged host copy is converted per field per output. |
| `SST` | Fields stream at the chosen type, and the I/O server relays whatever type the stream declares into HDF5. |
| `BP5` | As above; `output.bp5.precision` overrides `output.precision` when both are set. |

!!! warning "HDF5 output is also the restart source"

    A restart reads the HDF5 history files, so `precision: float32` on the HDF5
    or SST path means a later restart resumes from float32 state. HDF5 converts
    on read, so the run still starts — it just starts from less precise numbers,
    and is no longer bit-for-bit with an uninterrupted run. Keep `native` for
    any run you intend to restart. The writer says so at startup when both are
    set.

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
    on this class of system: segfaults inside the data plane's read-request
    handler during the first output step, reproduced across WAN/sockets, UCX,
    and the stock MPI transport. Reducing writer/reader fan-out and disabling
    speculative preload helped but was not shown to resolve it. This is what
    motivated the direct BP5 path, which uses no SST transport at all.

## BP5

Direct compute-rank output to a single multi-step `.bp` dataset. No I/O ranks,
no SST, no relay, and no patched ADIOS2. Both CPU and GPU builds support it.

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
# CPU
./submit.py --preset f1-cpu -c <case>.json \
    --compute 1024 --nodes 20 --cpus 2 --omp-threads 2 \
    --partition ct2k --account <account> -t 64:00:00

# GPU
./submit.py --preset blaze -c <case>.json \
    --compute 8 --nodes 1 --gpus 8 -t 04:00:00
```

### BP5 options

| Key | Meaning |
|---|---|
| `aggregation_type` | Currently must be `TwoLevelShm`. |
| `num_subfiles` | Data subfile count. Start with the number of nodes. |
| `stats_level` | `0` minimises statistics work; `1` enables BP5 statistics. |
| `async_write` | Background file writing. Default `false`. |
| `buffer_mode` | `direct` passes compatible CPU memory with a memory selection; `pack` stages into persistent contiguous buffers. CUDA builds always resolve this to `pack` for host staging. |
| `precision` | BP5-only override of [`output.precision`](#output-precision). Omit it to follow that key. |
| `overwrite` | When `false`, refuse to replace an existing dataset. |

Unknown keys and invalid values are errors. Every resolved setting is compared
across all compute ranks before the dataset is opened, so ranks cannot disagree.

### Restarting from a BP5 dataset

A `.bp` dataset is a restart source like an HDF5 file, with one extra choice: it
holds every output time, so the run has to say which one to resume from.

```json
"restart": {
  "enable": true,
  "source_file": "./output/taiwanvvm_2048_bp5/vvm_output.bp",
  "step_index": -1
}
```

`step_index: -1` (the default) resumes from the last step written; any valid
index picks an earlier one. Everything else matches the HDF5 route — the same
`restart.variables_to_read` selection, the same clock recovery from
`model_time_s` / `model_step`, and the same rule that a field must be in
`output.fields_to_output` to be recoverable.

Two things are worth knowing:

- **The reader does not care how many ranks wrote the dataset.** A run written by
  8 ranks restarts on 1, or on 64: each rank reads its own slab of the global
  array.
- **The run must not overwrite the dataset it resumed from.** Point
  `output.output_dir` (or the prefix) somewhere else for the resumed run; the
  writer refuses to start otherwise, rather than deleting the history mid-run.

A narrowed dataset (`precision: float32`) restarts too, from float32 numbers —
the same caveat as narrowed HDF5 output.

### GrADS descriptor

Like the HDF5 and SST paths, a BP5 run writes `<output_dir>/vvm.ctl` next to the
dataset. It differs only where the container does: `DTYPE bp5`, `DSET` pointing
at the `.bp` directory, and no `OPTIONS template`, because one dataset holds
every step.

```text
DSET ^vvm_output.bp
DTYPE bp5
...
VARS 10
w 44 z,y,x Vertical wind (m s-1)
Tg=>tg 0 y,x Surface skin temperature (K)
precip_liq_surf_mass=>precip_liq_surf 0 y,x Surface liquid precipitation flux (kg m-2 s-1)
ENDVARS
```

Reading it needs a GrADS build with BP5 support, which stock GrADS and OpenGrADS
do not have. Build it from
[Aaron-Hsieh-0129/opengrads-update](https://github.com/Aaron-Hsieh-0129/opengrads-update),
an OpenGrADS fork with a BP5 reader added; point its configure at the same ADIOS2
install the model uses. `q config` on the resulting binary reports `adios2-bp5`
when the support is present.

Two consequences of what GrADS accepts are worth knowing:

- GrADS variable names are lowercase and at most 15 characters, so a field whose
  name is neither gets an alias (`Tg=>tg`). Use the name on the right in GrADS.
- Every GrADS variable must map one x and one y dimension, so z-only profile
  fields (`thbar`, `rhobar`, …) cannot appear at all — declaring one would fail
  the whole `open`. They are listed in a comment line instead and remain
  readable from the dataset through ADIOS2.

### BP5 precision and buffering

BP5 takes its precision from the engine-neutral
[`output.precision`](#output-precision), and `output.bp5.precision` overrides it
for BP5 alone — useful for writing narrowed BP5 history while an HDF5 restart
file stays lossless.

Converting requires a staging buffer, so `buffer_mode: direct` resolves to
`pack` automatically when precision converts. CUDA builds also always resolve
to `pack`, because fields must be copied from device memory into host buffers
for the non-GPU-aware ADIOS2 writer. This is reported, not silent:

```
[BP5] Field precision: float32 (requested 'float32', model VVM::Real is float64)
[BP5] Field buffer mode: pack  (requested 'direct'; CUDA fields require host staging)
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

### GPU host staging

On CUDA, each selected field is mirrored to host memory with a synchronized
Kokkos deep copy, then packed into the same persistent buffer used by CPU
`pack` mode. ADIOS2 therefore receives only host pointers, and a request for
`buffer_mode: direct` is safely downgraded to `pack`.

The extra mirror currently allocates once per field and output step. This is
correct but may be worth optimizing with persistent mirrors after production
measurements show that host allocation is material.

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

BP5 datasets need ADIOS2, or the BP5-capable GrADS from
[Aaron-Hsieh-0129/opengrads-update](https://github.com/Aaron-Hsieh-0129/opengrads-update)
reading the `vvm.ctl` the run writes — see
[GrADS descriptor](#grads-descriptor). To inspect one:

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

### Converting to HDF5

VVMex ships no converter, but ADIOS2 does: `bp2h5`, present in the ADIOS2 prefix
whenever ADIOS2 was built with HDF5 support, which the environment guides enable.

```bash
export ADIOS2=<adios2-prefix>
export LD_LIBRARY_PATH=$ADIOS2/lib64:<hdf5-and-compiler-runtime-paths>:$LD_LIBRARY_PATH
export HDF5_USE_FILE_LOCKING=FALSE

$ADIOS2/bin/bp2h5     output/my_case/vvm_output.bp converted.h5   # serial
$ADIOS2/bin/bp2h5_mpi output/my_case/vvm_output.bp converted.h5   # MPI
```

The wrapper sets no library paths of its own. ADIOS2's own libraries are found
through `RPATH`, but HDF5 and the compiler runtime are not.

`bp2h5` is a thin wrapper around `adios2_reorganize` that fixes the engines and
their parameters:

```text
BPFile "StreamReader=ON"  ->  HDF5 ""
```

It writes **one** HDF5 file containing `Step0 … StepN`. The VVMex HDF5 engine
writes one file per output time, each holding a single `Step0` group. Same
per-step structure, different file granularity — tooling written against VVMex
HDF5 output must walk steps rather than files.

#### Converting while the run is still going

This works, and needs no extra flags. `StreamReader=ON` is what makes it work,
and the wrapper already sets it.

`StreamReader` is a BP5 **reader** parameter, default `false`. With it off the
reader parses the whole metadata index at open, so a dataset that is still being
appended is frozen at whatever existed at that moment. With it on the index is
parsed incrementally and steps written after open become visible.

`adios2_reorganize` polls with a ten-second timeout per step. Because `bp2h5`
names the read engine `BPFile`, the tool treats its input as a file rather than a
stream and **stops** at the first ten-second gap instead of following the writer:

```text
Timeout waiting for next step. If this is a live stream through file, use a
different reading engine, like FileStream or BP4. If it is an unclosed BP file,
you may manually close it with using adios_deactive_bp.sh.
Bye after processing 37 steps
```

That message is the normal exit path, not an error. The HDF5 file left behind is
complete and valid for the steps it converted.

What follows from that:

- **It is a snapshot, not a follow.** Each invocation converts what has been
  flushed so far and exits. Re-run it for a newer snapshot.
- **There is no incremental mode.** Every run starts from step 0 and writes a
  fresh output, so repeatedly converting a multi-terabyte dataset re-reads and
  re-writes all of it each time.
- **Storage doubles.** A 6.8 TB dataset becomes 6.8 TB of BP5 plus 6.8 TB of HDF5.
- With `async_write: true` a step reaches disk later than `EndStep` returns, so
  the newest step or two may not be visible yet.

!!! warning "Not during a performance measurement"

    The converter reads the whole dataset and writes an equally large HDF5 copy
    to the same filesystem the running job is writing to. That contention slows
    the model and corrupts exactly the timings a benchmark exists to produce.
    Prove the workflow on a throwaway run instead.

If a job died without calling `Close()`, the dataset's active flag is still set
and readers wait for steps that will never arrive. ADIOS2 ships
`adios_deactive_bp.sh` to clear it.

Two ready-made cases exercise this end to end:
`rundata/input_configs/default_cases/bp5_live_test.json` and its `_async` twin.
Both are a 32×32×33 bubble writing 101 output steps over a run long enough to
attach a converter mid-flight, with `overwrite: true` so they can be re-run
freely.

The BP5 reader also accepts `SelectSteps`, which converts a step range instead of
restarting from step 0. `bp2h5` hardcodes its reader parameters, so this needs the
underlying tool directly:

```bash
$ADIOS2/bin/adios2_reorganize in.bp out.h5 \
    BPFile "StreamReader=ON,SelectSteps=10:20" HDF5 ""
```

The value is a space-separated list of `start:end:step` expressions, indexed from
0, with **`end` inclusive**. Rules are unioned, and `n` as the end means
unbounded:

| Expression | Steps selected |
|---|---|
| `10:20` | 10 through 20 inclusive — 11 steps |
| `0 6 3 2` | 0, 2, 3, 6 — a bare number selects that single step |
| `2:n` | everything from step 2 onward |
| `0:n:2` | every other step from the beginning |
| `0:n:3 10:n:5` | every third step, plus every fifth from step 10 |

Separators are strict: `10-20` is rejected at open with `could not cast the
entire string '10-20' to a single integer number`.

`adios2_reorganize` prints `WARNING: steps ... were missed when advancing` for
every gap the filter creates. That is expected with `SelectSteps`, not a fault.

**Output steps are renumbered from 0.** The converter opens one output step per
surviving input step and passes no index, so `SelectSteps=10:20` produces
`Step0 … Step10`, not `Step10 … Step20`. Nothing is lost — `model_step` and
`model_time_s` are written into every step and travel with the data — but the
group name is not a model step number, and two separately converted ranges cannot
be told apart by group name alone. Read `model_step` to place a converted chunk.

`SelectSteps` and `StreamReader` do compose; the invocation above is verified.
Whether a range selection also picks up steps appended *after* open, on a dataset
a writer still holds open, has not been tested separately.

## Where the code lives

| Path | Role |
|---|---|
| `src/io/OutputManager.*` | HDF5 and SST writer, schema, field packing |
| `src/io/IOServer.*` | SST reader and HDF5 relay |
| `src/io/history/HistoryWriter.hpp` | Neutral `write`/`close` interface used by `main.cpp` |
| `src/io/history/GradsCtl.*` | GrADS descriptor shared by every engine |
| `src/io/bp5/` | Direct BP5 writer: config, schema, CPU staging, lifecycle |

For BP5 internals — architecture, dataset layout, the compatibility contract
against HDF5 output, buffer lifetime, and the validation matrix — see
[BP5 output internals](../developer-guides/bp5-output.md).
