# BP5 output internals

!!! note "Looking for how to use it?"

    This page documents how the direct BP5 writer is built and what it
    guarantees. For engine selection, configuration keys, precision, submission,
    and dataset sizing, see [Output](../user-guides/output.md).

VVMex has an independent **CPU** history-output path that writes a multi-step
ADIOS2 BP5 dataset directly from the compute ranks. It uses no `IOServer`, no
SST transport, and no patched ADIOS2 — an unmodified upstream 2.12.1 install is
enough. The original HDF5 and SST writers are untouched and remain available.

It follows the ADIOS2 multi-step BP pattern: open one dataset, repeat
`BeginStep`/`Put`/`EndStep`, close once. One `*.bp` dataset therefore holds every
output time rather than one file per time, and BP5 manages its own metadata and
data subfiles inside it.

## Architecture

`main.cpp` selects a writer through a deliberately small interface, so the two
implementations share a call site and nothing else:

```text
output.engine = BP5
    -> Bp5HistoryWriter
    -> CPU field selections or persistent packed buffers
    -> ADIOS2 BP5 TwoLevelShm aggregation
    -> parallel filesystem

all other engines
    -> LegacyHistoryWriter
    -> original OutputManager / IOServer behaviour
```

```cpp
class HistoryWriter {
public:
    virtual ~HistoryWriter() = default;
    virtual void write(std::size_t step, VVM::Real time) = 0;
    virtual void close() = 0;
};
```

`close()` is explicit so a collective close happens before `Kokkos::finalize()`
and `MPI_Finalize()`. The destructor is a fallback, not the normal path.

### Source layout

| File | Responsibility |
|---|---|
| `src/io/history/HistoryWriter.hpp` | Neutral `write`/`close` interface |
| `src/io/history/LegacyHistoryWriter.hpp` | Adapter around the original `OutputManager` |
| `src/io/history/GradsCtl.*` | GrADS descriptor emitter shared with the legacy writer |
| `src/io/OutputPrecision.*` | Engine-neutral `output.precision` parsing and element-type resolution |
| `src/io/bp5/Bp5HistoryWriter.*` | ADIOS2 lifecycle and step orchestration |
| `src/io/bp5/Bp5RestartReader.*` | Restart from one step of a `.bp` dataset |
| `src/io/RestartVariables.*` | Which fields a restart recovers, shared with the HDF5 reader |
| `src/io/bp5/Bp5OutputConfig.*` | Parse and validate the `output.bp5` block |
| `src/io/bp5/Bp5FieldSchema.*` | Global shapes, per-rank selections, output bounds |
| `src/io/bp5/Bp5BufferSet.hpp` | Persistent per-field staging buffers, one map per element type |
| `src/io/bp5/CpuFieldSource.*` | Direct-selection and packing strategies |
| `src/io/bp5/FieldInput.hpp` | Staging descriptor: pointer, element count, memory space |
| `src/io/bp5/Bp5PathPolicy.hpp` | Dataset path resolution and overwrite safety |
| `src/io/bp5/Bp5CollectiveValidation.hpp` | Cross-rank agreement on resolved settings |

No BP5 source includes `IOServer.hpp` or any SST helper.

## Dataset layout

Each BP5 step contains:

- `time` (`VVM::Real`), `model_time_s` (`double`), `model_step` (`int64_t`);
- `coordinates/x`, `coordinates/y`, `coordinates/z_mid`;
- the configured 1-D through 4-D fields;
- per-field `units`, `long_name`, `standard_name`, `comment`, and
  `grid_staggering` attributes, when the field defines them.

Dataset-level attributes describe how to interpret the file:

| Attribute | Meaning |
|---|---|
| `vvm_schema_version` | Output schema version |
| `vvm_output_role` | `history` |
| `vvm_real_precision` | The model's working precision |
| `vvm_field_precision` | The on-disk field element type |
| `vvm_coordinate_order` | `z,y,x` |
| `vvm_global_grid_shape` | Global `{nz, ny, nx}` |
| `vvm_output_bounds_zyx` | Resolved inclusive output bounds |

`vvm_real_precision` and `vvm_field_precision` differ exactly when the resolved
precision — `output.precision`, or the deprecated `output.bp5.precision`
where a configuration still carries it — narrows or widens
the history relative to `VVM::Real`. The HDF5 writer records the same two
attributes for the same reason.
Matching values mean the file is a lossless copy of the model state.

Global variable shapes are independent of the MPI decomposition. Halo cells are
excluded, `output_grid` bounds are honoured, uneven decompositions are valid, and
ranks holding no selected cells still participate in the collective step calls
with an empty selection. Variable definitions are created before `Open`, and
`LockWriterDefinitions()` is called afterwards because the schema is fixed.

## Compatibility with HDF5 output

BP5 changes the container, not the meaning. The writer preserves the scientific
content of the original `OutputManager` output:

| Original output content | BP5 guarantee |
|---|---|
| Names from `output.fields_to_output` | Every configured field name preserved exactly |
| State values | Same values; same type when `precision` is `native` |
| `time` | Scalar `VVM::Real`, elapsed simulation seconds |
| `model_time_s` | Scalar `double`, elapsed simulation seconds |
| `model_step` | Exact scalar `int64_t` |
| `coordinates/x`, `/y`, `/z_mid` | Same names, values, and `meter` units |
| 1-D fields | Global Z shape and values |
| 2-D fields | Global `{y, x}` order and values |
| 3-D fields | Global `{z, y, x}` order and values |
| 4-D fields | Global `{component, z, y, x}` order and values |
| `units`, `long_name`, `standard_name`, `comment`, `grid_staggering` | Preserved when present |
| `output.output_grid` | Same inclusive subsetting semantics |
| Halo handling | Same physical cells written, halo excluded |
| A configured field the run never registered | Skipped with a message, as HDF5 and SST do |
| Invalid field selection | Fails with a clear error, as before |
| `<output_dir>/vvm.ctl` | Same descriptor, retargeted at the `.bp` dataset |

Scalar metadata is likewise unchanged:

```text
time:         units="s", long_name="elapsed simulation time"
model_time_s: units="s", long_name="elapsed simulation time"
model_step:   units="1", long_name="integration step count"
```

This is **semantic** equivalence, not binary equivalence:

- HDF5 places one output time under a `Step0` group; BP5 uses native steps 0, 1,
  2, … in a single dataset.
- HDF5 dimension-scale objects are format-specific and have no BP5 equivalent.
  Their meaning is carried by the coordinate variables and attributes above.
- BP5 adds the schema and provenance attributes listed earlier, but renames and
  reinterprets nothing.
- Reduced precision is the one intentional, opt-in departure. It is never a
  default and is always recorded in `vvm_field_precision`.

All three engines are held to this. `Compare_engine_outputs_model_smoke` runs
the same deterministic case three times — HDF5, BP5, and SST (two compute ranks
each, plus one I/O rank for SST) — and compares names, types, shapes, steps,
attributes, and values through each format's own reader. Values and metadata
both: the SST relay is the path where every number can be right while an
attribute quietly goes missing, which is exactly what happened before it copied
attributes wholesale.

## GrADS descriptor

Both writers build their `vvm.ctl` through `src/io/history/GradsCtl.*`, so the
axis maths, time formatting, and variable-name rules have one implementation.
Only the header lines and the variable records differ:

| | HDF5 / SST | BP5 |
|---|---|---|
| `DSET` | `^<prefix>_%tm6.h5` | `^<prefix>.bp` |
| `DTYPE` | `hdf5_grid` | `bp5` |
| `OPTIONS template` | yes, one file per step | no, one multi-step dataset |
| Variable record | `/Step0/<name>=><grads>` | `<name>=><grads>` |
| Description | field name | `long_name (units)` |

The BP5 records are shaped by what the GrADS BP5 reader validates on `open`.
That reader lives in
[Aaron-Hsieh-0129/opengrads-update](https://github.com/Aaron-Hsieh-0129/opengrads-update)
(`cola/src/gaadios.c`), an OpenGrADS fork with BP5 support added; stock GrADS
cannot open a `dtype bp5` descriptor at all. Its rules:

- Names are lowercased and truncated to 15 characters, and must start with a
  letter, so `unique_grads_variable_name` sanitises each name and de-duplicates
  the truncations. Untouched names are written without an alias.
- A variable must map exactly one x and one y dimension. A z-only profile field
  cannot be expressed, and declaring one fails the whole `open` rather than that
  variable, so profiles are omitted and named in a `*` comment line.
- A 4-D field is declared as `0,z,y,x`, pinning the component axis to a fixed
  index, because the descriptor's dimension count must equal the variable's
  rank.
- Variables keep the full global shape even under `output.output_grid`, so the
  axes always describe the global grid.
- `TDEF` must not exceed the number of steps in the dataset, so it counts the
  steps the run will write, `output.output_initial_step` included.

## Restart

`Bp5RestartReader` (`src/io/bp5/Bp5RestartReader.*`) is the read side, chosen by
`Initializer` when `restart.source_file` ends in `.bp`. It differs from the HDF5
reader in three places and matches it everywhere else:

| | HDF5 reader | BP5 reader |
|---|---|---|
| Which time | the file is one time | `restart.step_index`, default `-1` = last step |
| Open mode | `H5Fopen` | `Mode::ReadRandomAccess`, so any step is reachable |
| Stored type | HDF5 converts on read | the reader dispatches on `VariableType` and converts |

Field selection is literally shared code (`src/io/RestartVariables.*`), extracted
from the HDF5 reader so a run restarted from BP5 loads exactly what the same run
restarted from HDF5 would. Each rank reads its own slab of the global array, so
the rank count at restart is independent of the rank count that wrote the
dataset. Halo exchange after loading uses the per-field overload, for the same
CUDA-graph reason the HDF5 reader documents.

One guard is specific to BP5: because a `.bp` dataset is a directory the writer
may be asked to overwrite, `Bp5HistoryWriter` refuses to start when its target
resolves to the same path as `restart.source_file`. Without it, a resumed run
with `overwrite: true` would delete the history it had just read.

## CPU staging

`CpuFieldSource` implements two equivalent routes to the same bytes, selected by
`output.bp5.buffer_mode`:

**Direct** — pass the model's own ghosted `LayoutRight` allocation to ADIOS2 and
describe the halo-free region with `SetMemorySelection`. No application-side
pack, no second full buffer. Requires the on-disk element type to be `VVM::Real`.

**Pack** — copy the selected physical cells into a persistent contiguous host
buffer, allocated once during construction rather than per step.

`Bp5HistoryWriter` resolves these once in its constructor:

```cpp
element_type_          = config_.element_type();          // native -> concrete type
effective_buffer_mode_ = config_.effective_buffer_mode(); // direct -> pack if converting
```

A precision conversion cannot hand ADIOS2 the model's memory, so a converting
configuration resolves `direct` to `pack` and reports it. The conversion itself
is a `static_cast` inside the existing pack loop, so it costs no copy beyond the
pack that `pack` mode already performs.

Field variables are held as `std::variant<Variable<float>, Variable<double>>`;
`FieldInput::data` is `const void*` and the writer casts it back against the
resolved element type. Coordinates and clocks are deliberately excluded from the
precision switch.

### Put mode and buffer lifetime

| `async_write` | Put mode | Why |
|---|---|---|
| `false` | `Mode::Deferred` | Source memory stays valid through `EndStep()`; no copy |
| `true` | `Mode::Sync` | ADIOS2 copies before `write()` returns, so the model may modify its fields immediately |

The `Sync` choice under `async_write` is the safety property that makes async
usable at all: with a deferred put, ADIOS2's background thread would still be
reading the live field while the model advanced it. Scalars are always `Sync`
because they point at stack locals.

`AggregationType` is restricted to `TwoLevelShm`, which is what lets the async
path work under plain `MPI_Init` — its background thread does shared-memory
reads and POSIX writes, never MPI.

## Validation

```bash
# GPU
cmake --preset <gpu-preset> -DBUILD_TESTS=ON -DVVM_TEST_BP5=ON
ctest --test-dir build --output-on-failure -L bp5

# CPU
cmake --preset <cpu-preset> -DBUILD_TESTS=ON -DVVM_TEST_BP5=ON
ctest --test-dir build_cpu --output-on-failure -L bp5
```

The matrix covers:

- configuration, schema, path-policy, and field-source packing unit tests;
- the GrADS descriptor emitter (`ctest -R test_grads_ctl`, part of the default
  unit tier rather than the BP5 tier);
- precision parsing, that BP5 takes the engine-neutral `output.precision`, and
  that the deprecated `output.bp5.precision` still overrides it
  (`ctest -R test_output_precision`), plus the same option applied to the HDF5
  writer and the SST relay (`ctest -L precision`);
- 1-, 2-, and 4-rank synchronous direct output — the rank axis, where 1 rank is
  undecomposed, 2 ranks split the domain, and 4 ranks are the only case that
  leaves some ranks with an empty output selection;
- 2- and 4-rank synchronous packed output (4 ranks because staging an empty
  selection is the one place the two buffer modes can differ);
- 2-rank asynchronous direct output;
  on CUDA, requested-direct variants intentionally exercise host packing;
- collective configuration mismatch handling;
- 1-rank read-back of data written by different writer counts;
- exact values and shapes for 1-D through 4-D fields, coordinates, clocks, and
  metadata over three ADIOS steps;
- ten computed model steps plus initial output, synchronous and asynchronous;
- restart from a `.bp` step: a resumed run that takes no further steps writes
  back exactly the fields and clock of the step it resumed from;
- bit-for-bit comparison of HDF5, SST, and BP5 model output, values and
  attributes, for all eleven output steps;
- 2-rank `float32` and `float64` output, asserting the field variables **are**
  the requested type and **are not** the other one, plus a converting `float32`
  run through the asynchronous path.

Buffer mode, async and precision are all per-rank-local choices, so they are
pinned at 2 ranks rather than swept: repeating them at 1 and 4 ranks re-tested
the same slab arithmetic and cost 4 more GPUs.

The full matrix passes against an unmodified ADIOS2 2.12.1. Precision tests
carry their own label:

```bash
ctest --test-dir build --output-on-failure -L precision
```

A production scheduler run is still needed to settle the final `num_subfiles`
value and to decide whether asynchronous mode improves total simulated-step
time.

## Current boundaries

- **No separate checkpoint writer.** As with HDF5, the history dataset *is* the
  restart source; there is no second, denser checkpoint stream. Narrowing the
  history with `precision` therefore narrows what a restart recovers, on every
  engine — see [Output](../user-guides/output.md#output-precision).
- **CUDA output is host-staged.** `effective_buffer_mode()` always returns
  `pack` in a CUDA build. The source creates a synchronized host mirror, then
  copies the selected cells into a persistent `Bp5BufferSet` allocation before
  calling ADIOS2. Its direct branch also throws under CUDA, enforcing the
  no-device-pointer invariant at the source boundary. The host mirror itself is
  currently allocated per field per step; cache it only after measurement shows
  that allocation is material.
- **No dataset rotation.** One run writes one dataset, however large it grows.
- **No VVMex-native BP5-to-HDF5 conversion,** and none is planned: ADIOS2's own
  `bp2h5` covers it, including against a dataset a running job still has open.
  It converts what has been flushed so far and exits; it does not follow the
  writer, and it has no incremental mode. See
  [Converting to HDF5](../user-guides/output.md#converting-to-hdf5).
- SST remains switchable because the external ADIOS2 build includes it. BP5 uses
  no SST data plane or transport setting.
