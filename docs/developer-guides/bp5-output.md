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
| `src/io/bp5/Bp5HistoryWriter.*` | ADIOS2 lifecycle and step orchestration |
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

`vvm_real_precision` and `vvm_field_precision` differ exactly when
`output.bp5.precision` narrows or widens the history relative to `VVM::Real`.
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
| Invalid field selection | Fails with a clear error, as before |

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

A test generates HDF5 and BP5 output from the same deterministic run and
compares names, types, shapes, steps, attributes, and values through each
format's own reader.

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
- 1-, 2-, and 4-rank synchronous direct and packed output;
- 1-, 2-, and 4-rank asynchronous direct output;
  on CUDA, requested-direct variants intentionally exercise host packing;
- collective configuration mismatch handling;
- 1-rank read-back of data written by different writer counts;
- exact values and shapes for 1-D through 4-D fields, coordinates, clocks, and
  metadata over three ADIOS steps;
- ten computed model steps plus initial output, synchronous and asynchronous;
- bit-for-bit comparison against HDF5 model output for all eleven output steps;
- 1-, 2-, and 4-rank `float32` and `float64` output, asserting the field
  variables **are** the requested type and **are not** the other one, plus a
  converting `float32` run through the asynchronous path.

The full matrix passes against an unmodified ADIOS2 2.12.1. Precision tests
carry their own label:

```bash
ctest --test-dir build --output-on-failure -L precision
```

A production scheduler run is still needed to settle the final `num_subfiles`
value and to decide whether asynchronous mode improves total simulated-step
time.

## Current boundaries

- **History output only.** HDF5 remains the supported checkpoint/restart route,
  so `precision` can only ever narrow history — it cannot silently reduce
  restart fidelity. A separate checkpoint writer and a `Bp5RestartReader` are
  future work.
- **CUDA output is host-staged.** `effective_buffer_mode()` always returns
  `pack` in a CUDA build. The source creates a synchronized host mirror, then
  copies the selected cells into a persistent `Bp5BufferSet` allocation before
  calling ADIOS2. Its direct branch also throws under CUDA, enforcing the
  no-device-pointer invariant at the source boundary. The host mirror itself is
  currently allocated per field per step; cache it only after measurement shows
  that allocation is material.
- **No dataset rotation.** One run writes one dataset, however large it grows.
- **No offline BP5-to-HDF5 conversion.** Read BP5 directly; see
  [Output](../user-guides/output.md#reading-the-output).
- SST remains switchable because the external ADIOS2 build includes it. BP5 uses
  no SST data plane or transport setting.
