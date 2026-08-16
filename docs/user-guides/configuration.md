# Model configuration

Runtime settings are loaded from a **single JSON file**. The recommended `submit.py` workflow passes this path to the executable for you. If you run the executable directly, pass the JSON path as the first non-option argument (see [Command-line options](#command-line-options)).

This page is the complete key reference: every key the model actually reads, with its type, its value when omitted, and what it does. The repository ships runnable sample configurations under `rundata/input_configs/default_cases/`. Each default-case JSON references a matching profile under `rundata/initial_conditions/profiles/default_cases/` and, when needed, a spatial NetCDF file under `rundata/initial_conditions/spatial/default_cases/`.

## How to read this page

| Default column | Meaning |
| --- | --- |
| *required* | The key must be present. The run stops at startup with `Configuration error: Key '<path>' not found.` |
| a value | The key is optional and this is what the code uses when it is absent. |

Three things about how the file is parsed are worth knowing before you edit one:

- **Keys are looked up by dotted path, not validated as a schema.** A key the model does not know is simply never read.
- **A misspelled optional key is therefore silent.** `"enable_p3"` written as `"enable_P3"` leaves P3 off and prints no warning. When a setting appears to have no effect, check the spelling and nesting against this page first.
- **The one exception is `output.bp5`,** which rejects unknown keys and invalid values outright, and cross-checks every resolved setting across all compute ranks before opening the dataset.
- **Keys starting with `__` or `_` are inline comments** for humans and are never read.

Types are `int`, `real` (double, or float in a single-precision build), `bool`, `string`, `list`, or `object`.

## Command-line options

| Argument | Meaning |
| -------- | ------- |
| `path/to/config.json` | Optional. If present as the first non-flag argument, selects the configuration file. Flags such as `--io-tasks` are skipped when resolving this path. |
| `--io-tasks N` | Reserve **N** MPI ranks for asynchronous I/O. Only meaningful with `output.engine = "SST"`; see [Output](output.md). |

Example:

```bash
cd $VVM_ROOT
./submit.py -c /path/to/my_run.json --compute 1 --local

# Advanced direct MPI only
mpirun -np 1 ./build/vvm /path/to/my_run.json
```

The full `submit.py` option list is in [Job submission](job-submission.md#full-option-reference).

## How to edit a run

Most experiments follow the same editing order:

1. Set the grid and time controls (`grid`, `simulation`).
2. Choose the sounding profile and spatial NetCDF inputs (`initial_conditions`, `netcdf_reader`).
3. Choose restart behavior, if any (`restart`).
4. Choose output engine, fields, and subdomain (`output`).
5. Enable dynamics forcings and tendency terms (`dynamics`).
6. Enable physics packages and their call frequencies (`physics`).
7. Tune acceleration options (`optimization`) only after the run is physically configured.

Keep JSON paths relative to the directory where `submit.py` starts the model, normally the project root.

## Default-case inputs

Use the default cases when you want a known sample setup before designing your own experiment:

```bash
./submit.py --local --preset <your_preset_name> -c ./rundata/input_configs/default_cases/advection_u.json --compute 1
```

The three default-case directories are:

| Directory | Purpose |
| --------- | ------- |
| `rundata/input_configs/default_cases/` | Runnable JSON files such as `advection_u.json`, `2dbubble.json`, `rcemip.json`, `sea_grass_mountain.json`, and `taiwanvvm_2048.json`. |
| `rundata/initial_conditions/profiles/default_cases/` | Sounding/profile text files used by those JSON files. |
| `rundata/initial_conditions/spatial/default_cases/` | Spatial NetCDF inputs for topography, surface, and land fields. These can be generated with `tools/generate_init_nc.py`. |

For the complete case list and generation notes, see [Default cases](../examples/default-cases.md).

## `grid`

Global mesh, halo width, horizontal boundary behavior, and vertical coordinate construction.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `nx`, `ny`, `nz` | int | *required* | Global domain size in x, y, and z. MPI decomposes this domain across compute ranks. |
| `n_halo_cells` | int | *required* | Halo width used by stencil operations and MPI halo exchange. `2` suits every scheme except `weno5`, whose flux stencil reaches three cells past the last interior cell. This is one grid-wide setting, so **a single tracer using `weno5` forces `3` for the whole run**, whatever the other tracers use. A run that gets this wrong stops at startup with `Tracer '<name>': weno5 halo width is insufficient; configured 2, required 3.` |
| `dx`, `dy` | real | *required* | Horizontal grid spacing in meters. Uniform. |
| `dz` | real | *required* | Nominal vertical spacing in meters. |
| `dz1` | real | *required* | Spacing of the stretched lower layers. With `dz1 < dz` the vertical grid is stretched from `dz1` near the surface toward `dz` aloft; `dz1 == dz` gives a uniform column. |
| `boundary_condition.x` | string | `"periodic"` | Lateral boundary type in x: `periodic` or `zero_gradient`. Anything other than `zero_gradient` is treated as `periodic`, so a typo silently gives you a periodic boundary. |
| `boundary_condition.y` | string | `"periodic"` | Same, in y. |
| `fix_lonlat` | bool | `false` | Fixed longitude/latitude handling for Taiwan-oriented real-case inputs. |
| `vertical_coordinate_type` | string | `"default"` | `default`, `taiwanvvm`, or `rcemip`. Selects how `z_up`/`z_mid` are built, how the profile is interpolated, and the output coordinate convention. |
| `rcemip_grid_data_path` | string | `./rundata/initial_conditions/profiles/snd_rcemip_anal300_v3.txt` | Profile the `rcemip` coordinate is built from. Read only when `vertical_coordinate_type` is `rcemip`. |

Use `taiwanvvm` when you need the TaiwanVVM-style vertical coordinate and output coordinate handling. Use `default` for ordinary idealized or simple real-case tests unless the case explicitly requires another coordinate.

## `simulation`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `total_time_s` | real | *required* | Total simulated time in seconds. |
| `dt_s` | real | *required* | Model time step in seconds. Several physics frequencies must be divisible by this value. |
| `output_interval_s` | real | *required* | Output cadence in simulated seconds. With `dt_s: 1.0` and `output_interval_s: 600.0` you get one output every 600 model steps. |
| `idealized_test` | string | `"none"` | Built-in dynamics test: `none`, `advection_u`, `advection_v`, `advection_w`, `stretching`, `twisting`, or `2dbubble`. |

Setting `idealized_test` to anything but `none` replaces normal initialization with the test's analytic state and disables the spatial NetCDF read. For production-like runs, keep it `none`.

## `initial_conditions`

The one-dimensional sounding/profile and optional initial perturbations.

A run with `simulation.idealized_test: "none"` — that is, any real case — **must** supply `initial_conditions.format` (which must be `txt`), `initial_conditions.source_file`, and `netcdf_reader.source_file`; the run stops at startup otherwise. Built-in idealized tests build their own state and may omit all three.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `format` | string | *required unless idealized* | `txt` or `netcdf`; any other value is rejected. `txt` reads a sounding through `TxtReader`. |
| `source_file` | string | *required unless idealized* | Path to the profile used to initialize pressure, thermodynamics, and winds. Required whenever `format` is `txt`. |
| `perturbation` | string | `"none"` | Initial perturbation preset: `none`, `2dbubble`, or `3dbubble`. Skipped entirely when `restart.enable` is true. |
| `constant_upper_wind.enable` | bool | `false` | Hold winds above `pressure_threshold_Pa` constant while reading the text profile. |
| `constant_upper_wind.pressure_threshold_Pa` | real | `25000.0` | Pressure threshold for that handling. |
| `reapply_spatial_initial_conditions` | bool | `false` | Re-read `netcdf_reader.source_file` after the base state is assigned, so spatial fields overwrite anything the profile/topography path derived. Applied before restart loading. |

!!! note "`pressure_threshold_Pa` has two different fallbacks"

    The profile reader falls back to `25000.0` Pa, but the area-mean nudging
    forcing (`dynamics.forcings.areamn`) falls back to `3000.0` Pa for the same
    key. If you enable both `constant_upper_wind` and `areamn`, set the key
    explicitly so the two agree.

Default-case profiles live under `rundata/initial_conditions/profiles/default_cases/`. The profile must be consistent with the vertical coordinate choice and the physics you enable.

## `netcdf_reader`

Spatial two-dimensional fields such as longitude, latitude, topography, land mask, vegetation, soil, and surface parameters.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `source_file` | string | *required unless idealized* | Spatial NetCDF file, usually under `rundata/initial_conditions/spatial/`. When absent, no spatial read happens at all. |
| `Tg_source` | string | `"atmosphere"` | `atmosphere` derives ground temperature from the atmospheric state at the topography level; `netcdf` keeps the `Tg` field read from the file. Any other value keeps existing values and warns. |
| `variables_to_read.1d` | list | *(empty)* | 1-D variables to read from the file. |
| `variables_to_read.2d` | list | *(empty)* | 2-D variables to read. Typically `lon`, `lat`, `topo`, `sea_land_ice_mask`, `vegtype`, `soiltype`, `slopetype`, `Tg`, `albedo`, `gvf`, `lai`, `shdmax`, `shdmin`. |
| `variables_to_read.3d` | list | *(empty)* | 3-D variables to read. Enabled tracers and their `_source` fields are always appended to this list, whether or not it is given. |

Every listed variable must exist in the file — a missing one is an error, not a warning. The default spatial NetCDF files can be generated with `tools/generate_init_nc.py`. For Taiwan-style cases, generate or prepare the file before submission; see [TaiwanVVM example](../examples/taiwan-vvm.md).

## `restart`

Whether the model initializes prognostic fields from an existing output file.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable` | bool | `false` | Enables restart loading. |
| `source_file` | string | *required when enabled* | Restart source. `.h5` goes through the HDF5 restart reader, `.bp` through the BP5 restart reader, `.nc` through the PnetCDF reader; any other extension is an error. |
| `legacy_time_s` | real | *(none)* | Elapsed simulation seconds to resume from, for a file that stores no clock of its own. Explicit opt-in; warns when used. |
| `allow_filename_time_fallback` | bool | `false` | Restores the old behaviour of deriving the restart time from the digits in the file name (`index * file_interval_s`). Warns loudly when enabled. |
| `file_interval_s` | real | `3600.0` | Seconds represented by one numbered output index. Consulted only when `allow_filename_time_fallback` is true. |
| `ignore_stored_step` | bool | `false` | Discard the `model_step` stored in the file and re-derive the step from `simulation.dt_s`. Use when `dt` was deliberately changed on restart. |
| `step_index` | int | `-1` | **BP5 only.** Which ADIOS2 step of the `.bp` dataset to resume from; `-1` is the last one written. An `.h5` file holds a single time, so this key has no meaning there. |
| `variables_to_read.1d` | list | *(none)* | Explicit 1-D restart variable list. |
| `variables_to_read.2d` | list | *(none)* | Explicit 2-D restart variable list. |
| `variables_to_read.3d` | list | inferred | Explicit 3-D restart variable list. See below for how it is inferred. |

If no explicit `restart.variables_to_read.3d` is supplied, the reader (HDF5 or BP5 — they share this selection) selects prognostic variables from `dynamics.prognostic_variables` and filters them through `output.fields_to_output`. That means fields needed after restart should either be listed explicitly under `restart.variables_to_read` or be included in `output.fields_to_output`.

When restart is enabled, the restart state replaces the normal perturbation initialization.

### How the restart time and step are recovered

The simulation clock comes from metadata stored **inside** the restart file, never from its name. Renaming a restart file does not change the time it resumes from. Output files carry two scalars for this:

| Variable | Meaning |
| --- | ---- |
| `model_time_s` | Elapsed simulation time in seconds (`double`). |
| `model_step` | Exact integration-step count (64-bit integer). |

The model tries, in order:

1. Stored time and stored step. They are cross-checked against `simulation.dt_s`; if `time` and `step * dt` disagree by more than a serialization tolerance the run stops and prints both values, the configured `dt`, the expected time and the source file. Neither value is silently preferred.
2. Stored time alone, with `step = round(time / dt)`.
3. `restart.legacy_time_s`, for files predating the metadata. Prints a rank-0 warning.
4. The digits in the file name, only when `restart.allow_filename_time_fallback` is true. Prints a prominent rank-0 warning, because the file name then *is* the clock.

Without one of these the run stops rather than guessing. The recovered values are broadcast from rank 0, so every rank resumes from an identical time and step, and rank 0 reports them:

```text
[Initializer] Restart metadata: time=7200 s, step=3600 (source: file model_time_s + model_step)
```

NetCDF restart sources are written outside this model, so only unambiguous metadata is accepted: `model_time_s` / `model_step` as scalar variables or as global attributes, or a `time` variable whose `units` attribute says plain seconds. A calendar `time` ("hours since ...") is deliberately *not* read as an elapsed time — such a file needs `restart.legacy_time_s`.

Restart files are ordinary output files; there is no separate restart output path. What each engine leaves behind to restart from:

| `output.engine` | Restart source it produces | Read by |
| --- | --- | --- |
| `HDF5` | `<prefix>_NNNNNN.h5`, one per output time | `Hdf5RestartReader` |
| `SST` | the same `.h5` files, written by the I/O server rather than the compute ranks | `Hdf5RestartReader` — SST is a transport, not a restart format |
| `BP5` | `<prefix>.bp`, one dataset holding every output time | `Bp5RestartReader`, which also needs `step_index` |

The keys above are the same for all three; only `step_index` is engine-specific, because only BP5 stores more than one time per source. Field selection and clock recovery are shared code, so the same run resumes identically whichever of the three wrote its source.

### Worked examples

Resuming an HDF5 (or SST) run from the file written at t = 3600 s:

```json
"restart": {
  "enable": true,
  "source_file": "./output/my_case/vvm_output_000006.h5"
},
"simulation": {
  "total_time_s": 7200.0,
  "dt_s": 1.0,
  "output_interval_s": 600.0
},
"output": {
  "engine": "HDF5",
  "output_dir": "./output/my_case_resumed",
  "output_filename_prefix": "vvm_output",
  "fields_to_output": ["thbar", "rhobar", "topo", "u", "v", "w", "th", "qv", "xi", "eta", "zeta"]
}
```

The same run resuming from a BP5 dataset instead:

```json
"restart": {
  "enable": true,
  "source_file": "./output/my_case/vvm_output.bp",
  "step_index": -1
},
"simulation": {
  "total_time_s": 7200.0,
  "dt_s": 1.0,
  "output_interval_s": 600.0
},
"output": {
  "engine": "BP5",
  "output_dir": "./output/my_case_resumed",
  "output_filename_prefix": "vvm_output",
  "fields_to_output": ["thbar", "rhobar", "topo", "u", "v", "w", "th", "qv", "xi", "eta", "zeta"],
  "bp5": { "num_subfiles": 2, "overwrite": false }
}
```

Four things decide whether these work:

- **`total_time_s` is absolute, not additional.** The clock resumes at the time stored in the source, so `7200.0` means "run until t = 7200 s", which is one more hour after resuming at 3600 s. Setting it to the restart time produces a run that takes no steps and writes only the loaded state.
- **`fields_to_output` has to contain what the restart needs.** The inferred variable list is filtered through it, so a field that was never written cannot be recovered. List it explicitly under `restart.variables_to_read` only when you want a set other than the inferred one.
- **Write somewhere else.** Both examples resume into `my_case_resumed`. Writing back into `my_case` re-emits the same numbered HDF5 files, and on BP5 the writer refuses outright rather than deleting the dataset it just read.
- **The first step after any restart is first order.** The AB2 tendency history is not stored, so a resumed run is not bit-for-bit with an uninterrupted one. The model warns about this on every restart.

Output resumes on the same index grid: the example above writes `vvm_output_000006` (the restart state, from `output_initial_step`) and then `000007` onward.

## `output`

ADIOS2 output, field selection, and optional subsetting. Engine trade-offs and sizing guidance are in [Output](output.md).

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `output_dir` | string | *required* | Destination directory. `submit.py` creates it before launching. |
| `output_filename_prefix` | string | *required* | Base name for the files, the `.bp` dataset, or the SST stream. |
| `engine` | string | `"HDF5"` | `HDF5` (one file per output time), `SST` (streams to dedicated I/O ranks), or `BP5` (one multi-step `.bp` dataset written directly by compute ranks). |
| `fields_to_output` | list | *required* | Ordered list of state fields to write. A name the run never registered is skipped with a message; on HDF5 and SST a name that is not a known optional field either is an error, so typos are still caught. Must be non-empty and duplicate-free for BP5. |
| `precision` | string | `"native"` | On-disk float type for **field data only**, on every engine: `native` (follows `VVM::Real`), `float32`/`float`/`single`, or `float64`/`double`. Case-insensitive. Clocks and coordinates always stay `VVM::Real`. Narrowing output narrows what a later restart recovers, on whichever engine produced the restart source. |
| `output_initial_step` | bool | `true` | Write step 0 before the first model step. |
| `output_grid.x_start`, `.y_start`, `.z_start` | int | `0` | Inclusive lower index bound per direction. |
| `output_grid.x_end`, `.y_end`, `.z_end` | int | `-1` | Inclusive upper index bound; `-1` means "to the end". Halo cells are never written. |

### `output` — HDF5 only

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `hdf5_collective_mpio` | bool | `false` | Use collective MPI-IO in the HDF5 engine instead of independent writes. |

### `output` — SST only

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `queue_limit` | int | `1` | ADIOS2 SST queue depth. The queue-full policy is `Block`, so a small value keeps the writers close to the readers. |
| `data_transport` | string | `"WAN"` | SST data plane: `WAN` (sockets), `RDMA`, or empty/`AUTO` to let ADIOS2 choose. Case-insensitive. |
| `control_transport` | string | `"sockets"` | SST control plane. |

### `output.bp5`

Read only when `engine` is `BP5`. Unlike the rest of the file, **unknown keys and invalid values here are errors**, and every resolved setting is compared across all compute ranks before the dataset is opened.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `aggregation_type` | string | `"TwoLevelShm"` | Currently the only accepted value. |
| `num_subfiles` | int | `10` | Data subfile count; must be positive. Start with the number of nodes. |
| `stats_level` | int | `0` | `0` minimises statistics work; `1` enables BP5 statistics. No other value is accepted. |
| `async_write` | bool | `false` | Let ADIOS2 write to disk in the background. VVMex switches to `Mode::Sync` puts so the model may modify its fields immediately. |
| `buffer_mode` | string | `"direct"` | `direct` hands compatible CPU memory to ADIOS2 with a memory selection; `pack` (or `packed`) stages into persistent contiguous buffers. Resolves to `pack` automatically under CUDA or whenever precision converts. |
| `precision` | string | `output.precision` | BP5-only override of the engine-neutral `output.precision`, with the same values. Omit it to follow that key. |
| `overwrite` | bool | `false` | When `false`, refuse to replace an existing dataset in `output_dir`. |

Use `fields_to_output` deliberately. A large list is convenient for diagnostics but increases file size and I/O cost. Common field groups are:

| Field group | Examples |
| --- | --- |
| Base state | `thbar`, `pibar`, `rhobar`, `rhobar_up` |
| Dynamics | `u`, `v`, `w`, `th`, `xi`, `eta`, `zeta` |
| Moisture/P3 | `qv`, `qc`, `qr`, `qi`, `qm`, `nc`, `nr`, `ni`, `bm` |
| Radiation | `sw_heating`, `lw_heating`, `swdn`, `lwdn`, `lwup`, `swup_toa`, `swdn_toa`, `lwup_toa`, `lwdn_toa`, `swup_sfc`, `swdn_sfc`, `lwup_sfc`, `lwdn_sfc` |
| Surface/land | `Tg`, `sfc_flux_th`, `sfc_flux_qv`, `sfc_flux_u`, `sfc_flux_v`, `le`, `hfx`, `st1`, `st2`, `st3`, `st4`, `gfx`, `topo` |

## `dynamics`

### `dynamics.solver`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `w_solver_method` | string | *required* | Vertical velocity solver: `tridiagonal` for the original method, `jacobi` for the 3D-parallel iteration path. |
| `iteration` | int | *required* | Fixed iteration count for the iterative solver path. |
| `WRXMU` | real | *required* | Relaxation/control parameter used by the wind solver. |

### `dynamics.forcings.sponge_layer`

Damps selected fields above `sponge_layer_base`.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable` | bool | `false` | Turns sponge-layer forcing on. |
| `damp_thermo` | bool | `true` | Damp thermodynamic variables. |
| `damp_vort` | bool | `true` | Damp vorticity variables. |
| `sponge_layer_base` | real | `-1` | Height in meters where damping begins. |
| `inv_CRAD` | real | `-1.0` | Inverse damping timescale used to build the damping coefficient. |

### `dynamics.forcings.random_perturbation`

Useful for triggering convection in otherwise smooth initial states.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable` | bool | `false` | Turns random forcing on. |
| `time_s` | real | `50.0` | Apply perturbations until this simulated time. |
| `amplitude` | real | `1.0` | Perturbation magnitude. |
| `z_start_m`, `z_end_m` | real | `0` | Vertical layer where perturbations are applied. |
| `random_seed` | int | `12345` | Seed, for repeatable perturbations. |

### `dynamics.forcings.lateral_boundary_nudging`

Relaxes selected variables toward large-scale forcing near selected boundaries.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable` | bool | `false` | Turns nudging on. |
| `boundaries.west`, `.east`, `.south`, `.north` | bool | `false` | Which boundaries are nudged. |
| `tau_b` | real | `300.0` | Nudging timescale in seconds. |
| `offset` | real | `2500.0` | Distance from the boundary at which the nudging zone starts, in meters. |
| `width` | real | `600.0` | Width of the nudging zone, in meters. |
| `radius` | real | `2500.0` | Corner-taper radius, in meters. |
| `target_vars` | list | `["th", "qv"]` | Variables to nudge. |
| `forcing_data.time_varying` | bool | `false` | Select forcing files by time using `file_prefix` and `update_interval_s` instead of using one fixed file. |
| `forcing_data.directory` | string | `"../rundata/LS_forcings/"` | Directory containing large-scale forcing files. |
| `forcing_data.file_name_for_not_varying` | string | `"ls_forcing_constant.nc"` | File used when forcing is constant in time. |
| `forcing_data.file_prefix` | string | `"ls_forcing_"` | Prefix for time-varying forcing files. |
| `forcing_data.update_interval_s` | real | `3600.0` | Time spacing between time-varying forcing files. |

### `dynamics.forcings.areamn`

Stores and relaxes area-mean vorticity/wind reference quantities.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable` | bool | `false` | Turns area-mean nudging on. |
| `uvtau` | real | `0.0` | Relaxation timescale/control value. |
| `nudge_start_m` | real | `0.0` | Height in meters where nudging starts. |
| `target_source` | string | `"initial"` | Where the nudging target comes from: `initial` holds the initial-state area means, `netcdf` reads them from the forcing files. Any other value is an error. |
| `forcing_data.directory` | string | *required when `target_source` is `netcdf`* | Directory containing the forcing files. |
| `forcing_data.file_prefix` | string | `"ls_forcing_"` | Prefix for the forcing files. |
| `forcing_data.update_interval_s` | real | `3600.0` | Time spacing between forcing files. Must be positive. |

### `dynamics.prognostic_variables`

An object whose keys are field names. Each names the tendency terms that advance it and the schemes each term uses.

```json
"th": {
  "tendency_terms": {
    "advection": {
      "enable": true,
      "temporal_scheme": "AdamsBashforth2",
      "spatial_scheme": "Takacs"
    }
  }
}
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `<var>.tendency_terms` | object | *(none)* | Terms to apply. Term names: `advection`, `buoyancy`, `stretching`, `twisting`, `coriolis`. An unknown name is an error. |
| `<var>.tendency_terms.<term>.enable` | bool | `true` | Skip the term when false. A disabled term is reported as `[Disabled]` at startup. |
| `<var>.tendency_terms.<term>.spatial_scheme` | string | *required* | `Takacs`, `MUSCL`, or `weno5`. |
| `<var>.tendency_terms.<term>.temporal_scheme` | string | `"AdamsBashforth2"` | `AdamsBashforth2`, `ForwardEuler`, or `SSPRK2`. |
| `<var>.tendency_terms.<term>.scheme_options` | object | *(none)* | Scheme-specific options; see below. |

Fields listed here are allocated whether or not they exist by default, so a typo creates a new, unused field rather than failing. Common prognostic variables are `th`, `xi`, `eta`, `zeta`, `qv`, `qc`, `qr`, `qi`, `qm`, `nc`, `nr`, `ni`, and `bm`. P3-related fields should be present when P3 is enabled.

Coriolis is a special case: it is enabled only when **all three** of `xi`, `eta`, and `zeta` set `tendency_terms.coriolis.enable` to true. Enabling it on one or two of them leaves it off.

#### Scheme pairing rules

Rejected combinations stop the run at startup rather than silently degrading:

| Spatial scheme | Requires | Allowed on |
| --- | --- | --- |
| `Takacs` | — | any prognostic field and term |
| `MUSCL` | `temporal_scheme: SSPRK2` | advection of a thermodynamic scalar or configured tracer |
| `weno5` | `temporal_scheme: SSPRK2`, `grid.n_halo_cells >= 3`, and advection as the field's only enabled term | advection of a configured passive tracer |

`SSPRK2` is only available with `MUSCL` or `weno5`.

#### `scheme_options` for MUSCL

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `limiter` | string | `"vanLeer"` | The only supported limiter. |
| `lower_bound` | real | `0.0` | Nonnegative floor applied by the limiter. |
| `max_cfl` | real | `0.9` | CFL cap; must be greater than zero. |

#### `scheme_options` for WENO5

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `epsilon` | real | `1.0e-6` | Nonlinear-weight epsilon; must be finite and positive. No other key is accepted here. |

```json
"tracer1": {
  "enable": true,
  "tendency_terms": {
    "advection": {
      "enable": true,
      "temporal_scheme": "SSPRK2",
      "spatial_scheme": "weno5",
      "scheme_options": { "epsilon": 1.0e-6 }
    }
  }
}
```

The WENO5 coefficients assume uniform spacing, so only the uniform horizontal x and y directions use WENO reconstruction. Vertical tracer transport retains the existing Takacs scheme, including its stretched-grid metric and boundary handling. The default epsilon is the reference value used for regression testing; because it is dimensional, simulations with very different tracer magnitudes or floating-point precision may need to configure it. WENO5 is not available for vorticity or other dynamical-core advection.

### `dynamics.tracers`

An object of passive tracers, each allocated as a 3-D field and advanced by the advection configuration given under the same key.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `<name>.enable` | bool | `true` | Skip the tracer entirely when false. |
| `<name>.source.enable` | bool | `true` when `source` is present | Allocate a companion `<name>_source` tendency field, read from the spatial NetCDF file alongside the tracer itself. |
| `<name>.tendency_terms` | object | *(none)* | Same shape as a prognostic variable, but **advection is the only supported term**. |

Tracer names are validated against every field VVMex allocates and every name reserved by the initial-condition and output formats, and against internal suffixes (`d_*`, `fe_tendency_*`, `*_m`, `*_ls`). A collision is an error at startup, not a silent overwrite.

## `physics`

### `physics.p3`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable_p3` | bool | `false` | Enables P3 microphysics. |
| `make_lookup_table` | bool | `false` | Generate the P3 lookup table instead of reading the shipped one. |
| `do_predict_nc` | bool | `true` | Predict cloud droplet number concentration. |
| `do_prescribed_ccn` | bool | `false` | Use prescribed CCN. |
| `max_total_ni` | real | `2000.0e3` | Cap on total ice number concentration. |

### `physics.turbulence`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable_turbulence` | bool | `false` | Enables the subgrid turbulence scheme, which produces `RKM`/`RKH`. |

### `physics.surface_process`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable` | bool | `false` | Enables the surface process wrapper. |
| `frequency_s` | real | `1` | Surface/land/ocean call frequency in seconds. Should divide evenly by `simulation.dt_s`. |
| `land_scheme` | string | `"none"` | `none` or `noahlsm`. `noahlsm` calls the Fortran Noah land model on land points. |
| `ocean_scheme` | string | `"none"` | `none`, `sflux_2d`, `sflux_tc_2d`, or `tco_ocean`. |

Choosing `sflux_2d` or `sflux_tc_2d` uses the C++ surface-flux implementation for ocean points and disables the ocean part inside the land process. `grid.vertical_coordinate_type: rcemip` also relaxes the surface wind-speed floor from `1e-3` to `1`.

### `physics.rrtmgp`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable_rrtmgp` | bool | `false` | Enables RRTMGP radiation. When on, `th` additionally integrates a forward-Euler radiative tendency. |
| `rad_frequency_s` | real | `1.0` | Radiation call frequency in seconds. Should divide evenly by `simulation.dt_s`. |
| `column_chunk_size` | int | all columns on the rank | Columns processed per chunk. Larger is usually faster but uses more memory; tune with domain size and rank count. |
| `pool_size_multiplier` | real | `1.0` | Scales the radiation memory pool. |
| `active_gases` | list | `["h2o","co2","o3","n2o","co","ch4","o2","n2"]` | Gases passed to RRTMGP. |
| `do_aerosol_rad` | bool | `false` | Include aerosol optics. |
| `extra_clnsky_diag` | bool | `false` | Extra clean-sky diagnostics. |
| `extra_clnclrsky_diag` | bool | `false` | Extra clean-clear-sky diagnostics. |

#### Prescribed gas volume mixing ratios

Surface values, mol/mol. Used for any gas in `active_gases` that the model does not carry itself.

| Key | Default | | Key | Default |
| --- | --- | --- | --- | --- |
| `co2vmr` | `355.03e-6` | | `o2vmr` | `0.209` |
| `n2ovmr` | `320e-9` | | `n2vmr` | `0.7906` |
| `ch4vmr` | `1700e-9` | | `f11vmr` | `0.0` |
| `covmr` | `1.0e-7` | | `f12vmr` | `0.0` |
| `o3vmr` | `0.3017e-7` | | | |

#### Sun and calendar

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `time.year`, `.month`, `.day`, `.hour`, `.minute`, `.second` | int | `-9999` | Calendar start time driving the solar cycle. `-9999` leaves the component unset. Note `time.hour` is also read by the output layer (defaulting to `16` there) to derive the GrADS start hour as `(hour + 8) % 24`. |
| `orbital_eccentricity` | real | `-9999.0` | Override Earth's orbital eccentricity. Values below zero mean "use the default orbit". |
| `orbital_obliquity` | real | `-9999.0` | Override the obliquity. |
| `orbital_mvelp` | real | `-9999.0` | Override the moving vernal equinox longitude of perihelion. |
| `fixed_total_solar_irradiance` | real | `-9999.0` | When positive, prescribes an invariant TOA solar constant (W m⁻²) instead of the orbital one. For idealized experiments such as RCE. |
| `fixed_solar_zenith_angle` | real | `-9999.0` | When positive, this value is used directly as the cosine of the solar zenith angle for every column, bypassing the orbital calculation. |

#### Data files

| Key | Type | Default |
| --- | --- | --- |
| `coefficients_file_lw` | string | `<VVM_ROOT>/rundata/rrtmgp/rrtmgp-data-lw-g128-210809.nc` |
| `coefficients_file_sw` | string | `<VVM_ROOT>/rundata/rrtmgp/rrtmgp-data-sw-g112-210809.nc` |
| `cloud_optics_file_lw` | string | `<VVM_ROOT>/rundata/rrtmgp/rrtmgp-cloud-optics-coeffs-lw.nc` |
| `cloud_optics_file_sw` | string | `<VVM_ROOT>/rundata/rrtmgp/rrtmgp-cloud-optics-coeffs-sw.nc` |

The g-point counts in the default file names must match the top-level `nlwgpts`/`nswgpts` below. Change both together or radiation allocates the wrong buffer sizes.

### Top-level radiation keys

Three RRTMGP settings are read from the **root** of the JSON, not from under `physics.rrtmgp`. This is inherited from the EAMxx interface and is easy to miss.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `nlwgpts` | int | `128` | Longwave g-points. Must match `coefficients_file_lw`. |
| `nswgpts` | int | `112` | Shortwave g-points. Must match `coefficients_file_sw`. |
| `do_subcol_sampling` | bool | `true` | Use McICA sub-column sampling for cloud overlap. |

## `optimization`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `cuda_graph_halo_exchange` | list | *(none)* | Fields whose halo exchange is captured into a CUDA graph, e.g. `["u", "w", "xi", "eta", "zeta", "th"]` plus hydrometeors. Ignored on CPU builds. |

Only include fields that exist in the current state and are exercised by the run. If you disable a physics package, remove its fields from the list unless the code still allocates them for your case.

## `performance.timing`

Controls the built-in timing instrumentation. All optional.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable` | bool | `true` | Collect timers at all. |
| `warmup_steps` | int | `0` | Skip this many steps before any timer records, so JIT, first-touch allocation and graph capture do not land in the averages. |
| `fence_gpu` | bool | `false` | Fence the device around each timed region. Gives correct per-region GPU times at the cost of removing asynchrony — use it for attribution, not for headline throughput numbers. |
| `print_interval_steps` | int | `0` | Print the timing report every N steps. `0` prints only at the end of the run. |
| `reset_after_interval_print` | bool | `false` | Zero the accumulators after each interval print, so each report covers only the interval rather than the run so far. |

## `constants`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `gravity` | real | *required* | Gravitational acceleration. |
| `Rd` | real | *required* | Dry-air gas constant. |
| `Cp` | real | *required* | Heat capacity at constant pressure. |
| `Lv` | real | *required* | Latent heat of vaporization. |
| `P0` | real | *required* | Reference pressure. |
| `PI` | real | *required* | Pi. |
| `OMEGA` | real | `7.292e-5` | Earth rotation rate, used by the default latitude-dependent Coriolis parameter. |

### Coriolis on an f- or beta-plane

Present `constants.coriolis_parameter` and the latitude-dependent formula is replaced by a plane approximation, `f = f0 + beta * (y - y_ref)`. Absent, the sphere formula using `OMEGA` is used, so existing configurations are unchanged.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `coriolis_parameter` | real | *(absent)* | `f0`. Its **presence** is what switches on the plane approximation. |
| `coriolis_beta` | real | `0.0` | Meridional gradient of `f`. Left at zero this is an f-plane. |
| `coriolis_reference_y_m` | real | domain centre | `y_ref`, in meters. |

Change constants only for controlled sensitivity experiments.

## Keys that look real but are not read

These appear in some shipped sample configurations and have no effect. They are listed so you do not spend time tuning them:

| Key | Status |
| --- | --- |
| `output.enable_netcdf` | Never read. Output engine selection is `output.engine` alone. |
| `optimization.cuda_graph_solver` | Never read. Only `optimization.cuda_graph_halo_exchange` exists. |
| `constants.PSFC` | Never read. A `PSFC` parameter view is allocated but nothing assigns it from the configuration. Surface pressure comes from the sounding profile. |

## Consistency checklist

Before submitting a long run, check:

- `physics.rrtmgp.rad_frequency_s`, `physics.surface_process.frequency_s`, and `simulation.output_interval_s` are sensible multiples of `simulation.dt_s`.
- Every `output.fields_to_output` name is allocated by the selected dynamics/physics configuration.
- If `output.engine` is `SST`, submit with `submit.py --io N`. If it is `HDF5` or `BP5`, omit `--io`.
- `grid.n_halo_cells` is `3` if any tracer uses `weno5`.
- If `restart.enable` is true, the restart file stores `model_time_s`/`model_step` (or an elapsed-seconds `time`); otherwise set `restart.legacy_time_s` explicitly. The file *name* no longer affects the restart time.
- The NetCDF variables listed in `netcdf_reader.variables_to_read.2d` exist in `netcdf_reader.source_file`.
- P3 hydrometeor variables are present in `dynamics.prognostic_variables` when `physics.p3.enable_p3` is true.
- CUDA graph field lists match the fields actually allocated in the run.
- Optional keys you added are spelled exactly as in this page — a typo is silently ignored.

## CMake and environment

Library paths and compilers are **not** set in this JSON file. Use `CMakePresets.json` and CMake cache variables for HDF5, NetCDF, PnetCDF, NVHPC, Kokkos, and MPI (see [Quick Start](../quick-start.md)).
