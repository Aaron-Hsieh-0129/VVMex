# CLEO super-droplet microphysics

VVMex can replace P3 with [CLEO](https://github.com/yoctoyotta1024/CLEO), a
Lagrangian super-droplet model (SDM; Shima et al., 2009). Each superdroplet
stands for a multiplicity `xi` of identical droplets with a radius and a
soluble aerosol mass. At every coupling step VVMex hands CLEO the winds and the
thermodynamic state, CLEO moves the superdroplets and runs condensation and
collision–coalescence–breakup, and VVMex takes back `th`, `qv` and the
condensate. Ice is not represented.

## Requirements

- **A Kokkos built with relocatable device code** on GPU systems:
  `-DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON -DBUILD_SHARED_LIBS=OFF`
  (see [GPU environment](../environment/gpu.md)).
- **The CLEO source.** CMake fetches the fork named in
  `externals/cleo/CMakeLists.txt`. To build against a local checkout instead,
  add `-DFETCHCONTENT_SOURCE_DIR_CLEO=/path/to/CLEO`. Only this one target,
  `vvm_cleo`, is compiled as C++20.
- **Python for the input generator**: numpy and cleopy's dependencies. The
  generator imports cleopy from the CLEO checkout the build uses
  (`CLEO_SOURCE_DIR` in `build/CMakeCache.txt`), so the binaries are made with
  the same `cleoconstants.hpp` the model is compiled with.

## Running a case

1. **Configure.** Start from
   `rundata/input_configs/default_cases/cleo_2dbubble.json`, or add to any
   Cartesian case:

    ```json
    "physics": {
        "p3": { "enable_p3": false },
        "microphysics": {
            "engine": "cleo",
            "aerosol": { ... },
            "cleo": { ... }
        }
    }
    ```

    Every key is listed under
    [`physics.microphysics`](configuration.md#physicsmicrophysics).
    `cleo_2dbubble.json` uses the dry `profile_dry.txt`, so its superdroplets
    stay haze; point `initial_conditions.source_file` at a moist sounding such
    as `tpe_20240717i.txt` to form cloud and rain.

2. **Generate CLEO's inputs** from the same JSON:

    ```bash
    .venv/bin/python tools/cleo/gen_cleo_inputs.py -c <case.json>
    ```

    It writes the gridbox boundaries (`init_gbx_path`), the initial
    superdroplets (`init_supers_path`) and a `cleo_config.yaml` that cleopy
    needs. The gridboxes are the model's physical cells, with the vertical
    levels rebuilt as `Initializer.cpp` builds them (`default`, including the
    `dz1` stretching, or `rcemip`; `taiwanvvm` is not supported). Superdroplets
    are sampled from the lognormal modes in `physics.microphysics.aerosol`, `per_gridbox` in
    every gridbox below `seed_below_z_m`. Regenerate whenever the grid or these
    settings change. Relative paths are taken from the repository root, where
    `submit.py` runs the model.

3. **Run** as any other case, e.g.
   `python3 submit.py -c <case.json> --compute N`.

## What the model receives

| Field | Meaning |
| --- | --- |
| `th`, `qv` | Updated by condensation and its latent heating. |
| `qp` | All condensate carried by the superdroplets, haze included, from initialization on; read by the buoyancy. |
| `qc`, `nc` | Cloud: superdroplets with `cloud_min_radius_um <= r < rain_min_radius_um`. |
| `qr`, `nr` | Rain: superdroplets with `r >= rain_min_radius_um`. |
| `diag_eff_radius_qc`, `diag_eff_radius_qr` | `sum(xi r^3) / sum(xi r^2)` [um]; 10 and 500 um in empty cells, as P3. Read by RRTMGP. |
| `qi`, `diag_eff_radius_qi` | Zero and 25 um (P3's ice-free value), for RRTMGP. |
| `precip_liq_surf_flux` | Condensate leaving the bottom gridboxes through the ground, as a mean over the last coupling step [kg m-2 s-1]. Read by the land model. |
| `precip_liq_surf_mass` | The same, averaged over each output interval, as P3 does. |
| `precip_ice_surf_*` | Zero. |

`qc`, `qr`, `nc` and `nr` are diagnostic: they are recomputed from the
superdroplets after every coupling and are not advected.

## CLEO's own output

CLEO writes a zarr store to `output.output_dir` + `output.zarrbasedir` every
`timesteps_s.observation` from t = 0 up to `simulation.total_time_s`, between
couplings too: every superdroplet
(`sdId`, `xi`, `radius`, `msol`, coordinates, gridbox), the gridbox state
(`press`, `temp`, `qvap`, `qcond`, winds), mass moments and process monitors.
Each array carries `units` and `scale_factor` attributes; multiply by
`scale_factor` (CF-aware readers do this) to get those units.

## How the coupling works

- **When.** `VVM_CLEO_Interface::run` is called once per model step, after the
  thermodynamic update and before turbulence, where P3 would run. When the model
  time reaches the next coupling time (the first at t = 0), CLEO is advanced by
  one `timesteps_s.coupling` interval, stopping at each observation time on the
  way. CLEO thus runs one interval ahead of the model, and its last interval
  ends at `simulation.total_time_s`.
- **Initialization.** `qp`, CLEO's `qcond` and the cloud and rain diagnostics are
  computed from the initial superdroplets, so the initial output and the first
  observation already hold their water.
- **Sent.** For each gridbox: `u`, `v`, `w` on its two faces in each direction,
  `T = th * pibar`, `qv` and `pbar`.
- **Returned.** `th`, `qv` and `qp` in the physical cells, followed by a halo
  exchange; then the diagnostics and the surface precipitation above.
- **average_over_y.** In a quasi-2-D run the y cells are replicate samples of
  one x–z slice; their stochastic differences are noise that the dynamics can
  amplify. With `average_over_y` every y column gets the column mean.
- **MPI.** CLEO uses VVMex's x–y decomposition; superdroplets crossing a rank
  boundary are exchanged by CLEO, and every rank writes its share of one zarr
  store.
- **Code.** `src/physics/cleo/VVM_cleo_process_interface.cpp`. CLEO's headers
  are C++20, so they are included only there, behind a PIMPL; the rest of
  VVMex stays C++17. `-DVVM_CLEO_DEBUG=ON` builds that file at `-O0 -g`.

## Limitations

- Cartesian grids with periodic x and y only. CLEO wraps superdroplets around
  both horizontal edges, so other geometries and `zero_gradient` boundaries are
  refused.
- Flat terrain only: superdroplets would fall into raised ground, so a nonzero
  `topo` is refused.
- `simulation.total_time_s` must be a whole number of `timesteps_s.coupling`.
- No restart: superdroplets are not written to the checkpoint, so a restart
  would start again from the initial binary.
- `physics.microphysics.engine` selects CLEO only; P3 is still switched by
  `physics.p3.enable_p3`, and the two cannot run together.
- Large-scale subsidence is not passed to the superdroplets.
- With condensation active, repeated GPU runs agree to round-off, not bit for
  bit. With `process.collision_seed = 0` collisions are also seeded from the
  system; a fixed seed removes that source of difference.

## Tests

Configure with `-DVVM_TEST_CLEO=ON` and run `ctest -L cleo`. The tier covers
the field contract, the coupling scenarios (transport, condensation, y
averaging, rain, collisions, advection), the full model, and, with
`VVM_TEST_MULTIRANK`, 1- against 2- and 4-rank runs; see `tests/README.md`.
