# Default cases

VVMex ships a set of ready-to-run sample cases under `rundata/`. These cases are intended as starting points for new users: choose one JSON file, run it through `submit.py`, then copy it and edit the copy for your own experiment.

The default-case inputs are organized in three matching directories:

| Directory | Contents |
| --------- | -------- |
| `rundata/input_configs/default_cases/` | JSON runtime configurations passed to `submit.py -c` or directly to `build/vvm`. |
| `rundata/initial_conditions/profiles/default_cases/` | One-dimensional sounding/profile text files referenced by `initial_conditions.source_file`. |
| `rundata/initial_conditions/spatial/default_cases/` | Spatial NetCDF files referenced by `netcdf_reader.source_file`. These files can be generated with `tools/generate_init_nc.py`. |

## Run a sample case

From the project root:

```bash
cd $VVM_ROOT
./submit.py --local --preset <your_preset_name> -c ./rundata/input_configs/default_cases/advection_u.json --compute 1
```

For a larger local or SLURM run, keep the same `-c` path and change the resource options:

```bash
./submit.py --preset <your_preset_name> -c ./rundata/input_configs/default_cases/sea_grass_mountain.json --compute 16 --nodes 1 --gpus 16
```

If you copy a default case before editing, keep paths relative to the project root unless you also change your launch directory. If you do not know which values to provide to the wrapper, run `./submit.py` and answer the prompts step by step.

For local runs on specific GPUs, set `VVM_GPU_LIST` before the command:

```bash
VVM_GPU_LIST=0,1,2,3,4,5,6,7 ./submit.py --local \
  -c "rundata/input_configs/default_cases/taiwanvvm_2048.json" \
  --preset blaze \
  --compute 8 \
  --nodes 1
```

## Available cases

| Case config | Type | Profile | Spatial input |
| ----------- | ---- | ------- | ------------- |
| `2dbubble.json` | Idealized dynamics, `idealized_test = 2dbubble` | `profile_dry.txt` | `init_regression_test.nc` |
| `advection_u.json` | Idealized dynamics, `idealized_test = advection_u` | `profile_dry.txt` | `init_regression_test.nc` |
| `advection_v.json` | Idealized dynamics, `idealized_test = advection_v` | `profile_dry.txt` | `init_regression_test.nc` |
| `advection_w.json` | Idealized dynamics, `idealized_test = advection_w` | `profile_dry.txt` | `init_regression_test.nc` |
| `stretching.json` | Idealized dynamics, `idealized_test = stretching` | `profile_dry.txt` | `init_regression_test.nc` |
| `twisting.json` | Idealized dynamics, `idealized_test = twisting` | `profile_dry.txt` | `init_regression_test.nc` |
| `p3_bubble_shear.json` | Moist bubble / P3-oriented sample | `p3_bubble_shear.txt` | `p3_bubble_shear.nc` |
| `rcemip.json` | RCEMIP-style radiative-convective equilibrium | `RCEMIP_VVM_300k_init.txt` | `rcemip.nc` |
| `grass.json` | Full-physics land sample over grass surface | `tpe_20240717i.txt` | `grass.nc` |
| `evergreen.json` | Full-physics land sample over evergreen surface | `tpe_20240717i.txt` | `evergreen.nc` |
| `urban.json` | Full-physics land sample over urban surface | `tpe_20240717i.txt` | `urban.nc` |
| `sea_grass_mountain.json` | Mixed sea, grass, and mountain sample | `TNNUA.txt` | `sea_grass_mountain.nc` |
| `sea_urban_mountain.json` | Mixed sea, urban, and mountain sample | `TNNUA.txt` | `sea_urban_mountain.nc` |
| `taiwanvvm_2048.json` | Large TaiwanVVM-style case | `tpe_20120819_at004.txt` | `taiwanvvm_2048.nc` |

Before running large or machine-specific cases, check `output.output_dir`, `grid`, and resource settings. For example, `taiwanvvm_2048.json` is a large configuration and may use an output path that should be changed for your system.

## Regenerate spatial NetCDF inputs

The files under `rundata/initial_conditions/spatial/default_cases/` can be generated with `tools/generate_init_nc.py`. The script reads the JSON selected by `CONFIG_PATH` and writes the NetCDF file named by that config's `netcdf_reader.source_file`.

Typical workflow:

1. Set `VVM_ROOT` and install the Python dependencies used by the tool (`netCDF4`, `numpy`, `xarray`, `scipy`).
2. Edit `CONFIG_PATH` near the top of `tools/generate_init_nc.py`, for example:

   ```python
   CONFIG_PATH = './rundata/input_configs/default_cases/evergreen.json'
   ```

3. Set `USE_TAIWAN_TOPO` for the desired input mode. `True` coarsens Taiwan topography from `rundata/land/topolsm_TW.nc`; `False` uses the idealized terrain and land-type helper functions in the script.
4. Run the tool from anywhere after `VVM_ROOT` is set:

   ```bash
   python tools/generate_init_nc.py
   ```

When generating an idealized land case, adjust the helper functions in the script (`get_ideal_vegtype_data`, `get_ideal_soiltype_data`, `get_albedo_data`, and related fields) so the generated surface pattern matches the selected default-case JSON.
