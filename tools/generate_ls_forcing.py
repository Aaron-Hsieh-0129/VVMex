#!/usr/bin/env python3
"""Generate VVMex nudging files from local ERA5 pressure-level data."""

import json
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

# User settings. Relative paths are resolved from VVM_ROOT.
VVM_ROOT = Path(os.environ.get("VVM_ROOT", Path(__file__).resolve().parents[1]))
# True: interpolate local ERA5 data. False: call define_idealized_forcing().
USE_ERA5_FORCING = True
# CONFIG_PATH = Path(os.environ.get("VVM_CONFIG_PATH", VVM_ROOT / "rundata/input_configs/taiwanvvmex_nudge.json"))
CONFIG_PATH = Path(os.environ.get("VVM_CONFIG_PATH", VVM_ROOT / "rundata/input_configs/taiwanvvmex_nudge_areamn_time_varying_wide_tau3600wind_0610.json"))
DYNAMICS_FILE = Path("/raid/mog/aaron/era5/7c9d50d3a3297b30326e05aa513f04c0.grib")
HUMIDITY_FILE = Path("/raid/mog/aaron/era5/98cd14e8c8b23708b60a3dbbf54a215.grib")
START_TIME_UTC = np.datetime64("2012-06-09T16:00:00")
END_TIME_UTC = np.datetime64("2012-06-14T16:00:00")
LON_MIN, LON_MAX = 116.00, 126.00
LAT_MIN, LAT_MAX = 19.25, 28.25

# U and V are cosine-latitude-weighted over this independent ERA5 box.
WIND_MEAN_LON_MIN, WIND_MEAN_LON_MAX = 116.00, 126.00
WIND_MEAN_LAT_MIN, WIND_MEAN_LAT_MAX = 19.25, 28.25

# ERA5 diagnostic figures are saved below the forcing output directory.
PLOT_ERA5 = True
PLOT_TIME_UTC = START_TIME_UTC
PLOT_PRESSURE_PA = 85000.0
PLOT_SUBDIRECTORY = "visualizations"

# Time steps are generated in separate processes. Use 1 to disable multiprocessing.
PROCESS_COUNT = 16
OUTPUT_DTYPE = "f8"
USE_COMPRESSION = True
COMPRESSION_LEVEL = 2

VARIABLE_ALIASES = {
    "temperature": ("temperature", "t", "var130"),
    "specific_humidity": ("specific_humidity", "q", "var133"),
    "u": ("u", "var131"), "v": ("v", "var132"),
}
COORDINATE_ALIASES = {
    "time": ("time", "valid_time"),
    "level": ("level", "plev", "isobaricInPa", "isobaricInhPa", "pressure_level"),
    "latitude": ("latitude", "lat"), "longitude": ("longitude", "lon"),
}


class InputError(RuntimeError):
    pass


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else VVM_ROOT / path


def require(mapping, key, context):
    if key not in mapping:
        raise InputError(f"Missing '{key}' in {context}.")
    return mapping[key]


def read_config():
    path = resolve_path(CONFIG_PATH)
    try:
        with path.open() as stream:
            return json.load(stream), path
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f"Cannot read VVMex configuration '{path}': {error}") from error


def read_profile(path):
    """Read the whitespace-column format consumed by TxtReader."""
    path = resolve_path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise InputError(f"Cannot read VVMex profile '{path}': {error}") from error
    header, columns = None, {}
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#") or "===" in line:
            continue
        if header is None:
            header = line.split()
            if len(header) != len(set(header)):
                raise InputError(f"Profile '{path}' contains duplicate column names.")
            columns = {name: [] for name in header}
            continue
        values = line.split()
        if len(values) != len(header):
            raise InputError(
                f"Profile '{path}', line {line_number}, has {len(values)} values; "
                f"expected {len(header)}.")
        for name, value in zip(header, values):
            try:
                columns[name].append(float(value))
            except ValueError as error:
                raise InputError(
                    f"Profile '{path}', line {line_number}, column '{name}' "
                    f"is not numeric: {value!r}.") from error
    if header is None:
        raise InputError(f"Profile '{path}' has no header.")
    for name in ("pbar", "qvbar"):
        if name not in columns:
            raise InputError(f"Profile '{path}' is missing required column '{name}'.")
    if "Tbar" not in columns and "thbar" not in columns:
        raise InputError(f"Profile '{path}' is missing required column 'Tbar' or 'thbar'.")
    result = {name: np.asarray(values, dtype="f8") for name, values in columns.items()}
    if result["pbar"].size < 2:
        raise InputError(f"Profile '{path}' must contain at least two data rows.")
    if not all(np.all(np.isfinite(values)) for values in result.values()):
        raise InputError(f"Profile '{path}' contains non-finite values.")
    if np.any(result["pbar"] <= 0) or not np.all(np.diff(result["pbar"]) < 0):
        raise InputError(f"Profile '{path}' pbar must be positive and strictly decreasing.")
    return result, path


def interpolate_extrapolate(target, source, values):
    """Piecewise-linear interpolation with TxtReader-style endpoint extrapolation."""
    target, source, values = map(lambda value: np.asarray(value, dtype="f8"),
                                 (target, source, values))
    result = np.interp(target, source, values)
    below, above = target <= source[0], target >= source[-1]
    slope = (values[1] - values[0]) / (source[1] - source[0])
    result[below] = values[0] + slope * (target[below] - source[0])
    slope = (values[-1] - values[-2]) / (source[-1] - source[-2])
    result[above] = values[-1] + slope * (target[above] - source[-1])
    return result


def vvm_vertical_grid(grid):
    """Construct halo-inclusive z_mid and flex_height_coef_up as VVMex does."""
    nz = int(require(grid, "nz", "grid"))
    halo = int(require(grid, "n_halo_cells", "grid"))
    dz, dz1 = float(require(grid, "dz", "grid")), float(require(grid, "dz1", "grid"))
    if nz < 2 or halo < 1 or dz <= 0 or dz1 <= 0:
        raise InputError("grid.nz, n_halo_cells, dz, and dz1 define an invalid grid.")
    total = nz + 2 * halo
    domain = 15000.0
    cz2 = (dz - dz1) / (dz * (domain - dz))
    cz1 = 1.0 - cz2 * domain
    z_up, z_mid, flex_up = (np.zeros(total, dtype="f8") for _ in range(3))
    for k in range(halo, total):
        z_up[k] = z_up[k - 1] + dz
    z_mid[halo] = 0.5 * dz
    for k in range(halo + 1, total):
        z_mid[k] = z_mid[k - 1] + dz

    coordinate_type = grid.get("vertical_coordinate_type", "default")
    if coordinate_type == "default":
        for k in range(halo - 1, total):
            flex_up[k] = 1.0 / (cz1 + 2.0 * cz2 * z_up[k])
            z_up[k] *= cz1 + cz2 * z_up[k]
            z_mid[k] *= cz1 + cz2 * z_mid[k]
    elif coordinate_type == "taiwanvvm":
        for k in range(halo - 1, total):
            flex_up[k] = 1.0 / (cz1 + 2.0 * cz2 * z_up[k])
            z_up[k] *= cz1 + cz2 * z_up[k]
        kt = int((1.0 - cz1) / cz2 / 2.0 / dz)
        kt_index = min(max(kt + halo - 2, 0), total - 1)
        kt1 = int(z_up[kt_index] / dz1 + 0.999)
        z_up[halo - 1], flex_up[halo - 1] = 0.0, dz / dz1
        for level in range(2, kt1):
            center = level + halo - 2
            for k in range(total - 1, center, -1):
                flex_up[k], z_up[k] = flex_up[k - 1], z_up[k - 1] + dz1
            if center < total:
                flex_up[center], z_up[center] = dz / dz1, z_up[center - 1] + dz1
        for k in range(halo - 1, total):
            z_mid[k] *= cz1 + cz2 * z_mid[k]
        z_mid[halo - 1], z_mid[halo] = 0.0, 0.5 * dz1
        for level in range(3, kt1 + 1):
            center = level + halo - 2
            for k in range(total - 1, center, -1):
                z_mid[k] = z_mid[k - 1] + dz1
            if center < total:
                z_mid[center] = z_mid[center - 1] + dz1
        transition = kt1 + halo - 1
        if transition < total and transition + 1 < total:
            flex_up[transition] = dz / (
                z_mid[transition + 1] - z_mid[transition])
    else:
        raise InputError(f"Unsupported vertical_coordinate_type={coordinate_type!r}.")
    if (not np.all(np.diff(z_mid[halo - 1:]) > 0)
            or not np.all(np.isfinite(flex_up[halo - 1:]))
            or np.any(flex_up[halo - 1:] <= 0)):
        raise InputError("Calculated VVMex vertical grid is invalid.")
    return z_mid, z_up, flex_up, halo, total


def calculate_vvm_pressure(config):
    """Reproduce TxtReader's physical-layer pbar profile, excluding halos."""
    grid = require(config, "grid", "configuration")
    constants = require(config, "constants", "configuration")
    initial = require(config, "initial_conditions", "configuration")
    profile, profile_path = read_profile(require(initial, "source_file", "initial_conditions"))
    nz, halo = int(require(grid, "nz", "grid")), int(require(grid, "n_halo_cells", "grid"))
    if grid.get("vertical_coordinate_type", "default") == "rcemip":
        total, source = nz + 2 * halo, profile["pbar"]
        pbar = np.empty(total, dtype="f8")
        for k in range(halo - 1, total):
            index = k - (halo - 1)
            pbar[k] = (source[index] if index < source.size else
                       pbar[k - 1] * pbar[k - 1] / pbar[k - 2])
        return pbar[halo:halo + nz], profile_path

    z_mid, z_up, flex_up, halo, total = vvm_vertical_grid(grid)
    p0 = float(require(constants, "P0", "constants"))
    rd = float(require(constants, "Rd", "constants"))
    cp = float(require(constants, "Cp", "constants"))
    gravity = float(require(constants, "gravity", "constants"))
    dz = float(require(grid, "dz", "grid"))
    pressure = profile["pbar"]
    exner = np.power(pressure / p0, rd / cp)
    temperature = profile["Tbar"] if "Tbar" in profile else profile["thbar"] * exner
    virtual_temperature = temperature * (1.0 + 0.608 * profile["qvbar"])
    pilog_source = np.log(exner)
    input_height = np.zeros(pressure.size, dtype="f8")
    for k in range(1, pressure.size):
        input_height[k] = input_height[k - 1] - cp / (2.0 * gravity) * (
            pilog_source[k] - pilog_source[k - 1]) * (
                virtual_temperature[k] + virtual_temperature[k - 1])
    if not np.all(np.diff(input_height) > 0):
        raise InputError(f"Profile '{profile_path}' produces non-increasing heights.")

    tbar, qvbar, pbar = (np.empty(total, dtype="f8") for _ in range(3))
    pibar, pilog = np.empty(total, dtype="f8"), np.empty(total, dtype="f8")
    tbar[halo - 1], qvbar[halo - 1], pbar[halo - 1] = (
        temperature[0], profile["qvbar"][0], pressure[0])
    tbar[halo:] = interpolate_extrapolate(z_mid[halo:], input_height, temperature)
    qvbar[halo:] = interpolate_extrapolate(
        z_mid[halo:], input_height, profile["qvbar"])
    pibar[halo - 1] = np.power(pbar[halo - 1] / p0, rd / cp)
    pilog[halo - 1] = np.log(pibar[halo - 1])
    gdzbcp = 2.0 * gravity * dz / cp
    for _ in range(3):
        tvbar = tbar * (1.0 + 0.608 * qvbar)
        pilog[halo] = pilog[halo - 1] - gdzbcp / (
            tvbar[halo - 1] + tvbar[halo]) * (
                z_mid[halo] - z_up[halo - 1]) / dz
        for k in range(halo + 1, total):
            pilog[k] = pilog[k - 1] - gdzbcp / (
                tvbar[k - 1] + tvbar[k]) / flex_up[k - 1]
        pibar[halo:] = np.exp(pilog[halo:])
        pbar[halo:] = p0 * np.power(pibar[halo:], cp / rd)
    result = pbar[halo:halo + nz]
    if (not np.all(np.isfinite(result)) or np.any(result <= 0)
            or not np.all(np.diff(result) < 0)):
        raise InputError("Calculated VVMex physical-layer pressure is invalid.")
    return result, profile_path


def find_variable(dataset, candidates, path, description):
    """Find an ERA5 field using the same candidate search as era5.ipynb."""
    for name in candidates:
        if name in dataset.data_vars:
            print(f"ERA5 {description}: found variable '{name}' in {path}")
            return dataset[name]
    raise InputError(
        f"None of {candidates} was found for ERA5 {description} in '{path}'. "
        f"Available data variables: {list(dataset.data_vars)}.")


def find_coordinate(data, candidates, path, description):
    """Find a coordinate name using the same candidate search as era5.ipynb."""
    for name in candidates:
        if name in data.coords:
            return name
    raise InputError(
        f"None of {candidates} was found for the {description} coordinate in "
        f"ERA5 file '{path}'. Available coordinates: {list(data.coords)}.")


def open_era5(path):
    path = resolve_path(path)
    if not path.is_file():
        raise InputError(f"ERA5 input does not exist: '{path}'.")
    try:
        if path.suffix.lower() in {".grib", ".grb", ".grib2"}:
            data = xr.open_dataset(path, engine="cfgrib", backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}, "indexpath": ""})
        else:
            data = xr.open_dataset(path)
    except (ImportError, OSError, ValueError) as error:
        raise InputError(f"Cannot open ERA5 input '{path}': {error}") from error
    return data, path


def canonical_variable(dataset, path, canonical):
    data = find_variable(
        dataset, VARIABLE_ALIASES[canonical], path, canonical)
    rename = {}
    for coordinate, aliases in COORDINATE_ALIASES.items():
        actual = find_coordinate(data, aliases, path, coordinate)
        if actual != coordinate:
            rename[actual] = coordinate
    data = data.rename(rename)
    expected = {"time", "level", "latitude", "longitude"}
    for dimension in tuple(data.dims):
        if dimension not in expected:
            if data.sizes[dimension] != 1:
                raise InputError(
                    f"ERA5 '{data.name}' in '{path}' has unsupported dimension "
                    f"'{dimension}' of size {data.sizes[dimension]}.")
            data = data.isel({dimension: 0}, drop=True)
    missing = expected.difference(data.dims)
    if missing:
        raise InputError(
            f"ERA5 '{data.name}' in '{path}' is missing {sorted(missing)}.")
    return data.rename(canonical)


def load_era5():
    dynamics, dynamics_path = open_era5(DYNAMICS_FILE)
    humidity = None
    try:
        humidity_path = resolve_path(HUMIDITY_FILE)
        humidity, humidity_path = ((dynamics, dynamics_path) if humidity_path == dynamics_path
                                   else open_era5(HUMIDITY_FILE))
        variables = [canonical_variable(dynamics, dynamics_path, name)
                     for name in ("temperature", "u", "v")]
        variables.append(canonical_variable(humidity, humidity_path, "specific_humidity"))
        try:
            dataset = xr.merge(variables, join="exact", compat="equals").load()
        except (ValueError, xr.MergeError) as error:
            raise InputError(f"ERA5 variables have incompatible coordinates: {error}") from error
    finally:
        if humidity is not None and humidity is not dynamics:
            humidity.close()
        dynamics.close()

    longitude = np.asarray(dataset.longitude, dtype="f8")
    if longitude.min() >= 0 and longitude.max() <= 360:
        normalized = (longitude + 180.0) % 360.0 - 180.0
        if np.unique(normalized).size != normalized.size:
            raise InputError("ERA5 longitude has a duplicate 0/360 endpoint.")
        dataset = dataset.assign_coords(longitude=normalized)
    elif longitude.min() < -180 or longitude.max() > 180:
        raise InputError("ERA5 longitude must use -180..180 or 0..360.")
    dataset = dataset.sortby("longitude").sortby("latitude").sortby("time")
    for coordinate in ("longitude", "latitude"):
        values = np.asarray(dataset[coordinate], dtype="f8")
        if values.size < 2 or not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
            raise InputError(f"ERA5 {coordinate} must be finite, unique, and increasing.")
    times = np.asarray(dataset.time)
    if (not np.issubdtype(times.dtype, np.datetime64) or times.size == 0
            or np.unique(times).size != times.size
            or np.any(np.diff(times) <= np.timedelta64(0, "s"))):
        raise InputError("ERA5 times must decode to unique, increasing datetimes.")
    return dataset


def pressure_values(dataset):
    pressure = np.asarray(dataset.level, dtype="f8").copy()
    if "hpa" in str(dataset.level.attrs.get("units", "")).lower() or pressure.max() < 2000:
        pressure *= 100.0
    if (pressure.size < 2 or np.any(pressure <= 0) or not np.all(np.isfinite(pressure))
            or np.unique(pressure).size != pressure.size):
        raise InputError("ERA5 pressure levels must be finite, positive, and unique.")
    return pressure


def check_box(dataset, bounds, description):
    lon_min, lon_max, lat_min, lat_max = bounds
    values = np.asarray(bounds, dtype="f8")
    if (not np.all(np.isfinite(values)) or lon_min >= lon_max or lat_min >= lat_max):
        raise InputError(f"{description} bounds are invalid, reversed, or empty.")
    if lon_min < -180 or lon_max > 180 or lat_min < -90 or lat_max > 90:
        raise InputError(
            f"{description} bounds must use lon -180..180 and lat -90..90.")
    available = (float(dataset.longitude.min()), float(dataset.longitude.max()),
                 float(dataset.latitude.min()), float(dataset.latitude.max()))
    if (lon_min < available[0] or lon_max > available[1]
            or lat_min < available[2] or lat_max > available[3]):
        raise InputError(
            f"{description} lon={lon_min:g}..{lon_max:g}, "
            f"lat={lat_min:g}..{lat_max:g} "
            f"is outside ERA5 lon={available[0]:g}..{available[1]:g}, "
            f"lat={available[2]:g}..{available[3]:g}.")


def check_domains(dataset):
    check_box(dataset, (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), "Target domain")
    check_box(
        dataset,
        (WIND_MEAN_LON_MIN, WIND_MEAN_LON_MAX,
         WIND_MEAN_LAT_MIN, WIND_MEAN_LAT_MAX),
        "Wind-mean domain")


def wind_domain_mean(data):
    """Cosine-latitude-weighted mean, matching era5.ipynb."""
    selected = data.sel(
        longitude=slice(WIND_MEAN_LON_MIN, WIND_MEAN_LON_MAX),
        latitude=slice(WIND_MEAN_LAT_MIN, WIND_MEAN_LAT_MAX))
    if selected.sizes.get("longitude", 0) == 0 or selected.sizes.get("latitude", 0) == 0:
        raise InputError(
            "The wind-mean box contains no ERA5 longitude/latitude grid points.")
    weights = np.cos(np.deg2rad(selected.latitude))
    result = selected.weighted(weights).mean(dim=("latitude", "longitude"))
    if not np.all(np.isfinite(result)):
        raise InputError("The ERA5 wind-domain mean contains non-finite values.")
    return result


def plotting_modules():
    """Import optional plotting dependencies only when figures are requested."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
    except ImportError as error:
        raise InputError(
            "PLOT_ERA5 requires matplotlib and cartopy. Install both packages "
            "or set PLOT_ERA5 = False.") from error
    return plt, mdates, ccrs, cfeature, Patch, Rectangle


def exact_plot_coordinates(dataset):
    requested_time = np.datetime64(PLOT_TIME_UTC, "s")
    available_times = np.asarray(dataset.time).astype("datetime64[s]")
    time_matches = np.flatnonzero(available_times == requested_time)
    if time_matches.size != 1:
        raise InputError(
            f"PLOT_TIME_UTC={requested_time} is not an exact ERA5 timestamp; "
            f"available range is {available_times[0]} through {available_times[-1]}.")

    requested_pressure = float(PLOT_PRESSURE_PA)
    if not np.isfinite(requested_pressure) or requested_pressure <= 0:
        raise InputError("PLOT_PRESSURE_PA must be finite and positive.")
    available_pressure = pressure_values(dataset)
    level_matches = np.flatnonzero(np.isclose(
        available_pressure, requested_pressure, rtol=0.0, atol=0.1))
    if level_matches.size != 1:
        nearest = available_pressure[
            np.argmin(np.abs(available_pressure - requested_pressure))]
        raise InputError(
            f"PLOT_PRESSURE_PA={requested_pressure:g} Pa is not an ERA5 "
            f"pressure level; the nearest available level is {nearest:g} Pa.")
    return (int(time_matches[0]), requested_time,
            int(level_matches[0]), requested_pressure)


def geographic_subset(data, bounds, description):
    lon_min, lon_max, lat_min, lat_max = bounds
    selected = data.sel(
        longitude=slice(lon_min, lon_max),
        latitude=slice(lat_min, lat_max))
    if (selected.sizes.get("longitude", 0) < 2
            or selected.sizes.get("latitude", 0) < 2):
        raise InputError(
            f"{description} contains fewer than two ERA5 points in either "
            "horizontal direction and cannot be plotted.")
    return selected


def plot_era5_domains(dataset, output_directory):
    """Plot ERA5 coverage, the forcing domain, and the wind-mean domain."""
    plt, _, ccrs, cfeature, Patch, Rectangle = plotting_modules()
    time_index, plot_time, level_index, plot_pressure = (
        exact_plot_coordinates(dataset))
    selected = dataset[["u", "v"]].isel(
        time=time_index, level=level_index)
    target_bounds = (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    wind_bounds = (
        WIND_MEAN_LON_MIN, WIND_MEAN_LON_MAX,
        WIND_MEAN_LAT_MIN, WIND_MEAN_LAT_MAX)
    panels = (
        ("Complete ERA5 coverage", selected, None),
        ("VVM forcing target domain",
         geographic_subset(selected, target_bounds, "Target domain"),
         target_bounds),
        ("Wind-averaging domain",
         geographic_subset(selected, wind_bounds, "Wind-mean domain"),
         wind_bounds),
    )
    speed_full = np.hypot(selected.u, selected.v)
    vmin, vmax = float(speed_full.min()), float(speed_full.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise InputError("ERA5 map winds contain non-finite values.")
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    projection = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        1, 3, figsize=(20, 7), constrained_layout=True,
        subplot_kw={"projection": projection})
    mesh = None
    for axis, (title, data, bounds) in zip(axes, panels):
        longitude, latitude = np.asarray(data.longitude), np.asarray(data.latitude)
        u, v = np.asarray(data.u), np.asarray(data.v)
        mesh = axis.pcolormesh(
            longitude, latitude, np.hypot(u, v), shading="auto",
            vmin=vmin, vmax=vmax, cmap="viridis", transform=projection)
        skip = max(1, int(np.ceil(max(longitude.size, latitude.size) / 24.0)))
        axis.quiver(
            longitude[::skip], latitude[::skip],
            u[::skip, ::skip], v[::skip, ::skip],
            transform=projection, scale=450, width=0.0025)
        axis.coastlines(resolution="50m", linewidth=0.8)
        axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
        gridlines = axis.gridlines(
            draw_labels=True, linewidth=0.4, color="gray", alpha=0.5,
            linestyle="--")
        gridlines.top_labels = False
        gridlines.right_labels = False
        if bounds is None:
            axis.set_extent(
                [float(selected.longitude.min()), float(selected.longitude.max()),
                 float(selected.latitude.min()), float(selected.latitude.max())],
                crs=projection)
            for rectangle_bounds, color, linestyle in (
                    (target_bounds, "red", "-"),
                    (wind_bounds, "darkorange", "--")):
                lon_min, lon_max, lat_min, lat_max = rectangle_bounds
                axis.add_patch(Rectangle(
                    (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                    fill=False, edgecolor=color, linewidth=2.2,
                    linestyle=linestyle, transform=projection))
        else:
            lon_min, lon_max, lat_min, lat_max = bounds
            axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
        axis.set_title(title)

    axes[0].legend(
        handles=[
            Patch(facecolor="none", edgecolor="red", label="VVM forcing target"),
            Patch(facecolor="none", edgecolor="darkorange", linestyle="--",
                  label="Wind average"),
        ],
        loc="lower left")
    colorbar = figure.colorbar(
        mesh, ax=axes, orientation="horizontal", pad=0.08, shrink=0.72)
    colorbar.set_label("Horizontal wind speed (m s$^{-1}$)")
    figure.suptitle(
        f"ERA5 winds at {plot_pressure / 100.0:g} hPa, "
        f"{np.datetime_as_string(plot_time, unit='m')} UTC")
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "era5_domains.png"
    try:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    except OSError as error:
        raise InputError(f"Cannot save ERA5 domain figure '{path}': {error}") from error
    finally:
        plt.close(figure)
    print(f"Wrote {path}")
    return path


def read_generated_winds(paths):
    """Read exact time, pressure, U, and V values from generated forcing files."""
    times, u_profiles, v_profiles = [], [], []
    pressure = None
    for path in paths:
        try:
            with netCDF4.Dataset(path) as forcing:
                for name in ("pbar", "U", "V"):
                    if name not in forcing.variables:
                        raise InputError(
                            f"Generated forcing file '{path}' is missing '{name}'.")
                if "time" in forcing.variables:
                    time_variable = forcing.variables["time"]
                    seconds = float(np.asarray(time_variable[:]).reshape(-1)[0])
                    units = getattr(time_variable, "units", "")
                    if not units.startswith("seconds since 1970-01-01"):
                        raise InputError(
                            f"Generated forcing file '{path}' has unsupported "
                            f"time units {units!r}.")
                    current_time = (
                        np.datetime64("1970-01-01T00:00:00", "s")
                        + np.timedelta64(round(seconds), "s"))
                elif hasattr(forcing, "valid_time_utc"):
                    current_time = np.datetime64(forcing.valid_time_utc, "s")
                else:
                    raise InputError(
                        f"Generated forcing file '{path}' has neither a CF time "
                        "coordinate nor the legacy 'valid_time_utc' attribute.")
                current_pressure = np.asarray(forcing.variables["pbar"][:], dtype="f8")
                current_u = np.asarray(forcing.variables["U"][:], dtype="f8")
                current_v = np.asarray(forcing.variables["V"][:], dtype="f8")
        except OSError as error:
            raise InputError(
                f"Cannot read generated forcing file '{path}': {error}") from error
        if np.isnat(current_time):
            raise InputError(f"Generated forcing file '{path}' has invalid time metadata.")
        if pressure is None:
            pressure = current_pressure
        elif (current_pressure.shape != pressure.shape
              or not np.allclose(current_pressure, pressure, rtol=0.0, atol=1.0e-6)):
            raise InputError(
                f"Generated forcing file '{path}' has incompatible pressure levels.")
        if current_u.shape != pressure.shape or current_v.shape != pressure.shape:
            raise InputError(
                f"Generated forcing file '{path}' has incompatible U/V profiles.")
        if not all(np.all(np.isfinite(values))
                   for values in (current_pressure, current_u, current_v)):
            raise InputError(
                f"Generated forcing file '{path}' has non-finite wind data.")
        times.append(current_time)
        u_profiles.append(current_u)
        v_profiles.append(current_v)
    if not times:
        raise InputError("No generated forcing files are available for wind plotting.")
    order = np.argsort(np.asarray(times))
    return (
        np.asarray(times)[order], pressure,
        np.asarray(u_profiles, dtype="f8")[order],
        np.asarray(v_profiles, dtype="f8")[order])


def plot_wind_profiles(paths, output_directory):
    """Plot the written wind-average profiles in time-pressure coordinates."""
    plt, mdates, _, _, _, _ = plotting_modules()
    times, pressure, u_profiles, v_profiles = read_generated_winds(paths)
    time_skip = max(1, int(np.ceil(times.size / 42.0)))
    level_skip = max(1, int(np.ceil(pressure.size / 28.0)))
    local_times = times[::time_skip] + np.timedelta64(8, "h")
    plot_times = [
        value.astype("datetime64[ms]").astype(object) for value in local_times]
    plot_pressure = pressure[::level_skip] / 100.0
    plot_u = u_profiles[::time_skip, ::level_skip]
    plot_v = v_profiles[::time_skip, ::level_skip]
    wind_speed = np.hypot(plot_u, plot_v)
    time_grid, pressure_grid = np.meshgrid(
        mdates.date2num(plot_times), plot_pressure, indexing="ij")

    figure, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    quiver = axis.quiver(
        time_grid, pressure_grid, plot_u, plot_v, wind_speed,
        cmap="viridis", angles="uv", scale_units="inches", scale=45.0,
        pivot="middle", width=0.0025, headwidth=3.5, headlength=4.5,
        headaxislength=4.0)
    colorbar = figure.colorbar(quiver, ax=axis, pad=0.02)
    colorbar.set_label("Horizontal wind speed (m s$^{-1}$)")
    axis.quiverkey(
        quiver, X=0.88, Y=1.035, U=10.0,
        label="10 m s$^{-1}$", labelpos="E", coordinates="axes")
    axis.set_yscale("log")
    axis.invert_yaxis()
    standard_ticks = np.asarray(
        [1000, 850, 700, 500, 300, 200, 100, 50, 20], dtype="f8")
    within_range = (
        (standard_ticks <= float(np.max(pressure) / 100.0))
        & (standard_ticks >= float(np.min(pressure) / 100.0)))
    axis.set_yticks(standard_ticks[within_range])
    axis.set_yticklabels([f"{value:g}" for value in standard_ticks[within_range]])
    axis.xaxis_date()
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axis.set_xlabel("Taiwan local time (UTC+8)")
    axis.set_ylabel("VVM pressure (hPa)")
    axis.set_title(
        "ERA5 horizontal wind profiles used for VVMex nudging\n"
        f"cosine-latitude average: "
        f"{WIND_MEAN_LON_MIN:g}–{WIND_MEAN_LON_MAX:g}°E, "
        f"{WIND_MEAN_LAT_MIN:g}–{WIND_MEAN_LAT_MAX:g}°N")
    axis.grid(which="major", linewidth=0.6, alpha=0.35)
    axis.grid(which="minor", axis="x", linewidth=0.3, alpha=0.2)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "era5_averaged_wind_profiles.png"
    try:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    except OSError as error:
        raise InputError(
            f"Cannot save ERA5 wind-profile figure '{path}': {error}") from error
    finally:
        plt.close(figure)
    print(f"Wrote {path}")
    return path


def interpolate_column(values, source_pressure, target_pressure, log_values):
    valid = np.isfinite(values) & np.isfinite(source_pressure) & (source_pressure > 0)
    if np.count_nonzero(valid) < 2:
        return np.full(target_pressure.shape, np.nan)
    pressure, values = source_pressure[valid], np.asarray(values)[valid]
    order = np.argsort(pressure)
    pressure, values = pressure[order], values[order]
    if log_values:
        values = np.log(np.maximum(values, 1.0e-12))
    result = interpolate_extrapolate(
        np.log(target_pressure), np.log(pressure), values)
    return np.exp(result) if log_values else result


def vertical_interpolate(data, source_pressure, target_pressure, log_values=False):
    return xr.apply_ufunc(
        interpolate_column, data, xr.DataArray(source_pressure, dims="level"),
        xr.DataArray(target_pressure, dims="nz"),
        input_core_dims=[["level"], ["level"], ["nz"]], output_core_dims=[["nz"]],
        vectorize=True, dask="forbidden", kwargs={"log_values": log_values},
        output_dtypes=[np.float64])


def define_idealized_forcing(config, target_pressure, nx, ny, elapsed_seconds):
    """Define idealized large-scale fields from the external VVMex sounding.

    Edit this function to add horizontal structure or time dependence. The
    default is horizontally uniform and uses the configured profile file;
    elapsed_seconds is available for time-varying experiments.
    """
    del elapsed_seconds  # Remove when defining time-dependent idealized fields.
    constants = require(config, "constants", "configuration")
    initial = require(config, "initial_conditions", "configuration")
    profile, _ = read_profile(require(initial, "source_file", "initial_conditions"))
    source_pressure = profile["pbar"]
    p0 = float(require(constants, "P0", "constants"))
    rd = float(require(constants, "Rd", "constants"))
    cp = float(require(constants, "Cp", "constants"))
    source_exner = np.power(source_pressure / p0, rd / cp)
    source_temperature = (
        profile["Tbar"] if "Tbar" in profile else profile["thbar"] * source_exner)

    temperature = interpolate_column(
        source_temperature, source_pressure, target_pressure, False)
    qvbar = np.maximum(interpolate_column(
        profile["qvbar"], source_pressure, target_pressure, False), 0.0)
    winds = {}
    for output_name, profile_name in (("U", "U"), ("V", "V")):
        source = profile.get(profile_name, np.zeros(source_pressure.size))
        winds[output_name] = interpolate_column(
            source, source_pressure, target_pressure, False)

    qv = np.broadcast_to(qvbar[:, None, None], (target_pressure.size, ny, nx)).copy()
    theta_profile = temperature * np.power(p0 / target_pressure, rd / cp)
    theta = np.broadcast_to(
        theta_profile[:, None, None], (target_pressure.size, ny, nx)).copy()
    return {
        "qv": qv, "th": theta, "pbar": target_pressure,
        "Tbar": temperature, "qvbar": qvbar,
        "U": winds["U"], "V": winds["V"],
    }


def forcing_state(dataset, time, target_pressure, longitude, latitude, constants):
    try:
        selected = dataset.sel(time=time)
    except KeyError as error:
        times = np.asarray(dataset.time)
        raise InputError(
            f"ERA5 has no exact timestamp {time}; available range is "
            f"{times[0]} through {times[-1]}.") from error
    specific_humidity = selected.specific_humidity
    humidity_min = float(specific_humidity.min())
    humidity_max = float(specific_humidity.max())
    if humidity_min < 0 or humidity_max >= 1:
        warnings.warn(
            f"ERA5 specific humidity is outside [0,1) at {time} "
            f"(min={humidity_min:.6g}, max={humidity_max:.6g}); "
            "out-of-range values will be clipped and processing will continue.",
            RuntimeWarning,
            stacklevel=2)
        specific_humidity = specific_humidity.astype("f8").clip(
            min=0.0, max=1.0 - 1.0e-12)
    source_pressure = pressure_values(dataset)
    raw = {
        "T": vertical_interpolate(selected.temperature, source_pressure, target_pressure),
        "qv": vertical_interpolate(specific_humidity / (1 - specific_humidity),
                                   source_pressure, target_pressure, True),
    }
    arrays = {}
    for name, data in raw.items():
        arrays[name] = np.asarray(data.interp(
            longitude=longitude, latitude=latitude).transpose("nz", "ny", "nx"))
        if not np.all(np.isfinite(arrays[name])):
            raise InputError(f"Interpolated ERA5 field '{name}' has missing values at {time}.")
    wind_profiles = {
        "U": np.asarray(vertical_interpolate(
            wind_domain_mean(selected.u), source_pressure, target_pressure)),
        "V": np.asarray(vertical_interpolate(
            wind_domain_mean(selected.v), source_pressure, target_pressure)),
    }
    for name, values in wind_profiles.items():
        if values.shape != target_pressure.shape or not np.all(np.isfinite(values)):
            raise InputError(f"ERA5 wind-domain profile '{name}' is invalid at {time}.")
    p0, rd, cp = (float(require(constants, name, "constants"))
                  for name in ("P0", "Rd", "Cp"))
    return {
        "qv": arrays["qv"],
        "th": arrays["T"] * np.power(p0 / target_pressure[:, None, None], rd / cp),
        "pbar": target_pressure, "Tbar": arrays["T"].mean((1, 2)),
        "qvbar": arrays["qv"].mean((1, 2)), "U": wind_profiles["U"],
        "V": wind_profiles["V"],
    }


def write_forcing(path, data, timestamp, elapsed, longitude, latitude):
    nz, ny, nx = data["qv"].shape
    compression = ({"zlib": True, "complevel": COMPRESSION_LEVEL, "shuffle": True}
                   if USE_COMPRESSION else {})
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as output:
        for name, size in (("time", 1), ("nz", nz), ("ny", ny), ("nx", nx)):
            output.createDimension(name, size)
        for name, values, units, standard_name, axis in (
                ("nx", longitude, "degrees_east", "longitude", "X"),
                ("ny", latitude, "degrees_north", "latitude", "Y"),
                ("nz", data["pbar"], "Pa", "air_pressure", "Z")):
            variable = output.createVariable(name, "f8", (name,))
            variable[:], variable.units = values, units
            variable.standard_name, variable.axis = standard_name, axis
        output.variables["nz"].positive = "down"
        time = output.createVariable("time", "f8", ("time",))
        time[0] = float((timestamp - np.datetime64("1970-01-01T00:00:00"))
                        / np.timedelta64(1, "s"))
        time.units, time.calendar = ("seconds since 1970-01-01 00:00:00 UTC",
                                     "proleptic_gregorian")
        time.standard_name, time.axis = "time", "T"
        metadata = {
            "qv": ("kg kg-1", "water vapor mixing ratio"),
            "th": ("K", "air potential temperature"),
            "pbar": ("Pa", "VVMex reference-state air pressure"),
            "Tbar": ("K", "ERA5 horizontal-mean air temperature"),
            "qvbar": ("kg kg-1", "ERA5 horizontal-mean water vapor mixing ratio"),
            "U": ("m s-1", "ERA5 selected-domain mean eastward wind"),
            "V": ("m s-1", "ERA5 selected-domain mean northward wind"),
        }
        for name in ("qv", "th"):
            variable = output.createVariable(
                name, OUTPUT_DTYPE, ("nz", "ny", "nx"), **compression)
            variable[:], variable.units, variable.long_name = data[name], *metadata[name]
        for name in ("pbar", "Tbar", "qvbar", "U", "V"):
            variable = output.createVariable(name, OUTPUT_DTYPE, ("nz",))
            variable[:], variable.units, variable.long_name = data[name], *metadata[name]
        output.Conventions = "CF-1.8"
        output.title = "VVMex ERA5 nudging and large-scale forcing"
        output.source = "Local ERA5 pressure-level temperature, humidity, and wind"
        output.history = "Generated by tools/generate_ls_forcing.py"
        output.time_seconds = int(elapsed)


def forcing_settings(config):
    grid = require(config, "grid", "configuration")
    nx, ny = int(require(grid, "nx", "grid")), int(require(grid, "ny", "grid"))
    if nx < 2 or ny < 2:
        raise InputError("The forcing grid requires nx and ny of at least two.")
    try:
        forcing = config["dynamics"]["forcings"]["lateral_boundary_nudging"]["forcing_data"]
    except KeyError as error:
        raise InputError("Missing lateral_boundary_nudging.forcing_data in the config.") from error
    directory = resolve_path(forcing.get("directory", "./rundata/LS_forcings/"))
    if bool(forcing.get("time_varying", False)):
        interval = float(forcing.get("update_interval_s", 3600.0))
        if not np.isfinite(interval) or interval <= 0 or not interval.is_integer():
            raise InputError("forcing_data.update_interval_s must be a positive whole number.")
        naming = {"time_varying": True, "interval": int(interval),
                  "prefix": forcing.get("file_prefix", "ls_forcing_")}
    else:
        naming = {"time_varying": False, "constant_name": forcing.get(
            "file_name_for_not_varying", "ls_forcing_constant.nc")}
    return nx, ny, directory, naming


def requested_times(naming):
    start, end = np.datetime64(START_TIME_UTC, "s"), np.datetime64(END_TIME_UTC, "s")
    if np.isnat(start) or np.isnat(end) or end < start:
        raise InputError("START_TIME_UTC and END_TIME_UTC define an invalid range.")
    if not naming["time_varying"]:
        return np.asarray([start]), np.asarray([0], dtype="i8")
    duration, interval = int((end - start) / np.timedelta64(1, "s")), naming["interval"]
    if duration % interval:
        raise InputError(
            f"Time range ({duration} s) is not divisible by forcing interval ({interval} s).")
    elapsed = np.arange(0, duration + 1, interval, dtype="i8")
    return start + elapsed * np.timedelta64(1, "s"), elapsed


_WORKER_CONTEXT = None


def initialize_worker(context):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def process_forcing_time(task):
    """Generate and write one timestamp; called in a worker process."""
    timestamp, seconds = task
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("ERA5 worker context was not initialized.")
    naming = context["naming"]
    filename = (f"{naming['prefix']}{seconds:06d}.nc"
                if naming["time_varying"] else naming["constant_name"])
    path = context["output_directory"] / filename
    if context["mode"] == "era5":
        data = forcing_state(
            context["dataset"], timestamp, context["target_pressure"],
            context["longitude"], context["latitude"], context["constants"])
    else:
        data = define_idealized_forcing(
            context["config"], context["target_pressure"],
            context["nx"], context["ny"], seconds)
    write_forcing(
        path, data, timestamp, seconds,
        context["longitude_values"], context["latitude_values"])
    return str(path)


def main():
    try:
        config, config_path = read_config()
        target_pressure, profile_path = calculate_vvm_pressure(config)
        nx, ny, output_directory, naming = forcing_settings(config)
        times, elapsed = requested_times(naming)
        longitude_values = np.linspace(LON_MIN, LON_MAX, nx)
        latitude_values = np.linspace(LAT_MIN, LAT_MAX, ny)
        longitude = xr.DataArray(longitude_values, dims="nx")
        latitude = xr.DataArray(latitude_values, dims="ny")
        dataset = load_era5() if USE_ERA5_FORCING else None
        try:
            if USE_ERA5_FORCING:
                check_domains(dataset)
            output_directory.mkdir(parents=True, exist_ok=True)
            print(f"VVMex config:  {config_path}")
            print(f"VVMex profile: {profile_path}")
            print(f"Target pressure: {target_pressure.size} physical layers, "
                  f"{target_pressure[0]:g}..{target_pressure[-1]:g} Pa")
            if USE_ERA5_FORCING:
                print(f"Mode: ERA5")
                print(f"Target domain: lon={LON_MIN:g}..{LON_MAX:g}, "
                      f"lat={LAT_MIN:g}..{LAT_MAX:g}, grid={nx}x{ny}")
                print(
                    f"Wind-mean domain: lon={WIND_MEAN_LON_MIN:g}..{WIND_MEAN_LON_MAX:g}, "
                    f"lat={WIND_MEAN_LAT_MIN:g}..{WIND_MEAN_LAT_MAX:g}")
            else:
                print(f"Mode: idealized ({nx}x{ny})")
            if not isinstance(PROCESS_COUNT, int) or PROCESS_COUNT < 1:
                raise InputError("PROCESS_COUNT must be a positive integer.")
            tasks = [(np.datetime64(timestamp, "s"), int(seconds))
                     for timestamp, seconds in zip(times, elapsed)]
            context = {
                "mode": "era5" if USE_ERA5_FORCING else "idealized",
                "dataset": dataset, "target_pressure": target_pressure,
                "longitude": longitude, "latitude": latitude,
                "longitude_values": longitude_values,
                "latitude_values": latitude_values,
                "constants": require(config, "constants", "configuration"),
                "config": config, "nx": nx, "ny": ny,
                "output_directory": output_directory, "naming": naming,
            }
            worker_count = min(PROCESS_COUNT, len(tasks))
            print(f"Processing {len(tasks)} ERA5 time(s) with {worker_count} process(es)")
            generated_paths = []
            if worker_count == 1:
                initialize_worker(context)
                for path in map(process_forcing_time, tasks):
                    generated_paths.append(Path(path))
                    print(f"Wrote {path}")
            else:
                # fork shares the loaded, read-only ERA5 arrays copy-on-write;
                # each process creates only its own timestamp's output arrays.
                mp_context = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(
                        max_workers=worker_count, mp_context=mp_context,
                        initializer=initialize_worker, initargs=(context,)) as executor:
                    for path in executor.map(process_forcing_time, tasks, chunksize=1):
                        generated_paths.append(Path(path))
                        print(f"Wrote {path}")
            if PLOT_ERA5:
                if USE_ERA5_FORCING:
                    plot_directory = output_directory / PLOT_SUBDIRECTORY
                    plot_era5_domains(dataset, plot_directory)
                    plot_wind_profiles(generated_paths, plot_directory)
                else:
                    print("Skipping ERA5 visualization in idealized forcing mode.")
        finally:
            if dataset is not None:
                dataset.close()
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
