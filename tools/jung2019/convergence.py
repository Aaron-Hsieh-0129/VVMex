#!/usr/bin/env python3
"""Jung Section 4.2: all resolutions against their own 6.25 km reference."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment import read_zeta


def configuration(directory):
    return json.loads((directory / "config.json").read_text())


def outputs(directory, end_time):
    result = {}
    for path in sorted((directory / "output").glob("jung_*.h5")):
        with h5py.File(path) as file:
            time = float(np.asarray(file["Step0/model_time_s"]).squeeze())
        if time in result:
            raise ValueError(f"Duplicate output time in {directory}")
        result[time] = path
    if end_time not in result:
        raise ValueError(f"Missing configured final output in {directory}")
    return result


def field(path):
    with h5py.File(path) as file:
        value = read_zeta(file["Step0"])
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite native vorticity: {path}")
    return value


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--runs", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ref_config = configuration(args.reference)
ref_grid = ref_config["grid"]["horizontal"]
if (ref_grid["nx"], ref_grid["ny"]) != (6400, 1600):
    raise ValueError("The paper's five-level study requires the 6400×1600 reference")
case = ref_config["initial_conditions"]["jung2019"]["case"]
end_time = ref_config["simulation"]["total_time_s"]
if end_time != (168 if case == 1 else 120)*3600:
    raise ValueError("Reference does not reach the paper's comparison time")
reference = outputs(args.reference, end_time)
runs = sorted(args.runs, key=lambda p: configuration(p)["grid"]["horizontal"]["nx"])
configs = [configuration(p) for p in runs]
if [c["grid"]["horizontal"]["nx"] for c in configs] != [400, 800, 1600, 3200]:
    raise ValueError("Expected all four coarser paper resolutions")
histories = []
for directory, config in zip(runs, configs):
    grid = config["grid"]["horizontal"]
    if config["initial_conditions"] != ref_config["initial_conditions"] or grid["geometry"] != ref_grid["geometry"]:
        raise ValueError("Reference and coarse cases or geometry differ")
    if config["simulation"]["total_time_s"] != end_time or grid["ny"]*4 != grid["nx"]:
        raise ValueError("Incompatible coarse-grid domain or duration")
    histories.append(outputs(directory, end_time))
times = sorted(reference)
if any(set(history) != set(times) for history in histories):
    raise ValueError("All resolutions must have identical saved comparison times")
errors = [[] for _ in runs]
for time in times:
    fine = field(reference[time])
    for index, (history, config) in enumerate(zip(histories, configs)):
        grid = config["grid"]["horizontal"]
        ratio = ref_grid["nx"] // grid["nx"]
        target = fine[ratio-1::ratio, ratio-1::ratio]
        value = field(history[time])
        if value.shape != target.shape:
            raise ValueError("Native grid shapes do not align")
        south, north = np.deg2rad(grid["geometry"]["latitude_bounds_deg"])
        latitude = south + (np.arange(grid["ny"])+1)*(north-south)/grid["ny"]
        weights = np.cos(latitude)[:, None]
        error = float(np.sqrt(np.sum(weights*(value-target)**2)/np.sum(weights*target**2)))
        errors[index].append(error)
matrix = np.asarray(errors)
monotone = np.all(matrix[:-1] > matrix[1:], axis=0)
rates = np.log2(matrix[:-1]/matrix[1:])
report = {"case": case, "reference": str(args.reference), "reference_configuration": ref_config,
          "method": "Native coincident-Z sampling; full-band cos(latitude)-weighted relative L2; no fitted shift or smoothing.",
          "hours": [time/3600 for time in times],
          "runs": [{"directory": str(p), "nx": c["grid"]["horizontal"]["nx"], "dt_s": c["simulation"]["dt_s"], "relative_l2": e}
                   for p, c, e in zip(runs, configs, errors)],
          "observed_adjacent_rates": rates.tolist(), "monotone_at_each_output": monotone.tolist(),
          "decreasing_final_l2": bool(monotone[-1]),
          "scope": "Joint spatial/time-discretization self-convergence, not identity with the CVVM uniform-copy-boundary archive."}
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "convergence.json").write_text(json.dumps(report, indent=2)+"\n")
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
for config, error in zip(configs, errors):
    nx = config["grid"]["horizontal"]["nx"]
    dx = 2*np.pi*ref_grid["geometry"]["earth_radius_m"]/nx/1000
    ax.semilogy(report["hours"], error, marker="o", label=f"{dx:.2f} km (nx={nx})")
ax.set(xlabel="Time (hours)", ylabel="Relative L2 of native vertical vorticity", title=f"CASE {case}: reference nx=6400 (~6.25 km)")
ax.grid(True, which="both", alpha=.25)
ax.legend()
fig.savefig(args.output / "l2_history.png", dpi=160)
plt.close(fig)
final_fields = [field(history[end_time]) for history in histories] + [field(reference[end_time])]
scale = float(np.max(np.abs(final_fields[-1])))
fig, axes = plt.subplots(5, 1, figsize=(12, 12), constrained_layout=True)
for ax, value, config in zip(axes, final_fields, configs+[ref_config]):
    grid = config["grid"]["horizontal"]
    west, east = grid["geometry"]["longitude_bounds_deg"]
    south, north = grid["geometry"]["latitude_bounds_deg"]
    dx, dy = (east-west)/grid["nx"], (north-south)/grid["ny"]
    mesh = ax.imshow(value, origin="lower", interpolation="nearest", aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale,
                     extent=(west+.5*dx, east+.5*dx, south+.5*dy, north+.5*dy))
    ax.set(xlim=(west, east), ylim=(-30, 30), ylabel="Latitude (°N)", title=f"{grid['nx']}×{grid['ny']}, dt={config['simulation']['dt_s']:g} s")
    fig.colorbar(mesh, ax=ax, label="Relative vorticity (s⁻¹)")
axes[-1].set_xlabel("Longitude (°E)")
fig.suptitle(f"CASE {case}, {end_time/3600:g} h — native Z fields; L2 uses the entire ±45° band")
fig.savefig(args.output / "final_vorticity.png", dpi=150)
fig.savefig(args.output / "final_vorticity_preview.png", dpi=55)
print(json.dumps({"case": case, "final_l2": matrix[:, -1].tolist(), "decreasing_final_l2": bool(monotone[-1])}, indent=2))
raise SystemExit(0 if monotone[-1] else 1)
