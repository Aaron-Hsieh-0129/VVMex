#!/usr/bin/env python3
"""Compare nested native-Z grids without phase fitting or latitude cropping.

The fine field is sampled at exactly coincident coarse Z points. Report the
cos(latitude)-weighted relative L2 norm; no fitted acceptance threshold.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiment import read_zeta


def snapshots(directory):
    result = {}
    for path in sorted((directory / "output").glob("jung_*.h5")):
        with h5py.File(path) as file:
            data = file["Step0"]
            result[float(np.asarray(data["model_time_s"]).squeeze())] = path
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("coarse", type=Path)
parser.add_argument("fine", type=Path)
args = parser.parse_args()
configs = [json.loads((p / "config.json").read_text()) for p in (args.coarse, args.fine)]
if configs[0]["initial_conditions"] != configs[1]["initial_conditions"]:
    raise ValueError("Initial conditions differ")
grids = [c["grid"]["horizontal"] for c in configs]
if grids[0]["geometry"] != grids[1]["geometry"]:
    raise ValueError("Domains differ")
rx, ry = grids[1]["nx"] // grids[0]["nx"], grids[1]["ny"] // grids[0]["ny"]
if rx < 1 or ry < 1 or grids[1]["nx"] % grids[0]["nx"] or grids[1]["ny"] % grids[0]["ny"]:
    raise ValueError("Grids must be integer nested")
coarse, fine = snapshots(args.coarse), snapshots(args.fine)
end_time = configs[0]["simulation"]["total_time_s"]
if end_time != configs[1]["simulation"]["total_time_s"] or end_time not in coarse or end_time not in fine:
    raise ValueError("Both runs must reach the same configured final time")
south, north = np.deg2rad(grids[0]["geometry"]["latitude_bounds_deg"])
latitude = south + (np.arange(grids[0]["ny"]) + 1) * (north-south)/grids[0]["ny"]
weights = np.cos(latitude)[:, None]
rows = []
for time in sorted(coarse.keys() & fine.keys()):
    with h5py.File(coarse[time]) as a, h5py.File(fine[time]) as b:
        low = read_zeta(a["Step0"])
        fine_z = read_zeta(b["Step0"])
        high = fine_z[ry-1::ry, rx-1::rx]
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Nonfinite vorticity")
    error = low-high
    rows.append({"hours": time/3600, "relative_l2": float(np.sqrt(np.sum(weights*error**2)/np.sum(weights*high**2))),
                 "absolute_max": float(np.max(np.abs(error)))})
report = {"coarse": str(args.coarse), "fine": str(args.fine), "method": __doc__, "results": rows,
          "acceptance": "Descriptive only; a pairwise difference does not establish the paper's five-level convergence claim."}
destination = args.coarse / f"difference_from_{args.fine.name}.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
scale = max(np.max(np.abs(low)), np.max(np.abs(fine_z)))
for ax, field, grid, directory in zip(axes, (low, fine_z), grids, (args.coarse, args.fine)):
    west, east = grid["geometry"]["longitude_bounds_deg"]
    bottom, top = grid["geometry"]["latitude_bounds_deg"]
    longitude = west + (np.arange(grid["nx"])+1) * (east-west)/grid["nx"]
    latitude = bottom + (np.arange(grid["ny"])+1) * (top-bottom)/grid["ny"]
    mesh = ax.pcolormesh(longitude, latitude, field, cmap="RdBu_r", vmin=-scale, vmax=scale, shading="nearest")
    ax.set(title=f"{directory.name}, {end_time/3600:g} h", xlim=(west, east), ylim=(-30, 30), ylabel="Latitude (degrees north)")
    fig.colorbar(mesh, ax=ax, label="Relative vertical vorticity (s⁻¹)")
axes[-1].set_xlabel("Longitude (degrees east)")
fig.savefig(destination.with_suffix(".png"), dpi=150)
