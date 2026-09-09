#!/usr/bin/env python3
"""Validate and plot an existing RLL mountain run without launching the model."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"tests/scripts"))
from check_rll_mountain import check, read

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path)
args = parser.parse_args()
config = json.loads(args.config.read_text())
directory = Path(config["output"]["output_dir"])
if not directory.is_absolute():
    directory = ROOT/directory
prefix = config["output"]["output_filename_prefix"]
paths = sorted(directory.glob(prefix+"_*.h5"))
assert paths, "No model snapshots"
records = []
initial = None
for path in paths:
    data = read(path)
    check(data, config)
    if initial is None:
        initial = data
    solid = data["ITYPEW"] == 0
    records.append({"file": path.name, "time_s": float(data["model_time_s"].squeeze()),
        "max_abs_w_m_s": float(np.max(np.abs(data["w"]))),
        "max_abs_raw_solid_w_m_s": float(np.max(np.abs(data["w"][solid]))) if solid.any() else 0.,
        "max_abs_masked_solid_w_m_s": float(np.max(np.abs(data["w_topo"][solid]))) if solid.any() else 0.,
        "south_circulation_drift_m_s": float(data["u"][-1, 0].mean()-initial["u"][-1, 0].mean())})
times = np.array([r["time_s"] for r in records])
expected = np.arange(0, config["simulation"]["total_time_s"]+1e-6, config["simulation"]["output_interval_s"])
np.testing.assert_allclose(times, expected, atol=1e-9, rtol=0)
assert max(abs(r["south_circulation_drift_m_s"]) for r in records) < 1e-11
(directory/"terrain_validation.json").write_text(json.dumps(records, indent=2)+"\n")
height = data["rll_terrain_height"]/1000.
lon, lat = data["lon"], data["lat"]
ny = lat.shape[0]
j = ny//2
dz = config["grid"]["vertical"]["dz"]
z = (np.arange(data["w"].shape[0])+1)*dz/1000.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
mesh = axes[0].pcolormesh(lon, lat, height, shading="auto")
fig.colorbar(mesh, ax=axes[0], label="terrain height (km)")
axes[0].set(xlabel="longitude (degrees)", ylabel="latitude (degrees)", title="Centered RLL mountain")
w = np.ma.masked_where(data["ITYPEW"][:, j] == 0, data["w_topo"][:, j])
limit = max(float(np.max(np.abs(w))), 1e-12)
mesh = axes[1].pcolormesh(lon[j], z, w, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
axes[1].fill_between(lon[j], 0, height[j], color="0.3")
fig.colorbar(mesh, ax=axes[1], label="masked vertical wind (m/s)")
axes[1].set(xlabel="longitude (degrees)", ylabel="height (km)", xlim=(150, 210),
    title=f"Latitude {lat[j,0]:.2f}°, t={times[-1]:g}s")
fig.savefig(directory/"terrain.png", dpi=160)
print(json.dumps(records, indent=2))
print(f"PASS: {directory/'terrain_validation.json'}; {directory/'terrain.png'}")
