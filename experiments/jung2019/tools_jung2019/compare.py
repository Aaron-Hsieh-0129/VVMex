#!/usr/bin/env python3
"""Compare a completed VVMex RLL run with CVVM's centered output fields."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np


def center_z(z):
    # Z is on positive faces; the south wall lies just outside the output
    # physical rows. The admitted free-slip policy sets that wall to zero.
    south = np.concatenate([np.zeros_like(z[:1]), z[:-1]], axis=0)
    return .25 * (z + np.roll(z, 1, axis=1) + south + np.roll(south, 1, axis=1))


def norms(value, reference, weights):
    error = value - reference
    return {"relative_l1": float(np.sum(weights * np.abs(error)) / np.sum(weights * np.abs(reference))),
            "relative_l2": float(np.sqrt(np.sum(weights * error**2) / np.sum(weights * reference**2))),
            "relative_linf": float(np.max(np.abs(error)) / np.max(np.abs(reference))),
            "absolute_max": float(np.max(np.abs(error))),
            "correlation": float(np.corrcoef(value.ravel(), reference.ravel())[0, 1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--hours", type=int, help="Compare a saved intermediate time")
    args = parser.parse_args()
    configuration = json.loads((args.run / "config.json").read_text())
    expected = args.hours * 3600 if args.hours is not None else configuration["simulation"]["total_time_s"]
    outputs = sorted((args.run / "output").glob("*.h5"))
    data = None
    for path in outputs:
        with h5py.File(path) as file:
            group = file["Step0"]
            if "model_time_s" in group and float(np.asarray(group["model_time_s"]).squeeze()) == expected:
                data = {name: np.asarray(group[name]) for name in ("zeta", "u", "v", "lon", "lat")}
                output_path = path
                break
    if data is None:
        raise ValueError("No output at configured final time; refusing stale/partial comparison")
    hour = int(expected / 3600)
    refs = list(args.reference.glob(f"*L.{hour:04d}_*.nc"))
    if len(refs) != 1:
        raise ValueError(f"Expected exactly one reference at {hour} h, found {len(refs)}")
    with Dataset(refs[0]) as file:
        # CODE_BAR/ldoutput_nc.F replaces lower z3dz with psi. Only upper
        # z3dz is vorticity. All three archived fields are centered at T.
        ref = {name: np.asarray(file[key][-1], dtype=float) for name, key in
               [("zeta", "z3dz"), ("u", "u3dx"), ("v", "u3dy")]}
        latitude = np.asarray(file["latt"])
        longitude = np.asarray(file["lont"])
    # CVVM archive columns begin at 225 degrees and wrap through zero.
    # Align using recorded coordinates, never a fitted pattern displacement.
    order = np.argsort(np.mod(longitude[0], 360.0))
    longitude = np.mod(longitude[:, order], 360.0)
    latitude = latitude[:, order]
    ref = {name: field[:, order] for name, field in ref.items()}
    if ref["zeta"].shape != data["zeta"].shape[-2:]:
        raise ValueError("Reference and model resolution differ; explicit conservative remapping is required")
    if not np.allclose(latitude, data["lat"], atol=5e-5, rtol=0) or not np.allclose(longitude, data["lon"], atol=5e-5, rtol=0):
        raise ValueError("Reference/model geographic coordinates disagree")
    u, v = data["u"][-1], data["v"][-1]
    # CVVM averages contravariant winds before mapping to physical at T.
    centered = {"zeta": center_z(data["zeta"][-1]), "u": .5 * (u + np.roll(u, 1, axis=1)),
                "v": .5 * (v + np.concatenate([np.zeros_like(v[:1]), v[:-1]], axis=0))}
    if not all(np.isfinite(field).all() for field in [*centered.values(), *ref.values()]):
        raise ValueError("Nonfinite model/reference fields cannot support an archive comparison")
    weights = np.cos(np.deg2rad(latitude))
    report = {"time_hours": hour, "reference": str(refs[0]), "output": str(output_path),
              "reference_sha256": hashlib.sha256(refs[0].read_bytes()).hexdigest(),
              "norms": {name: norms(centered[name], ref[name], weights) for name in ref},
              "acceptance": "Descriptive only: the supplied CVVM uniform-copy lateral halos differ from the paper-wall VVMex configuration. Scientific self-convergence is assessed separately against each case's own high-resolution reference."}
    basename = "comparison" if args.hours is None else f"comparison_{hour:04d}h"
    (args.run / f"{basename}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    scale = np.max(np.abs(ref["zeta"]))
    for ax, field, title in zip(axes, [ref["zeta"], centered["zeta"], centered["zeta"]-ref["zeta"]],
                                 ["CVVM reference (T points)", "VVMex (Z averaged to T)", "VVMex minus CVVM"]):
        mesh = ax.pcolormesh(longitude, latitude, field, cmap="RdBu_r", vmin=-scale, vmax=scale, shading="nearest")
        ax.set(title=title, ylabel="Latitude (degrees north)", ylim=(-30, 30))
        fig.colorbar(mesh, ax=ax, label="Relative vorticity (s⁻¹)")
    axes[-1].set_xlabel("Longitude (degrees east)")
    fig.suptitle(f"Jung Section 4.2 CASE {configuration['initial_conditions']['jung2019']['case']}, {hour} h")
    fig.savefig(args.run / f"{basename}.png", dpi=150)
    fig.savefig(args.run / f"{basename}_preview.png", dpi=65)


if __name__ == "__main__":
    main()
