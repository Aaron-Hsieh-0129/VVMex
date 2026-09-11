#!/usr/bin/env python3
"""Integrated RLL terrain checks; tolerances bound roundoff, not a benchmark fit."""
import argparse
import copy
import json
from pathlib import Path
import resource
import subprocess
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("u", "v", "w", "xi", "eta", "zeta")


def read(path):
    with h5py.File(path) as f:
        return {k: np.asarray(v) for k, v in f["Step0"].items()}


def check(data, config):
    for name in FIELDS + ("u_topo", "v_topo", "w_topo", "xi_topo", "eta_topo"):
        assert np.isfinite(data[name]).all(), f"nonfinite {name}"
    height = data["rll_terrain_height"]
    dz = config["grid"]["vertical"]["dz"]
    nz, ny, nx = data["w"].shape
    radius = config["grid"]["horizontal"]["geometry"]["earth_radius_m"]
    lon, lat = np.deg2rad(data["lon"]), np.deg2rad(data["lat"])
    distance = radius*np.arccos(np.clip(np.cos(lat)*np.cos(lon-np.pi), -1, 1))
    mountain = config["initial_conditions"]["rll_mountain"]
    expected_height = np.where(distance < 3*mountain["half_width_m"],
        mountain["height_m"]*np.exp(-(distance/mountain["half_width_m"])**2), 0)
    np.testing.assert_array_equal(height, np.rint(expected_height/dz)*dz)
    mw = ((np.arange(nz)[:, None, None]+1)*dz > height).astype(float)
    np.testing.assert_array_equal(data["ITYPEW"], mw)
    np.testing.assert_array_equal(data["ITYPEU"], mw*np.roll(mw, -1, axis=2))
    np.testing.assert_array_equal(data["ITYPEV"][:, :-1], (mw*np.roll(mw, -1, axis=1))[:, :-1])
    for wind, mask in (("u", "ITYPEU"), ("v", "ITYPEV"), ("w", "ITYPEW")):
        np.testing.assert_array_equal(data[wind+"_topo"], np.where(data[mask] == 1, data[wind], 0))
    u, v, w = [data[n+"_topo"] for n in ("u", "v", "w")]
    xi = (np.roll(w, -1, axis=1)-w)/(radius*(np.pi/2/ny))
    eta = (np.roll(w, -1, axis=2)-w)/(radius*np.cos(lat)*(2*np.pi/nx))
    xi[:-1] -= np.diff(v, axis=0)/dz
    eta[:-1] -= np.diff(u, axis=0)/dz
    for name, curl, mask in (("xi", xi, "ITYPEV"), ("eta", eta, "ITYPEU")):
        expected = np.where(data[mask] == 1, data[name], curl)
        # Exclude the prescribed top and the north boundary row from curl oracle.
        np.testing.assert_allclose(data[name+"_topo"][:-1, :-1], expected[:-1, :-1], rtol=2e-13, atol=2e-15)
    assert np.max(np.abs(data["v"][:, -1])) == 0, "north-wall normal wind"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--wrapper", default="")
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.work.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=args.work.resolve()))
    base = json.loads((ROOT/"tests/configs/rll_mountain.json").read_text())
    base["grid"]["horizontal"].update(nx=80, ny=20)
    base["grid"]["vertical"]["nz"] = 12
    base["simulation"].update(dt_s=10, total_time_s=120, output_interval_s=60)
    base["dynamics"]["solver"]["WRXMU"] = 2/(2*np.pi/80)**2
    base["initial_conditions"]["rll_mountain"]["half_width_m"] = 1000000.
    base["initial_conditions"]["jung2019"]["perturbation_scale"] = 0.

    def run(name, config, rejection=None):
        directory = work/name
        directory.mkdir()
        config["output"]["output_dir"] = str(directory/"output")
        path = directory/"config.json"
        path.write_text(json.dumps(config, indent=2)+"\n")
        command = [args.launcher, "--bind-to", "none", "-n", str(args.ranks)]
        command += ([args.wrapper] if args.wrapper else []) + [str(args.executable.resolve()), str(path)]
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (directory/"run.log").write_text(result.stdout)
        if rejection:
            assert result.returncode != 0 and rejection in result.stdout, f"{name}: missing rejection, see {directory}"
            return
        assert result.returncode == 0, f"{name}: failed, see {directory}"
        snapshots = [read(p) for p in sorted((directory/"output").glob("mountain_*.h5"))]
        snapshots = [d for d in snapshots if "w" in d]
        assert len(snapshots) == 3, f"{name}: incomplete output"
        assert float(snapshots[-1]["model_time_s"].squeeze()) == 120
        if config["simulation"]["idealized_test"] == "rll_mountain":
            for data in snapshots:
                check(data, config)
        print(f"PASS {name}: {directory}", flush=True)
        return snapshots

    rest = copy.deepcopy(base)
    rest["initial_conditions"]["jung2019"]["jet_scale"] = 0.
    for data in run("rest", rest):
        assert all(np.max(np.abs(data[name])) == 0 for name in FIELDS)
    mountain = run("mountain", copy.deepcopy(base))
    assert np.max(np.abs(mountain[-1]["w"])) > 1e-8, "no terrain response"
    assert abs(mountain[-1]["u"][-1, 0].mean()-mountain[0]["u"][-1, 0].mean()) < 1e-11
    flat = copy.deepcopy(base)
    flat["initial_conditions"]["rll_mountain"]["height_m"] = 0.
    zero = run("zero_height", flat)
    flat = copy.deepcopy(flat)
    flat["simulation"]["idealized_test"] = "jung2019_barotropic"
    del flat["initial_conditions"]["rll_mountain"]
    flat["output"]["fields_to_output"] = list(FIELDS)
    original = run("original_flat", flat)
    for a, b in zip(zero, original):
        for name in FIELDS:
            np.testing.assert_array_equal(a[name], b[name], err_msg=f"flat limit {name}")
    for key, value in (("height_m", -1), ("half_width_m", 0), ("half_width_m", 2000000)):
        invalid = copy.deepcopy(base)
        invalid["initial_conditions"]["rll_mountain"][key] = value
        run(f"invalid_{key}_{value}", invalid, "RLL mountain requires")
    invalid = copy.deepcopy(base)
    invalid["simulation"]["idealized_test"] = "jung2019_barotropic"
    run("invalid_flat_terrain", invalid, "Mountain terrain requires")
    print(f"PASS RLL terrain integration, {args.ranks} ranks; results: {work}")


if __name__ == "__main__":
    main()
