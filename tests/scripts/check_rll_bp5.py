#!/usr/bin/env python3
"""Short rotating/shifted-jet RLL history parity and coordinate checks."""
import argparse
import copy
import json
import resource
from pathlib import Path
import subprocess
import tempfile

import adios2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path)
parser.add_argument("work", type=Path)
parser.add_argument("--wrapper", default="")
parser.add_argument("--launcher", default="mpiexec")
args = parser.parse_args()
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
args.work.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="run-", dir=args.work.resolve()))
base = json.loads((ROOT/"experiments/topography/configs/mountain.json").read_text())
base["grid"]["horizontal"].update(nx=80, ny=40)
base["grid"]["vertical"]["nz"] = 8
base["simulation"].update(dt_s=10, total_time_s=120, output_interval_s=60)
base["initial_conditions"]["rll_mountain"].update(center_latitude_deg=20., jet_center_latitude_deg=20.)
base["initial_conditions"]["jung2019"]["perturbation_scale"] = 0.
base["constants"]["OMEGA"] = 7.292e-5
base["dynamics"]["solver"]["WRXMU"] = 2/(2*np.pi/80)**2
for name in ("xi", "eta", "zeta"):
    base["dynamics"]["prognostic_variables"][name]["tendency_terms"]["coriolis"] = {
        "enable": True, "spatial_scheme": "Takacs", "temporal_scheme": "AdamsBashforth2"}
fields = ["lon", "lat", "u", "v", "w", "xi", "eta", "zeta", "f_2d", "rll_terrain_height", "rll_background_u"]
base["output"]["fields_to_output"] = fields
base["output"]["bp5"] = {"aggregation_type": "TwoLevelShm", "num_subfiles": 1,
    "stats_level": 0, "async_write": True, "buffer_mode": "pack", "overwrite": True}


def run(name, config):
    directory = work/name
    directory.mkdir()
    config["output"]["output_dir"] = str(directory/"output")
    path = directory/"config.json"
    path.write_text(json.dumps(config, indent=2)+"\n")
    command = [args.launcher, "--bind-to", "none", "-n", "1"]
    command += ([args.wrapper] if args.wrapper else [])+[str(args.executable.resolve()), str(path)]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    (directory/"run.log").write_text(result.stdout)
    assert result.returncode == 0, f"Model failed; see {directory/'run.log'}"
    return directory/"output"


hdf = run("hdf5", copy.deepcopy(base))
bp_config = copy.deepcopy(base)
bp_config["output"]["engine"] = "BP5"
bp = run("bp5", bp_config)
paths = sorted(hdf.glob("mountain_*.h5"))
assert len(paths) == 3
with adios2.FileReader(str(bp/"mountain.bp")) as reader:
    assert int(reader.available_variables()["model_time_s"]["AvailableStepsCount"]) == 3
    assert reader.read_attribute("coordinates/x/units") == "degrees_east"
    assert reader.read_attribute("coordinates/y/units") == "degrees_north"
    assert reader.read_attribute("horizontal_geometry") == "regular_latlon"
    for step, path in enumerate(paths):
        with h5py.File(path) as file:
            d = {name: np.asarray(file["Step0"][name]) for name in fields}
        for name in fields:
            value = np.asarray(reader.read(name, step_selection=[step, 1]))
            assert np.isfinite(value).all(), (name, step)
            np.testing.assert_array_equal(value, d[name], err_msg=f"BP5/HDF5 {name}, step{step}")
        time = float(np.asarray(reader.read("model_time_s", step_selection=[step, 1])).squeeze())
        assert time == step*60
        np.testing.assert_allclose(reader.read("coordinates/x", step_selection=[step, 1]), d["lon"][0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(reader.read("coordinates/y", step_selection=[step, 1]), d["lat"][:, 0], atol=1e-12, rtol=0)
        phi = np.deg2rad(d["lat"])
        np.testing.assert_allclose(d["f_2d"], 2*7.292e-5*np.sin(phi+np.pi/160), atol=1e-18, rtol=1e-13)
        shifted = phi-np.deg2rad(20.)
        jet = np.zeros_like(phi)
        inside = np.abs(shifted) < np.pi/8
        jet[inside] = 80*np.exp(1/((shifted[inside]+np.pi/8)*(shifted[inside]-np.pi/8))+4/(np.pi/4)**2)
        np.testing.assert_allclose(d["rll_background_u"], jet, atol=1e-11, rtol=1e-13)
        distance = 6371220*np.arccos(np.clip(np.sin(phi)*np.sin(np.deg2rad(20.))
            +np.cos(phi)*np.cos(np.deg2rad(20.))*np.cos(np.deg2rad(d["lon"])-np.pi), -1, 1))
        height = np.where(distance < 1500000., 2000*np.exp(-(distance/500000.)**2), 0)
        np.testing.assert_array_equal(d["rll_terrain_height"], np.rint(height/1000)*1000)
print(f"PASS: shifted jet, mountain, native-Z f, finite graph-enabled evolution and bitwise BP5/HDF5 parity; {work}")
