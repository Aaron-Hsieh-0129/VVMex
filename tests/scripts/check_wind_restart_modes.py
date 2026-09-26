#!/usr/bin/env python3
"""Check automatic wind-solver checkpoints in BP5/HDF5 and Cartesian/RLL."""
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
history = ("psi", "psinm1", "chi", "chinm1", "W3DNM1")
physical = ("u", "v", "w", "xi", "eta", "zeta")


def config_for(geometry, engine):
    if geometry == "rll":
        config = json.loads((ROOT / "tests/configs/rll_mountain.json").read_text())
        config["grid"]["horizontal"].update(nx=80, ny=40)
        config["grid"]["horizontal"]["geometry"]["latitude_bounds_deg"] = [-70.0, 70.0]
        config["grid"]["vertical"]["nz"] = 8
        config["initial_conditions"]["jung2019"]["perturbation_scale"] = 0.0
        config["initial_conditions"]["rll_mountain"].update(
            half_width_m=1200000.0, center_latitude_deg=30.0,
            center_longitude_deg=45.0, zonal_flow=True, u0_m_s=20.0)
        config["dynamics"]["solver"]["WRXMU"] = 2 / (2 * np.pi / 80) ** 2
        config["simulation"].update(total_time_s=120, output_interval_s=10)
        config["output"]["fields_to_output"] = list(physical)
    else:
        config = json.loads((ROOT / "tests/configs/bp5_restart_source.json").read_text())
        config["simulation"]["total_time_s"] = 10
    config["output"]["engine"] = engine
    config["output"]["output_initial_step"] = True
    if engine == "BP5":
        config["output"]["bp5"] = {"aggregation_type": "TwoLevelShm", "num_subfiles": 1,
            "stats_level": 0, "async_write": False, "buffer_mode": "pack", "overwrite": True}
    else:
        config["output"].pop("bp5", None)
    return config


def run(case, config):
    directory = work / case
    directory.mkdir()
    output = directory / "output"
    config["output"]["output_dir"] = str(output)
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    command = [args.launcher, "--bind-to", "none", "-n", "1"]
    command += ([args.wrapper] if args.wrapper else []) + [str(args.executable.resolve()), str(path)]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=180)
    (directory / "run.log").write_text(result.stdout)
    assert result.returncode == 0, f"{case} failed; see {directory / 'run.log'}"
    return output


def filename(output, prefix, engine, index):
    return output / (prefix + ".bp" if engine == "BP5" else f"{prefix}_{index:06d}.h5")


def read(path, engine, name, index):
    stored = "restart/" + name if name in history else name
    if engine == "BP5":
        with adios2.FileReader(str(path)) as reader:
            return np.asarray(reader.read(stored, step_selection=[index, 1]))
    with h5py.File(path) as file:
        return np.asarray(file["Step0"][stored])


for geometry in ("rll", "cartesian"):
    for engine in ("BP5", "HDF5"):
        label = f"{geometry}_{engine.lower()}"
        base = config_for(geometry, engine)
        total = int(base["simulation"]["total_time_s"])
        interval = int(base["simulation"]["output_interval_s"])
        midpoint = total // 2
        source_config = copy.deepcopy(base)
        source_config["simulation"]["total_time_s"] = midpoint
        continuous = run(label + "_continuous", copy.deepcopy(base))
        source = run(label + "_source", source_config)
        source_file = filename(source, base["output"]["output_filename_prefix"],
                               engine, midpoint // interval)
        restart_config = copy.deepcopy(base)
        restart_config["restart"] = {"enable": True, "source_file": str(source_file),
                                     "step_index": -1}
        resumed = run(label + "_restart", restart_config)
        prefix = base["output"]["output_filename_prefix"]
        checkpoint = filename(continuous, prefix, engine, midpoint // interval)
        resumed_checkpoint = filename(resumed, prefix, engine, midpoint // interval)
        final = filename(continuous, prefix, engine, total // interval)
        resumed_final = filename(resumed, prefix, engine, total // interval)
        if engine == "HDF5":
            with h5py.File(source_file) as file:
                assert all(name not in file["Step0"] for name in history)
                assert all("restart/" + name in file["Step0"] for name in history)
        else:
            with adios2.FileReader(str(source_file)) as file:
                names = file.available_variables()
                assert all(name not in names for name in history)
                assert all("restart/" + name in names for name in history)
        for name in physical + history:
            step = midpoint // interval if engine == "BP5" else 0
            resumed_step = 0
            np.testing.assert_array_equal(read(checkpoint, engine, name, step),
                read(source_file, engine, name, step), err_msg=f"{label} source {name}")
            np.testing.assert_allclose(read(checkpoint, engine, name, step),
                read(resumed_checkpoint, engine, name, resumed_step),
                atol=1e-12, rtol=1e-12, err_msg=f"{label} checkpoint {name}")
        for name in ("u", "v", "w", "zeta"):
            expected = read(final, engine, name, total // interval if engine == "BP5" else 0)
            actual = read(resumed_final, engine, name,
                          (total - midpoint) // interval if engine == "BP5" else 0)
            scale = np.sqrt(np.mean(expected ** 2))
            error = np.sqrt(np.mean((actual - expected) ** 2))
            assert error < 1e-2 * max(scale, 1e-12), (label, name, error, scale)
print(f"PASS: BP5/HDF5 Cartesian/RLL wind restart checkpoints; {work}")
