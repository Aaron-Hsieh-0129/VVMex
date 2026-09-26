#!/usr/bin/env python3
"""Compare a short RLL mountain run with a BP5 restart at its midpoint."""
import argparse
import copy
import json
import resource
from pathlib import Path
import subprocess
import tempfile

import adios2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path)
parser.add_argument("work", type=Path)
parser.add_argument("--wrapper", default="")
parser.add_argument("--launcher", default="mpiexec")
parser.add_argument("--ranks", type=int, default=2)
args = parser.parse_args()
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
args.work.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="run-", dir=args.work.resolve()))

base = json.loads((ROOT / "tests/configs/rll_mountain.json").read_text())
base["grid"]["horizontal"].update(nx=80, ny=40)
base["grid"]["horizontal"]["geometry"]["latitude_bounds_deg"] = [-70.0, 70.0]
base["grid"]["vertical"]["nz"] = 8
base["initial_conditions"]["jung2019"]["perturbation_scale"] = 0.0
base["initial_conditions"]["rll_mountain"].update(
    height_m=2000.0, half_width_m=1200000.0, center_longitude_deg=45.0,
    center_latitude_deg=30.0, zonal_flow=True, u0_m_s=20.0)
base["dynamics"]["solver"]["WRXMU"] = 2 / (2 * np.pi / 80) ** 2
base["constants"]["OMEGA"] = 7.292e-5
for name in ("xi", "eta", "zeta"):
    base["dynamics"]["prognostic_variables"][name]["tendency_terms"]["coriolis"] = {
        "enable": True, "spatial_scheme": "Takacs", "temporal_scheme": "AdamsBashforth2"}
base["simulation"].update(total_time_s=7200, output_interval_s=600)
base["output"]["engine"] = "BP5"
base["output"]["fields_to_output"] = ["u", "v", "w", "xi", "eta", "zeta"]
base["output"]["bp5"] = {"aggregation_type": "TwoLevelShm", "num_subfiles": 1,
    "stats_level": 0, "async_write": False, "buffer_mode": "pack", "overwrite": True}


def run(name, config):
    directory = work / name
    directory.mkdir()
    config["output"]["output_dir"] = str(directory / "output")
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    command = [args.launcher, "--bind-to", "none", "-n", str(args.ranks)]
    command += ([args.wrapper] if args.wrapper else []) + [str(args.executable.resolve()), str(path)]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=180)
    (directory / "run.log").write_text(result.stdout)
    assert result.returncode == 0, f"{name} failed; see {directory / 'run.log'}"
    return directory / "output" / "mountain.bp"


continuous = run("continuous", copy.deepcopy(base))
source_config = copy.deepcopy(base)
source_config["simulation"]["total_time_s"] = 3600
source = run("source", source_config)
restart_config = copy.deepcopy(base)
restart_config["restart"] = {"enable": True, "source_file": str(source), "step_index": -1}
restart = run("restart", restart_config)

fields = ("u", "v", "w", "xi", "eta", "zeta", "psi", "psinm1", "chi", "chinm1", "W3DNM1")
solver_history = {"psi", "psinm1", "chi", "chinm1", "W3DNM1"}


def read(reader, field, step):
    name = "restart/" + field if field in solver_history else field
    return np.asarray(reader.read(name, step_selection=[step, 1]))


with adios2.FileReader(str(continuous)) as whole, adios2.FileReader(str(source)) as first, \
        adios2.FileReader(str(restart)) as resumed:
    assert int(first.available_variables()["model_time_s"]["AvailableStepsCount"]) == 7
    assert int(resumed.available_variables()["model_time_s"]["AvailableStepsCount"]) == 7
    for field in solver_history:
        assert field not in whole.available_variables(), field
        assert "restart/" + field in whole.available_variables(), field
    for field in fields:
        checkpoint = read(first, field, 6)
        np.testing.assert_array_equal(checkpoint,
            read(whole, field, 6), err_msg=f"source {field}")
        np.testing.assert_array_equal(checkpoint,
            read(resumed, field, 0), err_msg=f"loaded {field}")
        for model_time in (4200, 4800, 7200):
            expected = read(whole, field, model_time // 600)
            actual = read(resumed, field, (model_time - 3600) // 600)
            scale = np.sqrt(np.mean(expected ** 2))
            error = np.sqrt(np.mean((actual - expected) ** 2))
            assert error <= 1e-4 * max(scale, 1e-12), (field, model_time, error, scale)
print(f"PASS: RLL mountain BP5 restart retains wind and solver history; {work}")
