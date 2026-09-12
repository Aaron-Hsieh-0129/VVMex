#!/usr/bin/env python3
"""Shared-model RLL stationary/coupled integration and capability rejection."""
import argparse
import copy
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_output(path):
    with h5py.File(path) as file:
        group = file["Step0"]
        return {
            name: np.asarray(group[name])
            for name in [
                "u", "v", "w", "xi", "eta", "zeta",
                "lon", "lat", "rll_background_u",
                "rll_background_zeta", "model_time_s",
            ]
        }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path)
parser.add_argument("work", type=Path)
parser.add_argument("--wrapper", default="")
parser.add_argument("--launcher", default="mpiexec")
args = parser.parse_args()
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
args.work.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="run-", dir=args.work))
base = json.loads((ROOT / "tests/configs/jung2019_case1.json").read_text())
base["grid"]["horizontal"].update(nx=80, ny=20)
base["simulation"].update(total_time_s=3600, output_interval_s=3600)
base["dynamics"]["solver"]["WRXMU"] = 2 / (2*np.pi/80)**2


def run(name, config, rejection=None):
    directory = work / name
    directory.mkdir()
    config["output"]["output_dir"] = str(directory / "output")
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    command = [args.launcher, "--bind-to", "none", "-n", "1"] + ([args.wrapper] if args.wrapper else []) + [str(args.executable.resolve()), str(path)]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    (directory / "run.log").write_text(result.stdout)
    if rejection is not None:
        if result.returncode == 0 or rejection not in result.stdout:
            raise AssertionError(f"{name}: expected rejection '{rejection}'; see {directory / 'run.log'}")
        if list((directory / "output").glob("*.h5")):
            raise AssertionError(f"{name}: rejected configuration wrote output")
        return
    if result.returncode:
        raise AssertionError(f"{name}: model failed; see {directory / 'run.log'}")
    outputs = sorted((directory / "output").glob("jung_*.h5"))
    if len(outputs) != 2:
        raise AssertionError("Expected initial and final output")
    for path in outputs:
        with h5py.File(path) as file:
            group = file["Step0"]
            if not np.array_equal(group["rll_zeta_top"][:], group["zeta"][-1]):
                raise AssertionError("Compact native-Z snapshot differs from prognostic zeta")
    initial, final = [read_output(path) for path in outputs]
    if float(final["model_time_s"].squeeze()) != 3600:
        raise AssertionError("Run did not reach final time")
    for data in (initial, final):
        for field in ("u", "v", "w", "xi", "eta", "zeta"):
            if not np.isfinite(data[field]).all():
                raise AssertionError(f"Nonfinite {field}")
        for field in ("w", "xi", "eta"):
            if np.max(np.abs(data[field])) != 0:
                raise AssertionError(f"Vertically homogeneous experiment generated {field}")
        if np.max(np.abs(data["v"][:, -1, :])) != 0:
            raise AssertionError("Nonzero north-wall normal wind")
        if name in ("rest", "jet"):
            if np.max(np.abs(data["u"][-1]-data["rll_background_u"])) > 1e-11 or np.max(np.abs(data["v"])) > 1e-11:
                raise AssertionError("Stationary wind drift")
            if np.max(np.abs(data["zeta"][-1]-data["rll_background_zeta"])) > 1e-15:
                raise AssertionError("Stationary vorticity drift")
    # Same row metric in both states, so comparing physical row means is
    # equivalent to comparing covariant circulation (not a domain-mean fix).
    if abs(final["u"][-1, 0].mean()-initial["u"][-1, 0].mean()) > 1e-11:
        raise AssertionError("Wall circulation drift")
    if name == "coupled" and np.array_equal(initial["zeta"], final["zeta"]):
        raise AssertionError("Perturbation did not evolve")


for name, jet, perturbation in (("rest", 0, 0), ("jet", 1, 0), ("coupled", 1, 1)):
    config = copy.deepcopy(base)
    config["initial_conditions"]["jung2019"].update(jet_scale=jet, perturbation_scale=perturbation)
    run(name, config)

negative = [
    ("restart", ("restart", "enable"), True, "Unsupported RLL option"),
    ("diffusion", ("dynamics", "prognostic_variables", "zeta", "tendency_terms", "diffusion"), {"enable": True}, "RLL vorticity currently supports"),
    ("disabled_transport", ("dynamics", "prognostic_variables", "zeta", "tendency_terms", "advection", "enable"), False, "requires enabled advection"),
    ("missing_vorticity", ("dynamics", "prognostic_variables", "xi"), {}, "requires all three vorticity"),
    ("zero_iterations", ("dynamics", "solver", "iteration"), 0, "positive fixed solver"),
    ("wrong_case", ("initial_conditions", "jung2019", "case"), 3, "case must be 1 or 2"),
    ("wrong_amplitude", ("initial_conditions", "jung2019", "jet_scale"), 2, "amplitudes must select"),
    ("external_initialization", ("initial_conditions", "source_file"), "not-read.nc", "analytic initial conditions"),
    ("bp5", ("output", "engine"), "BP5", "requires HDF5"),
]
for name, keys, value, message in negative:
    config = copy.deepcopy(base)
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    run(name, config, message)
print(f"PASS: shared-model rest, jet, coupled graphs and {len(negative)} rejection cases; evidence: {work}")
