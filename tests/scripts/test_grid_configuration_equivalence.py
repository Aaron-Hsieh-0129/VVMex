#!/usr/bin/env python3
"""Generate equivalent Cartesian cases and strictly compare their HDF5 output."""

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

VARIANTS = ("legacy", "horizontal", "vertical", "structured", "mixed")
FIELDS = ("u", "v", "w", "xi", "eta", "zeta", "th", "qv", "thbar", "pibar",
          "rhobar", "rhobar_up", "Tg", "topo", "lon", "lat", "f_2d")
HORIZONTAL_KEYS = ("nx", "ny", "n_halo_cells", "dx", "dy", "fix_lonlat", "boundary_condition")
VERTICAL_KEYS = ("nz", "dz", "dz1", "vertical_coordinate_type", "rcemip_grid_data_path")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")


def prepare(case, work, steps):
    base = json.loads(case.read_text(encoding="utf-8"))
    grid = base["grid"]

    require(steps >= 4, "Use at least four steps to exercise repeated stepping.")
    require(base["simulation"]["idealized_test"] == "2dbubble", "Expected the existing 2dbubble fixture.")
    require(base["simulation"]["dt_s"] == 1.0, "This fixture expects a one-second timestep.")
    require(grid.get("vertical_coordinate_type", "default") == "default" and grid["dz"] == grid["dz1"],
            "This test expects the existing uniform Cartesian vertical grid.")
    require(grid.get("fix_lonlat", False), "This fixture expects the existing fixed geographic location.")
    require(all(grid["boundary_condition"][axis] == "periodic" for axis in ("x", "y")),
            "This test requires periodic horizontal boundaries.")

    horizontal = {
        "nx": grid["nx"],
        "ny": grid["ny"],
        "n_halo_cells": grid["n_halo_cells"],
        "geometry": {
            "kind": "cartesian",
            "dx": grid["dx"],
            "dy": grid["dy"],
            "fix_lonlat": True
        },
        "topology": {"q1": "periodic", "q2": "periodic"}
    }

    vertical = {
        "nz": grid["nz"],
        "type": "default",
        "dz": grid["dz"],
        "dz1": grid["dz1"]
    }

    work.mkdir(parents=True, exist_ok=True)

    # Every preparation gets fresh output directories. Old output cannot
    # satisfy verification if a subsequent model run fails.
    run_root = Path(tempfile.mkdtemp(prefix="run_", dir=work)).resolve()
    manifest = {
        "grid": grid,
        "steps": steps,
        "run_root": str(run_root),
        "outputs": {}
    }

    for variant in VARIANTS:
        config = copy.deepcopy(base)
        selected = config["grid"]

        if variant in ("horizontal", "structured", "mixed"):
            for key in HORIZONTAL_KEYS:
                selected.pop(key, None)
            selected["horizontal"] = copy.deepcopy(horizontal)

        if variant in ("vertical", "structured", "mixed"):
            for key in VERTICAL_KEYS:
                selected.pop(key, None)
            selected["vertical"] = copy.deepcopy(vertical)

        if variant == "mixed":
            selected.update({
                "nx": 8,
                "ny": 8,
                "nz": 8,
                "n_halo_cells": 3,
                "dx": 111.0,
                "dy": 222.0,
                "dz": 333.0,
                "dz1": 111.0,
                "fix_lonlat": False,
                "vertical_coordinate_type": "taiwanvvm",
                "boundary_condition": {
                    "x": "zero_gradient",
                    "y": "zero_gradient"
                },
                "rcemip_grid_data_path": "intentionally_unused_profile.txt"
            })

        output = run_root / variant
        output.mkdir()

        config["simulation"]["total_time_s"] = float(steps)
        config["simulation"]["output_interval_s"] = float(steps)
        config.setdefault("restart", {})["enable"] = False

        config["output"].update({
            "engine": "HDF5",
            "precision": "native",
            "output_initial_step": True,
            "output_dir": str(output),
            "output_filename_prefix": "history",
            "fields_to_output": list(FIELDS),
            "output_grid": {
                "x_start": 0,
                "x_end": -1,
                "y_start": 0,
                "y_end": -1,
                "z_start": 0,
                "z_end": -1
            }
        })

        # Fixed paths are consumed by CTest. Copies alongside the output
        # preserve the exact configuration associated with each run.
        write_json(work / f"{variant}.json", config)
        write_json(run_root / f"{variant}.json", config)
        manifest["outputs"][variant] = str(output)

    write_json(work / "manifest.json", manifest)
    print(f"Prepared five equivalent cases; artifacts: {run_root}", flush=True)


def datasets(file):
    result = {}

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            require(obj.dtype.kind in "biuf", f"Unsupported dataset type: {name}: {obj.dtype}")
            data = np.asarray(obj[()])
            require(np.isfinite(data).all(), f"Nonfinite data: {name}")
            result[name] = data

    file.visititems(collect)
    return result


def metadata(file):
    # Compare semantic metadata, not file-local HDF5 object references.
    names = ("units", "long_name", "grid_staggering")
    result = {}

    for key, value in file.attrs.items():
        if key.endswith(tuple("/" + name for name in names)) or key in ("vvm_real_precision", "vvm_field_precision"):
            result["root:" + key] = np.asarray(value)

    def collect(name, obj):
        for key in names:
            if key in obj.attrs:
                result[name + ":" + key] = np.asarray(obj.attrs[key])

    file.visititems(collect)
    return result


def read_output(path, grid, expected_step):
    with h5py.File(path, "r") as file:
        data = datasets(file)
        attributes = metadata(file)

    required = set(FIELDS) | {
        "time",
        "model_time_s",
        "model_step",
        "coordinates/x",
        "coordinates/y",
        "coordinates/z_mid"
    }

    require({"Step0/" + name for name in required} <= data.keys(), f"Missing required datasets in {path}")

    for name in ("time", "model_time_s", "model_step"):
        clock = data["Step0/" + name]
        require(clock.size == 1 and clock.item() == expected_step, f"Incorrect {name} in {path}")

    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]

    for name in ("u", "v", "w", "xi", "eta", "zeta", "th", "qv"):
        require(data["Step0/" + name].shape == (nz, ny, nx), f"Incorrect {name} shape in {path}")

    axes = (
        ("x", nx, grid["dx"], 0.0),
        ("y", ny, grid["dy"], 0.0),
        ("z_mid", nz, grid["dz"], 0.5)
    )

    for name, length, spacing, offset in axes:
        actual = data["Step0/coordinates/" + name]
        expected = (np.arange(length, dtype=actual.dtype) + offset) * spacing
        require(actual.shape == (length,) and np.array_equal(actual, expected),
                f"Incorrect {name} coordinates in {path}")

    for name, value in (("lon", 120.95), ("lat", 23.458)):
        actual = data["Step0/" + name]
        require(actual.shape == (ny, nx) and np.all(actual == actual.dtype.type(value)),
                f"Incorrect fixed {name} in {path}")

    return data, attributes


def compare_outputs(reference, actual, label):
    left, left_metadata = reference
    right, right_metadata = actual

    require(left.keys() == right.keys(), f"{label}: dataset names differ")

    for name in left:
        a, b = left[name], right[name]
        require(a.shape == b.shape and a.dtype == b.dtype, f"{label}: shape or dtype differs for {name}")
        require(a.tobytes(order="C") == b.tobytes(order="C"), f"{label}: values differ for {name}")

    require(left_metadata.keys() == right_metadata.keys(), f"{label}: metadata names differ")

    for name in left_metadata:
        require(np.array_equal(left_metadata[name], right_metadata[name]), f"{label}: metadata differs for {name}")

    print(f"PASS: {label}: {len(left)} datasets match bit-for-bit", flush=True)


def verify(work):
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    snapshots = {}
    descriptors = {}

    for variant in VARIANTS:
        output = Path(manifest["outputs"][variant])
        expected_files = {"history_000000.h5", "history_000001.h5"}

        require({path.name for path in output.glob("history_*.h5")} == expected_files,
                f"{variant}: expected exactly initial and final history files")

        snapshots[variant] = [
            read_output(output / f"history_{index:06d}.h5", manifest["grid"], step)
            for index, step in enumerate((0, manifest["steps"]))
        ]

        descriptors[variant] = (output / "vvm.ctl").read_text(encoding="utf-8")

    for variant in VARIANTS[1:]:
        for index in (0, 1):
            compare_outputs(snapshots["legacy"][index], snapshots[variant][index],
                            f"legacy vs {variant}, output {index}")

        require(descriptors[variant] == descriptors["legacy"], f"{variant}: GrADS descriptor differs")

    initial = snapshots["legacy"][0][0]
    final = snapshots["legacy"][1][0]

    require(any(not np.array_equal(initial["Step0/" + name], final["Step0/" + name])
                for name in ("u", "w", "th", "eta")),
            "No dynamical evolution detected between initial and final output")

    print(f"PASS: Cartesian configuration equivalence. Artifacts: {manifest['run_root']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("prepare")
    setup.add_argument("--case", type=Path, required=True)
    setup.add_argument("--work", type=Path, required=True)
    setup.add_argument("--steps", type=int, default=12)

    comparison = commands.add_parser("verify")
    comparison.add_argument("--work", type=Path, required=True)

    args = parser.parse_args()

    try:
        if args.command == "prepare":
            prepare(args.case.resolve(), args.work.resolve(), args.steps)
        else:
            verify(args.work.resolve())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
