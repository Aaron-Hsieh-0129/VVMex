#!/usr/bin/env python3
"""Generate scoped shared-model Jung runs and inspect native-grid HDF5 output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def launch(args):
    directory = Path(args.directory).resolve()
    project_outputs = ROOT / "experiments/jung2019"
    if not directory.is_relative_to(project_outputs):
        raise ValueError("Experiment must stay under this project's experiments/jung2019")
    config = json.loads((directory / "config.json").read_text())
    output = Path(config["output"]["output_dir"]).resolve()
    if not output.is_relative_to(directory) or output.exists() or (directory / "run.log").exists():
        raise ValueError("Refusing an existing output or log; configure a unique run name")
    if args.ranks < 1 or args.threads < 1 or args.ranks * args.threads > 224:
        raise ValueError("Launch exceeds the 224-core/rank limit")
    devices = [int(d) for d in args.gpus.split(",")] if args.gpus else []
    if devices and (len(set(devices)) != len(devices) or any(d not in range(8) for d in devices) or args.ranks > len(devices)):
        raise ValueError("GPU ranks require distinct permitted devices 0–7")
    h, v = config["grid"]["horizontal"], config["grid"]["vertical"]
    simulation = config["simulation"]
    snapshots = 2 + int(simulation["total_time_s"] / simulation["output_interval_s"])
    fields = config["output"]["fields_to_output"]
    planes = sum(1 if name in ("rll_zeta_top", "lon", "lat", "rll_background_u", "rll_background_zeta") else v["nz"] for name in fields)
    estimate = h["nx"] * h["ny"] * 8 * (planes + 1) * snapshots + 1_000_000 * snapshots
    used = sum(p.stat().st_size for p in project_outputs.rglob("*") if p.is_file())
    if used + estimate > 10_000_000_000:
        raise ValueError("Conservative projected experiment storage exceeds 10 GB")
    executable = Path(args.executable).resolve()
    command = ["mpiexec", "--bind-to", "none", "-n", str(args.ranks)]
    if devices:
        command += [str(ROOT / "tests/scripts/one_gpu_per_rank.sh")]
    command += [str(executable), str(directory / "config.json")]
    env = os.environ.copy()
    env.update(VVM_ROOT=str(ROOT), OMP_NUM_THREADS=str(args.threads))
    if devices:
        env["VVM_TEST_GPUS"] = args.gpus
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT)
    manifest = {"command": command, "head": git("rev-parse", "HEAD").decode().strip(),
                "branch": git("branch", "--show-current").decode().strip(),
                "status": git("status", "--short").decode(),
                "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
                "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "configuration_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
                "environment": {key: env.get(key, "") for key in ("VVM_ROOT", "OMP_NUM_THREADS", "VVM_TEST_GPUS", "LD_LIBRARY_PATH")},
                "storage_before_bytes": used, "conservative_output_estimate_bytes": estimate}
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "src", "tests", "tools", "rundata/input_configs/jung2019").decode().splitlines()
    manifest["untracked_source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in untracked if (ROOT / name).suffix in (".cpp", ".hpp", ".h", ".py", ".json", ".cmake", ".sh")}
    cache = executable.parent / "CMakeCache.txt"
    if cache.exists():
        manifest["cmake_cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    destination = directory / "provenance.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()
    with (directory / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-start)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"Model failed; see {directory / 'run.log'}")
    print(destination)


def configure(args):
    config = json.loads((ROOT / "rundata/input_configs/jung2019/case1.json").read_text())
    h = config["grid"]["horizontal"]
    h.update(nx=args.nx, ny=args.nx // 4)
    config["initial_conditions"]["jung2019"].update(
        case=args.case, jet_scale=0.0 if args.state == "rest" else 1.0,
        perturbation_scale=1.0 if args.state == "perturbed" else 0.0)
    dt = args.dt if args.dt is not None else 600.0 * 400 / args.nx
    hours = args.hours if args.hours is not None else (168 if args.case == 1 else 120)
    config["simulation"].update(dt_s=dt, total_time_s=hours * 3600,
                                output_interval_s=min(hours * 3600, args.interval * 3600))
    config["dynamics"]["solver"].update(WRXMU=2 / (2 * np.pi / args.nx) ** 2,
                                         iteration=args.iterations, initial_iterations=args.initial_iterations)
    if args.compact:
        config["output"]["fields_to_output"] = ["rll_zeta_top"]
    directory = (ROOT / "experiments/jung2019" / args.name).resolve()
    if not directory.is_relative_to(ROOT / "experiments/jung2019"):
        raise ValueError("Run name must stay under experiments/jung2019")
    directory.mkdir(parents=True, exist_ok=False)
    config["output"]["output_dir"] = str(directory / "output")
    destination = directory / "config.json"
    destination.write_text(json.dumps(config, indent=2) + "\n")
    print(destination)


def read_output(path):
    with h5py.File(path) as file:
        group = file["Step0"]
        return {name: np.asarray(group[name]) for name in ["u", "v", "w", "xi", "eta", "zeta", "lon", "lat", "rll_background_u", "rll_background_zeta", "model_time_s"]}


def read_zeta(group):
    return np.asarray(group["rll_zeta_top"]) if "rll_zeta_top" in group else np.asarray(group["zeta"][-1])


def analyze(args):
    directory = Path(args.directory)
    config = json.loads((directory / "config.json").read_text())
    paths = sorted((directory / "output").glob("*.h5"))
    paths = [p for p in paths if "topo" not in p.name]
    if not paths:
        raise ValueError("No model outputs; refusing to analyze stale or absent results")
    h = config["grid"]["horizontal"]
    radius = h["geometry"]["earth_radius_m"]
    dy = np.pi / 2 / h["ny"]
    phi_u = -np.pi / 4 + (np.arange(h["ny"]) + .5) * dy
    phi_z = phi_u + .5 * dy
    rows = []
    for path in paths:
        if config["output"]["fields_to_output"] == ["rll_zeta_top"]:
            with h5py.File(path) as file:
                data = file["Step0"]
                zeta = read_zeta(data)
                rows.append({"file": str(path), "time_s": float(np.asarray(data["model_time_s"]).squeeze()),
                             "finite": bool(np.isfinite(zeta).all()), "zeta_min": float(zeta.min()), "zeta_max": float(zeta.max()),
                             "weighted_zeta_integral": float(np.sum(zeta*np.cos(phi_z)[:, None])*radius**2*dy*(2*np.pi/h["nx"])),
                             "wind_diagnostics": "not recorded in compact vorticity history", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            continue
        data = read_output(path)
        u, v, zeta = (data[n][-1] for n in ("u", "v", "zeta"))
        row = {"file": str(path), "time_s": float(data["model_time_s"].squeeze()),
               "finite": all(np.isfinite(data[n]).all() for n in ("u", "v", "w", "xi", "eta", "zeta")),
               "u_max": float(np.max(u)), "v_max_abs": float(np.max(np.abs(v))),
               "zeta_min": float(zeta.min()), "zeta_max": float(zeta.max()),
               "north_normal_wind_max": float(np.max(np.abs(v[-1]))),
               "south_covariant_circulation_mean": float(radius * np.cos(phi_u[0]) * u[0].mean()),
               "weighted_zeta_integral": float(np.sum(zeta * np.cos(phi_z)[:, None]) * radius**2 * dy * (2*np.pi/h["nx"])),
               "horizontal_vorticity_max": float(max(np.max(np.abs(data["xi"])), np.max(np.abs(data["eta"])))),
               "vertical_wind_max": float(np.max(np.abs(data["w"]))),
               "jet_error_max": float(np.max(np.abs(u - data["rll_background_u"]))),
               "background_zeta_error_max": float(np.max(np.abs(zeta - data["rll_background_zeta"]))),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        rows.append(row)
    if rows[-1]["time_s"] != config["simulation"]["total_time_s"]:
        raise ValueError("Run has not reached its configured end time")
    result = {"configuration": config, "outputs": rows}
    (directory / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    if not all(row["finite"] for row in rows):
        raise ValueError("Nonfinite model fields")
    if args.stationary:
        # Stationary rest/jet is an exact discrete solution with compatible
        # streamfunction wall values. Permit FP64 roundoff, not truncation drift.
        for row in rows:
            if max(row["jet_error_max"], row["v_max_abs"], row["vertical_wind_max"]) > 1e-11:
                raise ValueError("Stationary wind changed beyond FP64 roundoff")
            if max(row["horizontal_vorticity_max"], row["background_zeta_error_max"]) > 1e-15:
                raise ValueError("Stationary vorticity changed beyond FP64 roundoff")


parser = argparse.ArgumentParser(description=__doc__)
sub = parser.add_subparsers(dest="command", required=True)
p = sub.add_parser("configure")
p.add_argument("name")
p.add_argument("--case", type=int, choices=(1, 2), default=1)
p.add_argument("--nx", type=int, default=400)
p.add_argument("--state", choices=("rest", "jet", "perturbed"), default="perturbed")
p.add_argument("--dt", type=float)
p.add_argument("--hours", type=float)
p.add_argument("--interval", type=float, default=24)
p.add_argument("--iterations", type=int, default=200)
p.add_argument("--initial-iterations", type=int, default=1000)
p.add_argument("--compact", action="store_true", help="Save only the native top Z plane for high-resolution L2 studies")
p.set_defaults(func=configure)
p = sub.add_parser("launch")
p.add_argument("directory")
p.add_argument("--executable", default=str(ROOT / "build/vvm"))
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--gpus", default="", help="GPU list, e.g. 0,1,2,3; omit for a CPU executable")
p.set_defaults(func=launch)
p = sub.add_parser("analyze")
p.add_argument("directory")
p.add_argument("--stationary", action="store_true")
p.set_defaults(func=analyze)
if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
