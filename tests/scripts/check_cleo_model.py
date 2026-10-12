#!/usr/bin/env python3
"""Full-model CLEO run: the coupling as Model drives it.

A small x-z bubble (ny = 2; 4 with --ranks) with CLEO as the only microphysics, run through the
real `vvm` executable. It checks that:

  1. Model constructs, initialises, steps and finalises CLEO, and the run
     reaches total_time_s.
  2. Every CLEO-written field is in the output, finite and physically
     admissible: nothing negative, no ice, qp >= qc + qr, ice radius 25 um.
  3. What CLEO was sent is what the model holds, gridbox by gridbox: CLEO's
     own zarr record of temperature, pressure and vapour at t = 0 against the
     model's initial th * pibar, pbar and qv. This is the index mapping and the
     unit conversion of send_dynamics, read back from CLEO's side.
  4. With average_over_y, every CLEO-written field is identical across y.

With --ranks N it instead runs the same case on 1 and on N ranks, collisions
off so CLEO is deterministic, and requires the two to agree: the model
fields, CLEO's zarr record of the t = 0 state, and where every superdroplet
started. CollectiveDataset decides where each rank's share lands in the zarr,
and a wrong offset there corrupts the record without any error.
"""

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

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("executable", type=Path, help="Path to the VVM executable")
parser.add_argument("work", type=Path, help="Directory used for temporary test files")
parser.add_argument("--wrapper", default="", help="Optional GPU wrapper such as one_gpu_per_rank.sh")
parser.add_argument("--launcher", default="mpiexec", help="MPI launcher")
parser.add_argument("--ranks", type=int, default=1,
                    help="also run on this many ranks and compare with 1 rank")
parser.add_argument("--path2cleo", type=Path, default=None,
                    help="CLEO source root, passed through to gen_cleo_inputs.py")
args = parser.parse_args()

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
args.work.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="run-", dir=args.work))

# ny = 2 makes the single-rank case quasi-2-D, which is what average_over_y is
# for. The release Grid may split y between ranks, and a split needs at least
# n_halo_cells = 2 cells per rank, hence ny = 4 for the rank comparison.
NX, NY, NZ = 16, (2 if args.ranks == 1 else 4), 16
FIELDS_3D = ["th", "qv", "u", "w", "qp", "qc", "qr", "nc", "nr", "qi",
             "diag_eff_radius_qc", "diag_eff_radius_qr", "diag_eff_radius_qi"]
FIELDS_2D = ["precip_liq_surf_mass", "precip_liq_surf_flux",
             "precip_ice_surf_mass", "precip_ice_surf_flux"]
CLEO_WRITTEN_3D = ["qp", "qc", "qr", "nc", "nr", "diag_eff_radius_qc", "diag_eff_radius_qr"]
NONNEGATIVE = ["qv", "qp", "qc", "qr", "nc", "nr"] + FIELDS_2D

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        print(f"FAIL: {message}")


# =============================================================================
# Configuration: 2dbubble, made small and moist, with CLEO in place of P3
# =============================================================================

def make_config(case_dir, collisions, average_over_y):
    config = json.loads((ROOT / "tests" / "configs" / "2dbubble.json").read_text())
    template = json.loads((ROOT / "tests" / "configs" / "cleo_coupling.json").read_text())

    # 2dbubble is a standard test: no spatial NetCDF is read, and the one the
    # regression config names is for a 32 x 32 grid.
    config.pop("netcdf_reader", None)
    config["grid"].update(nx=NX, ny=NY, nz=NZ, dx=250.0, dy=250.0, dz=250.0, dz1=250.0)
    config["simulation"].update(total_time_s=20.0, dt_s=1.0, output_interval_s=10.0,
                                idealized_test="2dbubble")
    # TNNUA carries moisture and no wind: no wind means the first step's
    # advection leaves th and qv as initialised, which is what CLEO is sent
    # at t = 0 and what check 3 compares against. The 2dbubble perturbation
    # (centred at 1 km; the idealized_test's own bubble sits at 5 km, above
    # this domain) gives th the x structure that check needs to see an x
    # mapping error.
    config["initial_conditions"]["source_file"] = \
        "./rundata/initial_conditions/profiles/default_cases/TNNUA.txt"
    config["initial_conditions"]["perturbation"] = "2dbubble"

    cleo = copy.deepcopy(template["physics"]["microphysics"]["cleo"])
    cleo["average_over_y"] = average_over_y
    cleo["timesteps_s"].update(condensation=0.1, collision=0.1, motion=0.1,
                               coupling=1.0, observation=10.0)
    cleo["superdroplets"].update(per_gridbox=8, seed_below_z_m=None)
    cleo["process"].update(enable_condensation=True, enable_collision=collisions,
                           collision_seed=1234)
    share = case_dir / "share"
    cleo["init_gbx_path"] = str(share / "dimlessGBxboundaries.dat")
    cleo["init_supers_path"] = str(share / "dimlessSDsinit.dat")
    cleo["output"]["zarrbasedir"] = "cleo.zarr"

    config["physics"] = {
        "p3": {"enable_p3": False},
        "rrtmgp": {"enable_rrtmgp": False},
        "turbulence": {"enable_turbulence": False},
        "surface": {"enable_surface": False},
        "microphysics": {"engine": "cleo", "cleo": cleo},
    }
    output_fields = set(FIELDS_3D + FIELDS_2D + ["pibar", "pbar"])
    # float64, so comparing runs is not at the mercy of float32 rounding.
    config["output"].update(engine="HDF5", output_dir=f"{case_dir / 'output'}/",
                            output_filename_prefix="cleo_model", output_initial_step=True,
                            precision="float64", fields_to_output=sorted(output_fields))
    return config


def prepare(case_dir, config, inputs_from=None):
    """Write the config and generate CLEO's binaries -- or, with inputs_from,
    reuse another case's, so two runs start from the same superdroplets."""
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    (case_dir / "share").mkdir(parents=True, exist_ok=True)
    config_path = case_dir / "config.json"
    if inputs_from is not None:
        source = json.loads((inputs_from / "config.json").read_text())
        source = source["physics"]["microphysics"]["cleo"]
        cleo = config["physics"]["microphysics"]["cleo"]
        for key in ("init_gbx_path", "init_supers_path"):
            cleo[key] = source[key]
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        return config_path
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    command = [sys.executable, str(ROOT / "tools" / "cleo" / "gen_cleo_inputs.py"),
               "-c", str(config_path)]
    if args.path2cleo is not None:
        command += ["--path2cleo", str(args.path2cleo)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (case_dir / "gen_cleo_inputs.log").write_text(result.stdout)
    if result.returncode != 0:
        raise AssertionError(f"gen_cleo_inputs.py failed:\n{result.stdout}")
    return config_path


def run_model(case_dir, config_path, ranks):
    command = [args.launcher, "--bind-to", "none", "-n", str(ranks)]
    if args.wrapper:
        command.append(args.wrapper)
    command += [str(args.executable.resolve()), str(config_path)]
    print("Running:", " ".join(command), flush=True)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=900, cwd=ROOT)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("CLEO model run timed out") from exc
    log = case_dir / "run.log"
    log.write_text(result.stdout)
    if result.returncode != 0:
        raise AssertionError(f"CLEO model run failed with return code {result.returncode}.\n"
                             f"===== VVM OUTPUT =====\n{result.stdout}\n"
                             f"======================\nFull log: {log}")


# =============================================================================
# Reading
# =============================================================================

def read_outputs(case_dir):
    """{model_time_s: {field: array}}, 3-D fields as (z, y, x), physical cells."""
    out = {}
    paths = sorted((case_dir / "output").glob("cleo_model_*.h5"))
    if not paths:
        raise AssertionError(f"no HDF5 output in {case_dir / 'output'}")
    for path in paths:
        with h5py.File(path, "r") as f:
            g = f["Step0"]
            time = float(np.asarray(g["model_time_s"]).squeeze())
            fields = {}
            for name in FIELDS_3D:
                if name not in g:
                    check(False, f"{path.name}: {name} missing from the output")
                    continue
                a = np.asarray(g[name], dtype=np.float64)
                if a.size != NZ * NY * NX:
                    raise AssertionError(f"{path.name}: {name} has shape {a.shape}, "
                                         f"expected {NZ}*{NY}*{NX} physical cells")
                fields[name] = a.reshape(NZ, NY, NX)
            for name in FIELDS_2D:
                if name not in g:
                    check(False, f"{path.name}: {name} missing from the output")
                    continue
                fields[name] = np.asarray(g[name], dtype=np.float64).reshape(NY, NX)
            for name in ("pibar", "pbar"):
                a = np.asarray(g[name], dtype=np.float64).ravel()
                if a.size != NZ:
                    raise AssertionError(f"{path.name}: {name} has {a.size} levels, expected {NZ}")
                fields[name] = a
            out[time] = fields
    return out


def read_zarr(case_dir):
    """CLEO's zarr store: raw arrays with CF scale factors applied."""
    import zarr
    store = case_dir / "output" / "cleo.zarr"
    if not store.is_dir():
        raise AssertionError(f"CLEO wrote no zarr store at {store}")
    group = zarr.open_group(str(store), mode="r")

    def var(name):
        a = group[name]
        scale = a.attrs.get("scale_factor", 1.0)
        return np.asarray(a[...], dtype=np.float64) * scale

    return {
        "time": var("time"),
        "gbxindex": np.asarray(group["gbxindex"][...], dtype=np.int64),
        "temp": var("temp"),      # K
        "press": var("press"),    # hPa
        "qvap": var("qvap"),      # g/kg
        "nsupers": np.asarray(group["nsupers"][...], dtype=np.int64),
    }


def gbx_to_cell(gbxindex):
    """CLEO's Cartesian gridbox index -> (k, j, i): z fastest, then x, then y."""
    k = gbxindex % NZ
    i = (gbxindex // NZ) % NX
    j = gbxindex // (NZ * NX)
    return k, j, i


# =============================================================================
# Checks
# =============================================================================

def check_fields(outputs, total_time):
    times = sorted(outputs)
    check(np.isclose(times[-1], total_time, rtol=0, atol=1e-9),
          f"last output at {times[-1]} s, expected total_time_s = {total_time} s")
    for time in times:
        f = outputs[time]
        for name, a in f.items():
            check(np.isfinite(a).all(), f"t={time}: non-finite values in {name}")
        for name in NONNEGATIVE:
            if name in f:
                check((f[name] >= 0).all(), f"t={time}: {name} has negative values "
                                            f"(min {f[name].min():.3e})")
        check((f["qi"] == 0).all(), f"t={time}: qi is not zero (CLEO is warm-phase)")
        check((f["diag_eff_radius_qi"] == 25.0).all(),
              f"t={time}: diag_eff_radius_qi moved off 25 um")
        for name in ("precip_ice_surf_mass", "precip_ice_surf_flux"):
            check((f[name] == 0).all(), f"t={time}: {name} is not zero")
        if time > 0:
            check((f["qp"] >= f["qc"] + f["qr"] - 1e-15).all(),
                  f"t={time}: qp < qc + qr somewhere; qp counts every superdroplet")
            check((f["qp"] > 0).all(), f"t={time}: qp is zero in a seeded gridbox")
        print(f"t={time:6.1f}: max qp {f['qp'].max():.3e}, max qc {f['qc'].max():.3e}, "
              f"max qr {f['qr'].max():.3e}, th {f['th'].min():.3f}..{f['th'].max():.3f}")


def check_sent_state(outputs, zarr_data):
    """CLEO's record of t = 0 against the model's initial fields."""
    f = outputs[min(outputs)]
    # Without horizontal structure in th an x/y mapping error would pass.
    check(np.ptp(f["th"], axis=2).max() > 0.1,
          "initial th has no x structure; the bubble is missing, so check 3 is blind to x")
    check(np.isclose(zarr_data["time"][0], 0.0, atol=1e-9),
          f"first CLEO observation is at {zarr_data['time'][0]} s, expected 0")
    k, j, i = gbx_to_cell(zarr_data["gbxindex"])
    check(sorted(zarr_data["gbxindex"].tolist()) == list(range(NX * NY * NZ)),
          "zarr gbxindex is not a permutation of 0 .. ngbxs-1")

    temp = f["th"][k, j, i] * f["pibar"][k]
    press_hpa = f["pbar"][k] / 100.0
    qvap_gkg = f["qv"][k, j, i] * 1000.0
    # zarr stores float32, so a few parts in 1e7 is all that can agree.
    for name, want in (("temp", temp), ("press", press_hpa), ("qvap", qvap_gkg)):
        got = zarr_data[name][0]
        bad = ~np.isclose(got, want, rtol=2e-6, atol=0)
        if bad.any():
            n = int(np.argmax(bad))
            check(False, f"CLEO {name} at t=0 differs from the model in {bad.sum()} of "
                         f"{bad.size} gridboxes; first: gbx {zarr_data['gbxindex'][n]} "
                         f"(k,j,i)=({k[n]},{j[n]},{i[n]}) CLEO {got[n]:.7g}, model {want[n]:.7g}")


def check_uniform_in_y(outputs):
    for time, f in sorted(outputs.items()):
        for name in CLEO_WRITTEN_3D:
            check((f[name] == f[name][:, :1, :]).all(),
                  f"t={time}: {name} differs between y columns with average_over_y")
        for name in ("precip_liq_surf_flux", "precip_liq_surf_mass"):
            check((f[name] == f[name][:1, :]).all(),
                  f"t={time}: {name} differs between y columns with average_over_y")


def compare_runs(a, b, label):
    for time in sorted(a):
        if time not in b:
            check(False, f"{label}: no output at t={time}")
            continue
        for name in FIELDS_3D + FIELDS_2D:
            x, y = a[time][name], b[time][name]
            # Superdroplets are summed per gridbox in whatever order a rank
            # holds them, so the last bits may differ; anything visible is a
            # decomposition error.
            ok = np.allclose(x, y, rtol=1e-9, atol=1e-15)
            if not ok:
                diff = np.abs(x - y)
                n = np.unravel_index(np.argmax(diff), diff.shape)
                check(False, f"{label}: {name} at t={time} differs, max |diff| {diff[n]:.3e} "
                             f"at {tuple(int(v) for v in n)} ({x[n]:.10g} vs {y[n]:.10g})")


def compare_zarr(a, b, label):
    order_a = np.argsort(a["gbxindex"])
    order_b = np.argsort(b["gbxindex"])
    for name in ("temp", "press", "qvap"):
        check(np.array_equal(a[name][0][order_a], b[name][0][order_b]),
              f"{label}: CLEO {name} at t=0 is not the same gridbox by gridbox")
    check(np.array_equal(a["nsupers"][0][order_a], b["nsupers"][0][order_b]),
          f"{label}: superdroplets per gridbox at t=0 differ (initial partition)")
    check(a["nsupers"][-1].sum() == b["nsupers"][-1].sum(),
          f"{label}: total superdroplets at the end differ "
          f"({a['nsupers'][-1].sum()} vs {b['nsupers'][-1].sum()})")


# =============================================================================

def main():
    if args.ranks == 1:
        case = work / "r1"
        config = make_config(case, collisions=True, average_over_y=True)
        run_model(case, prepare(case, config), 1)
        outputs = read_outputs(case)
        check_fields(outputs, config["simulation"]["total_time_s"])
        check_sent_state(outputs, read_zarr(case))
        check_uniform_in_y(outputs)
    else:
        runs = {}
        for ranks in (1, args.ranks):
            case = work / f"r{ranks}"
            # y may be split across ranks, so no averaging over it; collisions
            # off because the collision pool is per thread, not per gridbox.
            config = make_config(case, collisions=False, average_over_y=False)
            # One set of binaries for both runs: the comparison is about the
            # decomposition, not about how the superdroplets were sampled.
            inputs_from = work / "r1" if ranks != 1 else None
            run_model(case, prepare(case, config, inputs_from), ranks)
            runs[ranks] = (read_outputs(case), read_zarr(case))
            check_fields(runs[ranks][0], config["simulation"]["total_time_s"])
            check_sent_state(*runs[ranks])
        label = f"1 rank vs {args.ranks} ranks"
        compare_runs(runs[1][0], runs[args.ranks][0], label)
        compare_zarr(runs[1][1], runs[args.ranks][1], label)

    if failures:
        print(f"\n{len(failures)} check(s) failed; run directory {work}")
        return 1
    print(f"\nall checks passed; run directory {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
