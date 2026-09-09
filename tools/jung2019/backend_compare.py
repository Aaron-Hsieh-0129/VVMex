#!/usr/bin/env python3
"""Check saved short-run CPU/MPI/CUDA parity using FP64 roundoff bounds."""
import argparse
import json
from pathlib import Path

import numpy as np
from experiment import read_output


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("runs", type=Path, nargs="+")
args = parser.parse_args()
baseline = sorted((args.reference / "output").glob("jung_*.h5"))
if not baseline:
    raise ValueError("Missing baseline output")
reports = []
passed = True
for directory in args.runs:
    paths = sorted((directory / "output").glob("jung_*.h5"))
    if len(paths) != len(baseline):
        raise ValueError("Output counts differ")
    for first, second in zip(baseline, paths):
        ref, val = read_output(first), read_output(second)
        if not np.array_equal(ref["model_time_s"], val["model_time_s"]):
            raise ValueError("Output times differ")
        errors = {}
        for name in ("u", "v", "w", "xi", "eta", "zeta"):
            if ref[name].shape != val[name].shape:
                raise ValueError("Field shapes differ")
            # These bounds qualify short FP64 backend/reduction parity only,
            # not agreement with the paper or nonlinear long-run trajectories.
            tolerance = 1e-10 if name in ("u", "v", "w") else 1e-15
            error = float(np.max(np.abs(val[name] - ref[name])))
            ok = bool(np.isfinite(error) and error <= tolerance)
            passed &= ok
            errors[name] = {"absolute_max": error, "tolerance": tolerance, "passed": ok}
        reports.append({"output": str(second), "fields": errors})
report = {"reference": str(args.reference), "passed": passed, "outputs": reports}
destination = args.reference / "backend_comparison.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
print("PASS" if passed else "FAIL")
raise SystemExit(0 if passed else 1)
