#!/usr/bin/env python3
"""Compare short RLL terrain histories across CPU/CUDA or MPI decompositions."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--report", type=Path)
args = parser.parse_args()
reference = sorted(args.reference.glob("mountain_*.h5"))
candidate = sorted(args.candidate.glob("mountain_*.h5"))
assert reference and [p.name for p in reference] == [p.name for p in candidate], "Missing/mismatched snapshots"
errors = {}
for a, b in zip(reference, candidate):
    with h5py.File(a) as fa, h5py.File(b) as fb:
        da, db = fa["Step0"], fb["Step0"]
        for name in ("model_time_s", "rll_terrain_height", "ITYPEU", "ITYPEV", "ITYPEW"):
            np.testing.assert_array_equal(da[name], db[name])
        for name in ("u", "v", "w", "xi", "eta", "zeta", "u_topo", "v_topo", "w_topo", "xi_topo", "eta_topo"):
            tolerance = 1e-10 if name[0] in "uvw" else 1e-15
            error = float(np.max(np.abs(np.asarray(da[name])-np.asarray(db[name]))))
            assert np.isfinite(error) and error <= tolerance, (a.name, name, error, tolerance)
            errors[name] = max(errors.get(name, 0), error)
report = {"reference": str(args.reference.resolve()), "candidate": str(args.candidate.resolve()), "snapshots": len(reference), "max_absolute_errors": errors}
text = json.dumps(report, indent=2)+"\n"
if args.report:
    args.report.write_text(text)
print(text, end="")
print("PASS: unchanged roundoff-scale wind/vorticity thresholds")
