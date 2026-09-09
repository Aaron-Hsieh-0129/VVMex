#!/usr/bin/env python3
"""Record geographic and boundary evidence directly from the CVVM archive."""
import argparse
import hashlib
import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=Path)
parser.add_argument("report", type=Path)
args = parser.parse_args()
rows = []
for path in sorted(args.archive.glob("*L.*.nc")):
    with Dataset(path) as file:
        row = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "first_longitude_deg": float(file["lont"][0, 0]),
               "latitude_extrema_deg": [float(file["latt"][:].min()), float(file["latt"][:].max())]}
        for name in ("u3dx", "u3dy", "z3dz"):
            data = np.asarray(file[name][-1])
            row[name] = {side: {"min": float(data[j].min()), "max": float(data[j].max()), "mean": float(data[j].mean())}
                         for side, j in (("southernmost_T_row", 0), ("northernmost_T_row", -1))}
        psi = np.asarray(file["z3dz"][0])
        row["psi_boundary_row_longitude_range_m2_s"] = [float(np.ptp(psi[j])) for j in (0, -1)]
        rows.append(row)
if not rows:
    raise ValueError("No archived time outputs found")
report = {"interpretation": "Rows are output T points, not exactly wall faces. Lower z3dz is psi by CODE_BAR/ldoutput_nc.F. Combined with uniform-copy psi halos in bound_extra.F, nonzero cross-channel wind is inconsistent with impermeable walls.", "outputs": rows}
args.report.write_text(json.dumps(report, indent=2) + "\n")
print(args.report)
