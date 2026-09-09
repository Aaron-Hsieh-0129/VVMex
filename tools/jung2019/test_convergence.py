#!/usr/bin/env python3
"""Independent synthetic checks of the production L2 analysis command.

Run: python -m unittest discover -s tools/jung2019 -p test_convergence.py -v
Temporary fixtures are compressed and removed by TemporaryDirectory.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np


class ConvergenceAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jung-l2-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = []
        for nx, error in [(400, .4), (800, .2), (1600, .1), (3200, .05), (6400, 0.)]:
            directory = self.root / str(nx)
            (directory / "output").mkdir(parents=True)
            config = {"grid": {"horizontal": {"nx": nx, "ny": nx//4, "geometry": {
                "earth_radius_m": 6371220., "longitude_bounds_deg": [0., 360.],
                "latitude_bounds_deg": [-45., 45.]}}},
                "initial_conditions": {"jung2019": {"case": 1}},
                "simulation": {"total_time_s": 604800., "dt_s": 120000./nx}}
            (directory / "config.json").write_text(json.dumps(config))
            # Nonuniform separable field sampled at positive-face native Z
            # coordinates. An off-by-one fine-grid index changes the norm.
            x = (np.arange(nx)+1)/nx
            y = (np.arange(nx//4)+1)/(nx//4)
            field = (1+y[:, None]+x[None, :]**2)*(1+error)
            with h5py.File(directory / "output/jung_final.h5", "w") as file:
                group = file.create_group("Step0")
                group["model_time_s"] = 604800.
                group.create_dataset("rll_zeta_top", data=field, compression="gzip")
            self.runs.append(directory)

    def invoke(self):
        return subprocess.run([sys.executable, str(Path(__file__).with_name("convergence.py")),
            "--reference", str(self.runs[-1]), "--runs", *map(str, self.runs[:-1]),
            "--output", str(self.root / "analysis")], capture_output=True, text=True)

    def test_known_relative_errors_and_native_alignment(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root / "analysis/convergence.json").read_text())
        np.testing.assert_allclose([r["relative_l2"][-1] for r in report["runs"]],
                                   [.4, .2, .1, .05], rtol=1e-13, atol=1e-15)
        self.assertTrue(report["decreasing_final_l2"])

    def test_rejects_nonfinite_reference(self):
        with h5py.File(self.runs[-1] / "output/jung_final.h5", "r+") as file:
            file["Step0/rll_zeta_top"][0, 0] = np.nan
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Nonfinite native vorticity", result.stderr)

    def test_rejects_missing_final_time(self):
        with h5py.File(self.runs[0] / "output/jung_final.h5", "r+") as file:
            file["Step0/model_time_s"][()] = 0.
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing configured final output", result.stderr)

    def test_nonmonotone_errors_fail_without_relaxing_assessment(self):
        with h5py.File(self.runs[-2] / "output/jung_final.h5", "r+") as file:
            value = file["Step0/rll_zeta_top"]
            value[...] = 2*np.asarray(value)
        result = self.invoke()
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads((self.root / "analysis/convergence.json").read_text())
        self.assertFalse(report["decreasing_final_l2"])


if __name__ == "__main__":
    unittest.main()
