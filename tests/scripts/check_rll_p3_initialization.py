#!/usr/bin/env python3

"""
Initialization-only smoke test for P3 on regular latitude-longitude geometry.

This test deliberately runs zero model timesteps.

It verifies that:
  1. RLL accepts physics.p3.enable_p3 = true.
  2. P3 can be constructed without an external initial-condition file.
  3. Model::init() successfully initializes P3.
  4. P3 prognostic and diagnostic State fields are created.
  5. All initialized fields are finite.

P3::run() is intentionally NOT exercised here. That belongs to the next
hierarchical validation step.
"""

import argparse
import json
import resource
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "executable",
    type=Path,
    help="Path to the VVM executable",
)

parser.add_argument(
    "work",
    type=Path,
    help="Directory used for temporary test files",
)

parser.add_argument(
    "--wrapper",
    default="",
    help="Optional GPU wrapper such as one_gpu_per_rank.sh",
)

parser.add_argument(
    "--launcher",
    default="mpiexec",
    help="MPI launcher",
)

args = parser.parse_args()


resource.setrlimit(
    resource.RLIMIT_CORE,
    (0, 0),
)


args.work.mkdir(
    parents=True,
    exist_ok=True,
)

work = Path(
    tempfile.mkdtemp(
        prefix="run-",
        dir=args.work,
    )
)

output_dir = work / "output"


# ============================================================================
# Self-contained RLL + P3 initialization configuration
# ============================================================================

config = {
    "grid": {
        "horizontal": {
            "nx": 32,
            "ny": 12,
            "n_halo_cells": 2,

            "geometry": {
                "kind": "regular_latlon",
                "earth_radius_m": 6371220.0,

                "longitude_bounds_deg": [
                    0.0,
                    360.0,
                ],

                "latitude_bounds_deg": [
                    -30.0,
                    30.0,
                ],
            },

            "topology": {
                "q1": "periodic",
                "q2": "bounded",
            },
        },

        "vertical": {
            "nz": 8,
            "type": "default",
            "dz": 1000.0,
            "dz1": 1000.0,
        },
    },

    # total_time_s = 0 is intentional.
    #
    # main.cpp calls Model::init(), writes the initial state, and only then
    # enters the timestep loop. Therefore this isolates P3 initialization
    # from P3::run().
    "simulation": {
        "idealized_test": "jung2019_barotropic",
        "dt_s": 60.0,
        "total_time_s": 0.0,
        "output_interval_s": 60.0,
    },

    "initial_conditions": {
        "jung2019": {
            "case": 1,
            "jet_scale": 1.0,
            "perturbation_scale": 1.0,
        },
    },

    "dynamics": {
        "solver": {
            "w_solver_method": "tridiagonal",
            "iteration": 40,
            "vertical_iterations": 10,
            "initial_iterations": 100,
            "WRXMU": 51.87644602487694,
        },

        "prognostic_variables": {
            "xi": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },

                    "stretching": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },

                    "twisting": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "eta": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },

                    "stretching": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },

                    "twisting": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "zeta": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },

                    "stretching": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },

                    "twisting": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },
        },
    },

    # =========================================================================
    # Feature under test
    # =========================================================================

    "physics": {
        "p3": {
            "enable_p3": True,
            "make_lookup_table": False,
            "do_predict_nc": True,
            "do_prescribed_ccn": False,
            "max_total_ni": 2000000.0,
        },
    },

    "output": {
        "engine": "HDF5",
        "output_dir": str(output_dir),
        "output_filename_prefix": "rll_p3_initialization",

        # We specifically want the state immediately after Model::init().
        "output_initial_step": True,

        "fields_to_output": [
            "lon",
            "lat",

            "th",
            "qv",

            # P3 prognostic fields.
            "qc",
            "qr",
            "qi",
            "qm",
            "nc",
            "nr",
            "ni",
            "bm",

            # Created specifically during VVM_P3_Interface::initialize().
            "qp",

            # P3 surface precipitation state created during initialize().
            "precip_liq_surf_mass",
            "precip_ice_surf_mass",
            "precip_liq_surf_flux",
            "precip_ice_surf_flux",
        ],

        "output_grid": {
            "x_start": 0,
            "x_end": -1,
            "y_start": 0,
            "y_end": -1,
            "z_start": 0,
            "z_end": -1,
        },
    },

    "constants": {
        "gravity": 9.80616,
        "Rd": 287.04,
        "PSFC": 100000.0,
        "P0": 100000.0,
        "Cp": 1004.5,
        "Lv": 2500000.0,
        "OMEGA": 0.0,
    },
}


config_path = work / "config.json"

config_path.write_text(
    json.dumps(config, indent=2) + "\n"
)


# ============================================================================
# Run VVM
# ============================================================================

command = [
    args.launcher,
    "--bind-to",
    "none",
    "-n",
    "1",
]

if args.wrapper:
    command.append(args.wrapper)

command.extend(
    [
        str(args.executable.resolve()),
        str(config_path),
    ]
)


print("Running:")
print(" ".join(command))
print(f"Working directory: {work}")


try:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=240,
    )

except subprocess.TimeoutExpired as exc:
    raise AssertionError(
        "RLL P3 initialization test timed out"
    ) from exc


log_path = work / "run.log"

log_path.write_text(
    result.stdout
)


if result.returncode != 0:
    raise AssertionError(
        "RLL P3 initialization failed with "
        f"return code {result.returncode}.\n\n"
        "===== VVM OUTPUT =====\n"
        f"{result.stdout}\n"
        "======================\n"
        f"Full log: {log_path}"
    )


# ============================================================================
# Find initial output
# ============================================================================

outputs = sorted(
    output_dir.glob(
        "rll_p3_initialization_*.h5"
    )
)

if len(outputs) != 1:
    raise AssertionError(
        "Expected exactly one initial HDF5 output, "
        f"found {len(outputs)} in {output_dir}"
    )


output_path = outputs[0]

print(f"Checking {output_path}")


# ============================================================================
# Validate P3 initialization
# ============================================================================

required_3d_fields = [
    "th",
    "qv",
    "qc",
    "qr",
    "qi",
    "qm",
    "nc",
    "nr",
    "ni",
    "bm",
    "qp",
]

required_2d_fields = [
    "precip_liq_surf_mass",
    "precip_ice_surf_mass",
    "precip_liq_surf_flux",
    "precip_ice_surf_flux",
]


with h5py.File(output_path, "r") as file:

    if "Step0" not in file:
        raise AssertionError(
            "Initial output does not contain Step0"
        )

    group = file["Step0"]

    if "model_time_s" not in group:
        raise AssertionError(
            "model_time_s missing from initial output"
        )

    model_time = float(
        np.asarray(
            group["model_time_s"]
        ).squeeze()
    )

    if not np.isclose(
        model_time,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError(
            "Initialization-only test unexpectedly "
            f"advanced model time to {model_time}"
        )

    for name in required_3d_fields + required_2d_fields:

        if name not in group:
            raise AssertionError(
                f"Required initialized P3 field "
                f"'{name}' is missing"
            )

        array = np.asarray(
            group[name]
        )

        if not np.isfinite(array).all():

            bad = np.argwhere(
                ~np.isfinite(array)
            )

            first = (
                tuple(bad[0])
                if bad.size
                else None
            )

            raise AssertionError(
                f"Nonfinite initialized value in {name}; "
                f"first bad index = {first}"
            )

        print(
            f"{name}: "
            f"shape={array.shape}, "
            f"min={float(np.min(array)):.17e}, "
            f"max={float(np.max(array)):.17e}"
        )


print(
    "PASS: regular latitude-longitude P3 initialization"
)

print(
    f"Evidence: {work}"
)
