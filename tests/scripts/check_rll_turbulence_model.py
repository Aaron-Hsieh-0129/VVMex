#!/usr/bin/env python3

"""
Full-model smoke test for Shutts-Gray turbulence on regular latitude-longitude
geometry.

This test is intentionally independent of the existing Jung/RLL integration
tests.  It generates its own configuration and checks that:

  1. The RLL model accepts physics.turbulence.enable_turbulence = true.
  2. The normal Model::init() path initializes TurbulenceProcess.
  3. A complete model timestep executes.
  4. RKM and RKH are produced and are finite/non-negative.
  5. Prognostic fields remain finite.
  6. The model reaches the requested final time.
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

# Do not leave multi-GB core files behind if a device/MPI test crashes.
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
# Self-contained RLL turbulence configuration
#
# Do not load or modify any existing Jung test configuration.
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

    # One complete timestep is sufficient for this smoke test.
    "simulation": {
        "idealized_test": "jung2019_barotropic",
        "dt_s": 60.0,
        "total_time_s": 60.0,
        "output_interval_s": 60.0,
    },

    # Use the coupled Jung state so the model is not merely an all-zero
    # dynamics configuration.
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

            # This is only a smoke test, not the Jung reproduction test.
            "iteration": 40,
            "vertical_iterations": 10,
            "initial_iterations": 100,

            # 2 / (2*pi/32)^2
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
    # This is the feature under test.
    # =========================================================================

    "physics": {
        "turbulence": {
            "enable_turbulence": True,
        },
    },

    "output": {
        "engine": "HDF5",
        "output_dir": str(output_dir),
        "output_filename_prefix": "rll_turbulence_smoke",

        # We care about the state after turbulence has actually executed.
        # Avoid inspecting the initial RKM/RKH fields before the first
        # compute_coefficients() call.
        "output_initial_step": False,

        "fields_to_output": [
            "lon",
            "lat",
            "u",
            "v",
            "w",
            "xi",
            "eta",
            "zeta",
            "th",
            "qv",
            "RKM",
            "RKH",
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

        # Jung configuration currently uses the nonrotating RLL path.
        "OMEGA": 0.0,
    },
}


config_path = work / "config.json"

config_path.write_text(
    json.dumps(config, indent=2) + "\n"
)


# ============================================================================
# Execute the real VVM model
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
        timeout=180,
    )

except subprocess.TimeoutExpired as exc:
    raise AssertionError(
        "RLL turbulence model timed out"
    ) from exc


log_path = work / "run.log"

log_path.write_text(
    result.stdout
)


if result.returncode != 0:
    raise AssertionError(
        "RLL turbulence model failed with "
        f"return code {result.returncode}; "
        f"see {log_path}"
    )


# ============================================================================
# Find output
# ============================================================================

outputs = sorted(
    output_dir.glob(
        "rll_turbulence_smoke_*.h5"
    )
)

if len(outputs) != 1:
    raise AssertionError(
        "Expected exactly one final HDF5 output, "
        f"found {len(outputs)} in {output_dir}"
    )


output_path = outputs[0]

print(f"Checking {output_path}")


# ============================================================================
# Validate final model state
# ============================================================================

required_fields = [
    "u",
    "v",
    "w",
    "xi",
    "eta",
    "zeta",
    "th",
    "qv",
    "RKM",
    "RKH",
]


with h5py.File(output_path, "r") as file:

    if "Step0" not in file:
        raise AssertionError(
            "Output does not contain Step0"
        )

    group = file["Step0"]

    # ------------------------------------------------------------------------
    # Model actually reached the requested time.
    # ------------------------------------------------------------------------

    if "model_time_s" not in group:
        raise AssertionError(
            "model_time_s missing from output"
        )

    final_time = float(
        np.asarray(
            group["model_time_s"]
        ).squeeze()
    )

    if not np.isclose(
        final_time,
        60.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError(
            "Model did not reach final time: "
            f"{final_time}"
        )

    # ------------------------------------------------------------------------
    # Required fields are present and finite.
    #
    # RKM/RKH are particularly important:
    # their presence demonstrates that TurbulenceProcess was constructed,
    # and their finite values demonstrate compute_coefficients() executed.
    # ------------------------------------------------------------------------

    data = {}

    for name in required_fields:

        if name not in group:
            raise AssertionError(
                f"Required field '{name}' "
                "missing from RLL turbulence output"
            )

        array = np.asarray(
            group[name]
        )

        data[name] = array

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
                f"Nonfinite value in {name}; "
                f"first bad index = {first}"
            )

    # ------------------------------------------------------------------------
    # Eddy coefficients must be physically admissible.
    # ------------------------------------------------------------------------

    for name in ("RKM", "RKH"):

        array = data[name]

        minimum = float(
            np.min(array)
        )

        maximum = float(
            np.max(array)
        )

        print(
            f"{name}: "
            f"min={minimum:.17e}, "
            f"max={maximum:.17e}"
        )

        if minimum < 0.0:
            raise AssertionError(
                f"{name} contains negative values: "
                f"minimum = {minimum}"
            )

        # If TurbulenceProcess never computed the coefficients, an
        # all-zero field would be suspicious.  The Shutts-Gray implementation
        # has a positive minimum diffusivity in active cells.
        if maximum <= 0.0:
            raise AssertionError(
                f"{name} was not populated: "
                f"maximum = {maximum}"
            )

    # ------------------------------------------------------------------------
    # Simple gross-instability guard.
    #
    # Not intended as a physical regression threshold.  This catches obvious
    # indexing/metric failures that generate enormous but finite coefficients.
    # ------------------------------------------------------------------------

    for name in ("RKM", "RKH"):

        maximum = float(
            np.max(data[name])
        )

        if maximum > 1.0e8:
            raise AssertionError(
                f"Unreasonably large {name}: "
                f"{maximum}"
            )


print(
    "PASS: full-model regular latitude-longitude "
    "turbulence smoke test"
)

print(
    f"Evidence: {work}"
)
