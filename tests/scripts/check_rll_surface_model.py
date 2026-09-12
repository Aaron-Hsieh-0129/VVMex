#!/usr/bin/env python3

"""
Full-model smoke test for surface physics on regular latitude-longitude geometry.

The test verifies that:
  1. RLL accepts physics.surface_process.enable = true.
  2. SurfaceProcess is initialized through the normal Model path.
  3. The normal surface scheduler executes compute_coefficients().
  4. Surface exchange fields are finite and VEN2D is populated.
  5. Prognostic fields remain finite after surface tendencies are applied.
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


# Do not leave large core files behind if a device/MPI test crashes.
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
# Self-contained RLL + surface configuration
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

    # Two timesteps are intentional.
    #
    # SurfaceProcess currently retains the legacy scheduler. With
    # frequency_s = 120 s and dt = 60 s, surface coefficients are computed
    # on model step 1 rather than step 0. Therefore the final output proves
    # that compute_coefficients() was actually reached.
    "simulation": {
        "idealized_test": "jung2019_barotropic",
        "dt_s": 60.0,
        "total_time_s": 120.0,
        "output_interval_s": 120.0,
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

    # Feature under test.
    "physics": {
        "surface_process": {
            "enable": True,

            # Use a two-step interval deliberately. See simulation comment.
            "frequency_s": 120.0,

            "ocean_scheme": "sflux_2d",
            "land_scheme": "none",
        },
    },

    "output": {
        "engine": "HDF5",
        "output_dir": str(output_dir),
        "output_filename_prefix": "rll_surface_smoke",

        # Only inspect the state after SurfaceProcess has executed.
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
            "VEN2D",
            "sfc_flux_u",
            "sfc_flux_v",
            "sfc_flux_th",
            "sfc_flux_qv",
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
        "RLL surface model timed out"
    ) from exc


log_path = work / "run.log"
log_path.write_text(result.stdout)

if result.returncode != 0:
    raise AssertionError(
        "RLL surface model failed with "
        f"return code {result.returncode}; "
        f"see {log_path}"
    )


# ============================================================================
# Find final output
# ============================================================================

outputs = sorted(
    output_dir.glob(
        "rll_surface_smoke_*.h5"
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
# Validate final state
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
    "VEN2D",
    "sfc_flux_u",
    "sfc_flux_v",
    "sfc_flux_th",
    "sfc_flux_qv",
]


with h5py.File(output_path, "r") as file:

    if "Step0" not in file:
        raise AssertionError(
            "Output does not contain Step0"
        )

    group = file["Step0"]

    # ------------------------------------------------------------------------
    # Confirm that both timesteps completed.
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
        120.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError(
            "Model did not reach final time: "
            f"{final_time}"
        )

    # ------------------------------------------------------------------------
    # Every prognostic and surface field must exist and remain finite.
    # ------------------------------------------------------------------------

    data = {}

    for name in required_fields:

        if name not in group:
            raise AssertionError(
                f"Required field '{name}' "
                "missing from RLL surface output"
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
    # VEN2D is the strongest simple evidence that the scheduled surface
    # coefficient calculation actually executed. It starts at zero.
    # ------------------------------------------------------------------------

    ven2d_max = float(
        np.max(
            np.abs(data["VEN2D"])
        )
    )

    print(
        "VEN2D: "
        f"max(abs)={ven2d_max:.17e}"
    )

    if ven2d_max <= 0.0:
        raise AssertionError(
            "VEN2D was not populated; "
            "SurfaceProcess::compute_coefficients() "
            "may not have executed"
        )

    # Report flux magnitudes for diagnostic purposes without imposing
    # unnecessary physical regression thresholds.
    for name in (
        "sfc_flux_u",
        "sfc_flux_v",
        "sfc_flux_th",
        "sfc_flux_qv",
    ):

        maximum = float(
            np.max(
                np.abs(data[name])
            )
        )

        print(
            f"{name}: "
            f"max(abs)={maximum:.17e}"
        )


print(
    "PASS: full-model regular latitude-longitude "
    "surface smoke test"
)

print(
    f"Evidence: {work}"
)
