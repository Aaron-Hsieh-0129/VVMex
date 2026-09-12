#!/usr/bin/env python3

"""
Full-model smoke test for combined Shutts-Gray turbulence and surface physics
on regular latitude-longitude geometry.

This test checks that:
  1. RLL accepts turbulence and surface physics enabled together.
  2. The normal Model::init() path initializes both processes.
  3. Two complete model timesteps execute.
  4. RKM and RKH are produced and remain finite/non-negative.
  5. SurfaceProcess actually computes VEN2D and surface fluxes.
  6. Turbulence and surface physics can share the FE tendency buffers.
  7. Prognostic fields remain finite.
  8. The model reaches the requested final time.
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
# Self-contained RLL turbulence + surface configuration
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
    # With dt = 60 s and surface frequency = 120 s, the current legacy
    # surface scheduler performs the first coefficient calculation on step 1.
    # Therefore the final output confirms that SurfaceProcess actually ran.
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

            # Smoke-test settings rather than a reproduction-quality
            # Jung integration.
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
    # Features under test
    # =========================================================================

    "physics": {
        "turbulence": {
            "enable_turbulence": True,
        },

        "surface_process": {
            "enable": True,

            # Deliberately use a two-step interval. With the existing legacy
            # scheduler this causes compute_coefficients() to execute on step 1.
            "frequency_s": 120.0,

            "ocean_scheme": "sflux_2d",
            "land_scheme": "none",
        },
    },

    "output": {
        "engine": "HDF5",
        "output_dir": str(output_dir),
        "output_filename_prefix": "rll_turbulence_surface_smoke",

        # Inspect only the state after both physics packages have executed.
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

            # Turbulence diagnostics
            "RKM",
            "RKH",

            # Surface diagnostics
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
        timeout=240,
    )

except subprocess.TimeoutExpired as exc:
    raise AssertionError(
        "RLL turbulence + surface model timed out"
    ) from exc


log_path = work / "run.log"

log_path.write_text(
    result.stdout
)


if result.returncode != 0:
    raise AssertionError(
        "RLL turbulence + surface model failed with "
        f"return code {result.returncode}.\n\n"
        "===== VVM OUTPUT =====\n"
        f"{result.stdout}\n"
        "======================\n"
        f"Full log: {log_path}"
    )


# ============================================================================
# Find final output
# ============================================================================

outputs = sorted(
    output_dir.glob(
        "rll_turbulence_surface_smoke_*.h5"
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
    # Required fields must exist and remain finite.
    # ------------------------------------------------------------------------

    data = {}

    for name in required_fields:

        if name not in group:
            raise AssertionError(
                f"Required field '{name}' "
                "missing from RLL turbulence + surface output"
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
    # Turbulence must actually have executed.
    #
    # RKM/RKH should be finite, non-negative, and populated.
    # ------------------------------------------------------------------------

    for name in (
        "RKM",
        "RKH",
    ):

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

        if maximum <= 0.0:
            raise AssertionError(
                f"{name} was not populated: "
                f"maximum = {maximum}"
            )

        # Gross-instability guard only; this is not intended as a
        # physical regression threshold.
        if maximum > 1.0e8:
            raise AssertionError(
                f"Unreasonably large {name}: "
                f"{maximum}"
            )

    # ------------------------------------------------------------------------
    # SurfaceProcess must actually have executed.
    #
    # VEN2D is created during SurfaceProcess initialization and written by
    # compute_coefficients(). A non-zero final value therefore demonstrates
    # that the scheduled coefficient calculation was reached.
    # ------------------------------------------------------------------------

    ven2d_min = float(
        np.min(data["VEN2D"])
    )

    ven2d_max = float(
        np.max(data["VEN2D"])
    )

    ven2d_abs_max = float(
        np.max(
            np.abs(data["VEN2D"])
        )
    )

    print(
        "VEN2D: "
        f"min={ven2d_min:.17e}, "
        f"max={ven2d_max:.17e}"
    )

    if ven2d_abs_max <= 0.0:
        raise AssertionError(
            "VEN2D was not populated; "
            "SurfaceProcess::compute_coefficients() "
            "may not have executed"
        )

    # ------------------------------------------------------------------------
    # Surface fluxes need to remain finite.
    #
    # Do not require every flux to be non-zero because some thermodynamic
    # fluxes may legitimately vanish for this idealized Jung state.
    # ------------------------------------------------------------------------

    for name in (
        "sfc_flux_u",
        "sfc_flux_v",
        "sfc_flux_th",
        "sfc_flux_qv",
    ):

        minimum = float(
            np.min(data[name])
        )

        maximum = float(
            np.max(data[name])
        )

        abs_maximum = float(
            np.max(
                np.abs(data[name])
            )
        )

        print(
            f"{name}: "
            f"min={minimum:.17e}, "
            f"max={maximum:.17e}, "
            f"max(abs)={abs_maximum:.17e}"
        )


print(
    "PASS: full-model regular latitude-longitude "
    "turbulence + surface smoke test"
)

print(
    f"Evidence: {work}"
)
