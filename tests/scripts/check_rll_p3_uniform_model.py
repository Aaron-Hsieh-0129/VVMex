#!/usr/bin/env python3

"""
One-step P3 execution test on a horizontally uniform dry RLL state.

This is deliberately the simplest possible runtime P3 test.

The Jung jet and perturbation are both disabled, so the initial state is
horizontally uniform and dry. P3 is then executed for one timestep.

The test verifies that:
  1. Model::run_step() actually reaches P3::run().
  2. P3 preprocessing/packing succeeds on RLL.
  3. p3_main() completes.
  4. P3 postprocessing/unpacking succeeds.
  5. Thermodynamic and microphysical fields remain finite.
  6. Water species remain non-negative.
  7. qp remains consistent with qc + qr + qi.
  8. The model reaches the requested final time.

This test does NOT yet introduce a moist perturbation.
"""

import argparse
import json
import resource
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

# =============================================================================
# Arguments
# =============================================================================

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


# =============================================================================
# Configuration
# =============================================================================

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

    # Exactly one timestep.
    "simulation": {
        "idealized_test": "jung2019_barotropic",
        "dt_s": 10.0,
        "total_time_s": 10.0,
        "output_interval_s": 10.0,
    },

    # Disable all Jung horizontal structure.
    #
    # jet_scale = 0:
    #     u = 0
    #     background zeta = 0
    #
    # perturbation_scale = 0:
    #     analytic zeta perturbation = 0
    #
    # This gives P3 a horizontally uniform dry state.
    "initial_conditions": {
        "jung2019": {
            "case": 1,
            "jet_scale": 0.0,
            "perturbation_scale": 0.0,
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
            "th": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "qv": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "qc": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "qr": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "qi": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "qm": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "nc": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "nr": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "ni": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

            "bm": {
                "tendency_terms": {
                    "advection": {
                        "enable": True,
                        "spatial_scheme": "Takacs",
                        "temporal_scheme": "AdamsBashforth2",
                    },
                },
            },

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
        "output_filename_prefix": "rll_p3_uniform",

        # We only want the state after P3 has executed.
        "output_initial_step": False,

        "fields_to_output": [
            "u",
            "v",
            "w",

            "th",
            "qv",
            "T",

            "qc",
            "qr",
            "qi",
            "qm",

            "nc",
            "nr",
            "ni",
            "bm",

            "qp",

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


# =============================================================================
# Run model
# =============================================================================

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
        timeout=300,
    )

except subprocess.TimeoutExpired as exc:
    raise AssertionError(
        "RLL P3 uniform-model test timed out"
    ) from exc


log_path = work / "run.log"

log_path.write_text(
    result.stdout
)


if result.returncode != 0:
    raise AssertionError(
        "RLL P3 uniform-model test failed with "
        f"return code {result.returncode}.\n\n"
        "===== VVM OUTPUT =====\n"
        f"{result.stdout}\n"
        "======================\n"
        f"Full log: {log_path}"
    )


# =============================================================================
# Find output
# =============================================================================

outputs = sorted(
    output_dir.glob(
        "rll_p3_uniform_*.h5"
    )
)


if len(outputs) != 1:
    raise AssertionError(
        "Expected exactly one final HDF5 output, "
        f"found {len(outputs)} in {output_dir}"
    )


output_path = outputs[0]

print(f"Checking {output_path}")


# =============================================================================
# Validate
# =============================================================================

required_3d_fields = [
    "u",
    "v",
    "w",

    "th",
    "qv",
    "T",

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


mass_fields = [
    "qv",
    "qc",
    "qr",
    "qi",
    "qm",
]


number_fields = [
    "nc",
    "nr",
    "ni",
]


with h5py.File(output_path, "r") as file:

    if "Step0" not in file:
        raise AssertionError(
            "Final output does not contain Step0"
        )

    group = file["Step0"]

    # -------------------------------------------------------------------------
    # Model must have completed exactly one timestep.
    # -------------------------------------------------------------------------

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
        10.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError(
            "Model did not reach the requested final time: "
            f"{final_time}"
        )

    # -------------------------------------------------------------------------
    # Every required field must exist and remain finite.
    # -------------------------------------------------------------------------

    data = {}

    for name in required_3d_fields + required_2d_fields:

        if name not in group:
            raise AssertionError(
                f"Required field '{name}' missing "
                "from RLL P3 output"
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

        print(
            f"{name}: "
            f"shape={array.shape}, "
            f"min={float(np.min(array)):.17e}, "
            f"max={float(np.max(array)):.17e}"
        )

    # -------------------------------------------------------------------------
    # Water mass mixing ratios must not become significantly negative.
    # -------------------------------------------------------------------------

    tolerance = 1.0e-14

    for name in mass_fields:

        minimum = float(
            np.min(data[name])
        )

        if minimum < -tolerance:
            raise AssertionError(
                f"{name} became negative after P3: "
                f"minimum = {minimum}"
            )

    # -------------------------------------------------------------------------
    # Number concentrations must also remain non-negative.
    # -------------------------------------------------------------------------

    for name in number_fields:

        minimum = float(
            np.min(data[name])
        )

        if minimum < -tolerance:
            raise AssertionError(
                f"{name} became negative after P3: "
                f"minimum = {minimum}"
            )

    # -------------------------------------------------------------------------
    # qm is rime mass and must not exceed total ice mass qi.
    # -------------------------------------------------------------------------

    if np.any(
        data["qm"] > data["qi"] + tolerance
    ):
        maximum_excess = float(
            np.max(
                data["qm"] - data["qi"]
            )
        )

        raise AssertionError(
            "P3 produced qm > qi; "
            f"maximum excess = {maximum_excess}"
        )

    # -------------------------------------------------------------------------
    # qp is defined by the VVM-P3 interface as qc + qr + qi.
    # -------------------------------------------------------------------------

    qp_expected = (
        data["qc"]
        + data["qr"]
        + data["qi"]
    )

    qp_error = float(
        np.max(
            np.abs(
                data["qp"] - qp_expected
            )
        )
    )

    print(
        "qp consistency error: "
        f"{qp_error:.17e}"
    )

    if qp_error > 1.0e-12:
        raise AssertionError(
            "qp is inconsistent with qc + qr + qi: "
            f"max error = {qp_error}"
        )

    # -------------------------------------------------------------------------
    # This is a completely dry test. P3 must not spontaneously create a
    # meaningful amount of hydrometeor mass.
    #
    # Keep this threshold loose enough that numerical roundoff is irrelevant.
    # -------------------------------------------------------------------------

    total_hydrometeor_max = float(
        np.max(
            np.abs(
                data["qc"]
                + data["qr"]
                + data["qi"]
            )
        )
    )

    print(
        "maximum total hydrometeor mixing ratio: "
        f"{total_hydrometeor_max:.17e}"
    )

    if total_hydrometeor_max > 1.0e-12:
        raise AssertionError(
            "Dry P3 smoke test created hydrometeor mass: "
            f"{total_hydrometeor_max}"
        )


print(
    "PASS: one-step uniform dry P3 execution on RLL"
)

print(
    f"Evidence: {work}"
)
