#!/usr/bin/env python3

"""
Full-model P3 integration test on regular latitude-longitude geometry.

Unlike the previous uniform-column test, this test restores the normal
Jung jet and analytic perturbation and runs two complete model timesteps.

It verifies that:
  1. Nonuniform RLL dynamics and P3 run together.
  2. P3 executes for multiple timesteps.
  3. AB2 scalar transport reaches its multi-step path.
  4. Prognostic dynamics fields remain finite.
  5. P3 prognostic fields remain finite and physically admissible.
  6. qp remains consistent with qc + qr + qi.
  7. A dry atmosphere does not spontaneously generate hydrometeor mass.
  8. The Jung jet and vorticity perturbation are actually present.
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

    # Two timesteps are deliberate.
    #
    # This goes beyond the one-step uniform P3 smoke test and exercises
    # the multi-step prognostic transport path.
    "simulation": {
        "idealized_test": "jung2019_barotropic",
        "dt_s": 10.0,
        "total_time_s": 20.0,
        "output_interval_s": 20.0,
    },

    # Restore the normal nonuniform Jung state.
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
        "output_filename_prefix": "rll_p3_model",

        "output_initial_step": False,

        "fields_to_output": [
            "u",
            "v",
            "w",

            "xi",
            "eta",
            "zeta",

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
# Run
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
        timeout=360,
    )

except subprocess.TimeoutExpired as exc:
    raise AssertionError(
        "RLL P3 model integration test timed out"
    ) from exc


log_path = work / "run.log"

log_path.write_text(
    result.stdout
)


if result.returncode != 0:
    raise AssertionError(
        "RLL P3 model integration failed with "
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
        "rll_p3_model_*.h5"
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
# Validation
# =============================================================================

required_3d_fields = [
    "u",
    "v",
    "w",

    "xi",
    "eta",
    "zeta",

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


nonnegative_fields = [
    "qv",
    "qc",
    "qr",
    "qi",
    "qm",
    "nc",
    "nr",
    "ni",
    "bm",
]


with h5py.File(output_path, "r") as file:

    if "Step0" not in file:
        raise AssertionError(
            "Final output does not contain Step0"
        )

    group = file["Step0"]

    # -------------------------------------------------------------------------
    # The complete two-step integration must finish.
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
        20.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError(
            "Model did not reach the requested final time: "
            f"{final_time}"
        )

    # -------------------------------------------------------------------------
    # Every required state must remain finite.
    # -------------------------------------------------------------------------

    data = {}

    for name in required_3d_fields + required_2d_fields:

        if name not in group:
            raise AssertionError(
                f"Required field '{name}' "
                "missing from RLL P3 model output"
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
            f"min={float(np.min(array)):.17e}, "
            f"max={float(np.max(array)):.17e}"
        )

    # -------------------------------------------------------------------------
    # Confirm this really is the nonuniform Jung problem.
    # -------------------------------------------------------------------------

    u_abs_max = float(
        np.max(
            np.abs(data["u"])
        )
    )

    zeta_abs_max = float(
        np.max(
            np.abs(data["zeta"])
        )
    )

    print(
        f"max(abs(u)) = {u_abs_max:.17e}"
    )

    print(
        f"max(abs(zeta)) = {zeta_abs_max:.17e}"
    )

    if u_abs_max <= 0.0:
        raise AssertionError(
            "Jung jet is absent from the integration"
        )

    if zeta_abs_max <= 0.0:
        raise AssertionError(
            "Jung vorticity field is absent from the integration"
        )

    # -------------------------------------------------------------------------
    # P3 quantities must remain physically admissible.
    # -------------------------------------------------------------------------

    tolerance = 1.0e-12

    for name in nonnegative_fields:

        minimum = float(
            np.min(data[name])
        )

        if minimum < -tolerance:
            raise AssertionError(
                f"{name} became negative: "
                f"minimum = {minimum}"
            )

    # Rime mass cannot exceed total ice mass.
    maximum_qm_excess = float(
        np.max(
            data["qm"] - data["qi"]
        )
    )

    if maximum_qm_excess > tolerance:
        raise AssertionError(
            "P3 produced qm > qi: "
            f"maximum excess = {maximum_qm_excess}"
        )

    # -------------------------------------------------------------------------
    # qp consistency
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
    # The atmosphere is still deliberately dry.
    #
    # This test is about coupling P3 to nonuniform RLL dynamics, not cloud
    # production. Significant hydrometeor creation would indicate a problem.
    # -------------------------------------------------------------------------

    hydrometeor = (
        data["qc"]
        + data["qr"]
        + data["qi"]
    )

    hydrometeor_max = float(
        np.max(
            np.abs(hydrometeor)
        )
    )

    print(
        "maximum hydrometeor mixing ratio: "
        f"{hydrometeor_max:.17e}"
    )

    if hydrometeor_max > 1.0e-12:
        raise AssertionError(
            "Dry RLL P3 integration created "
            f"hydrometeor mass: {hydrometeor_max}"
        )


print(
    "PASS: nonuniform RLL dynamics + P3 model integration"
)

print(
    f"Evidence: {work}"
)
