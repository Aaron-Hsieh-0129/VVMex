#!/usr/bin/env python3
"""Generate VVM mesh metadata and Fides JSON files for ParaView.

The generated companion BP contains only static visualization coordinates and
surface connectivity. Simulation fields remain in the input BP file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import adios2
import numpy as np


DEFAULT_X = "coordinates/x"
DEFAULT_Y = "coordinates/y"
DEFAULT_Z = "coordinates/z_mid"
QV_TOPO_VARIABLE = "qv"
QV_TOPO_STEP = 1


class ExampleArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if message == "the following arguments are required: input":
            examples = (self.epilog or "") % {"prog": self.prog}
            self.print_usage(sys.stderr)
            self.exit(2, f"{self.prog}: error: {message}\n\n{examples}\n")
        super().error(message)


def parse_shape(metadata: dict[str, str]) -> tuple[int, ...]:
    text = metadata.get("Shape", "").strip()
    if not text:
        return ()
    return tuple(int(value.strip()) for value in text.split(","))


def is_numeric(metadata: dict[str, str]) -> bool:
    data_type = metadata.get("Type", "").lower()
    return any(token in data_type for token in ("float", "double", "int"))


def read_step(reader: adios2.FileReader, variable: str, step: int) -> np.ndarray:
    return np.asarray(reader.read(variable, step_selection=[step, 1]))


def read_first_step(reader: adios2.FileReader, variable: str) -> np.ndarray:
    return read_step(reader, variable, 0)


def read_string_attribute(
    reader: adios2.FileReader, attributes: dict[str, dict[str, str]], name: str
) -> str | None:
    if name not in attributes:
        return None
    value = reader.read_attribute_string(name)
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0]) if len(value) else None
    return str(value)


def validate_axis(axis: np.ndarray, variable: str) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).squeeze()
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{variable} must be a one-dimensional array with at least 2 values")
    if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{variable} must contain finite, strictly increasing values")
    return axis


def visualization_axis(axis: np.ndarray, origin_mode: str) -> tuple[np.ndarray, float]:
    if origin_mode == "native":
        return axis.copy(), 0.0
    offset = float(axis[axis.size // 2])
    return axis - offset, offset


def interfaces_from_midpoints(z_mid: np.ndarray, lower_interface: float) -> np.ndarray:
    """Reconstruct full interfaces from midpoints and a declared lower boundary."""
    z_mid = validate_axis(z_mid, DEFAULT_Z)
    interfaces = np.empty(z_mid.size + 1, dtype=np.float64)
    interfaces[0] = lower_interface
    for k, midpoint in enumerate(z_mid):
        interfaces[k + 1] = 2.0 * midpoint - interfaces[k]
    if not np.all(np.isfinite(interfaces)) or np.any(np.diff(interfaces) <= 0.0):
        raise ValueError(
            "Reconstructed z interfaces are not increasing; provide an interface "
            "variable or the correct --lower-interface"
        )
    return interfaces


def physical_height_from_index(
    topo: np.ndarray, interfaces: np.ndarray, index_offset: int
) -> np.ndarray:
    rounded = np.rint(topo).astype(np.int64)
    if not np.allclose(topo, rounded):
        raise ValueError("The topography index variable contains non-integer values")
    indices = rounded + index_offset
    if indices.min(initial=0) < 0 or indices.max(initial=0) >= interfaces.size:
        raise ValueError(
            "Topography indices fall outside the interface array after applying "
            f"--topo-index-offset={index_offset}: range={indices.min()}..{indices.max()}, "
            f"valid=0..{interfaces.size - 1}"
        )
    return interfaces[indices]


def vertical_interfaces(
    reader: adios2.FileReader,
    variables: dict[str, dict[str, str]],
    z_mid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    interface_variable = args.interface_variable
    if interface_variable is None:
        for candidate in ("coordinates/z_interface_full", "coordinates/z_interface"):
            if candidate in variables:
                interface_variable = candidate
                break
    if interface_variable is not None:
        if interface_variable not in variables:
            raise ValueError(f"Interface variable {interface_variable!r} does not exist")
        interfaces = validate_axis(read_first_step(reader, interface_variable), interface_variable)
        if interfaces.size != z_mid.size + 1:
            raise ValueError(
                f"{interface_variable} has {interfaces.size} values; expected {z_mid.size + 1}"
            )
        return interfaces, interface_variable

    interfaces = interfaces_from_midpoints(z_mid, args.lower_interface)
    source = f"reconstructed from {args.z_variable}, lower={args.lower_interface:g} m"
    return interfaces, source


def surface_height_from_qv_zeros(
    reader: adios2.FileReader,
    variables: dict[str, dict[str, str]],
    expected_shape: tuple[int, int],
    z_mid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_qv_shape = (z_mid.size, *expected_shape)
    if QV_TOPO_VARIABLE not in variables:
        raise ValueError(
            f"--topography=qv-zero requires {QV_TOPO_VARIABLE!r} with shape "
            f"{expected_qv_shape}"
        )
    metadata = variables[QV_TOPO_VARIABLE]
    if parse_shape(metadata) != expected_qv_shape:
        raise ValueError(
            f"--topography=qv-zero requires {QV_TOPO_VARIABLE!r} with shape "
            f"{expected_qv_shape}; got {parse_shape(metadata)}"
        )
    available_steps = int(metadata.get("AvailableStepsCount", "0"))
    if available_steps <= QV_TOPO_STEP:
        raise ValueError(
            f"--topography=qv-zero requires a second {QV_TOPO_VARIABLE!r} step; "
            f"only {available_steps} step(s) are available"
        )

    qv = np.asarray(
        read_step(reader, QV_TOPO_VARIABLE, QV_TOPO_STEP), dtype=np.float64
    )
    if qv.shape != expected_qv_shape or not np.all(np.isfinite(qv)):
        raise ValueError(
            f"The second {QV_TOPO_VARIABLE!r} step must be finite with shape "
            f"{expected_qv_shape}; got {qv.shape}"
        )

    is_zero = qv == 0.0
    leading_zeros = np.logical_and.accumulate(is_zero, axis=0)
    if np.any(is_zero & ~leading_zeros):
        raise ValueError(
            f"The second {QV_TOPO_VARIABLE!r} step contains zero values above "
            "nonzero values; terrain zeros must form a contiguous prefix in each column"
        )
    if np.any(np.all(is_zero, axis=0)):
        raise ValueError(
            f"The second {QV_TOPO_VARIABLE!r} step contains an all-zero column, "
            "so its surface position cannot be determined"
        )

    blocked_levels = np.count_nonzero(leading_zeros, axis=0)
    interfaces, interface_source = vertical_interfaces(reader, variables, z_mid, args)
    height = interfaces[blocked_levels]
    details = {
        "qv_topo_variable": QV_TOPO_VARIABLE,
        "qv_topo_step_index": QV_TOPO_STEP,
        "qv_topo_rule": "leading qv == 0 levels; surface at upper interface",
        "z_interface_source": interface_source,
    }
    return height, details


def select_surface_height(
    reader: adios2.FileReader,
    variables: dict[str, dict[str, str]],
    attributes: dict[str, dict[str, str]],
    expected_shape: tuple[int, int],
    z_mid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    mode = args.topography
    height_exists = (
        args.height_variable in variables
        and parse_shape(variables[args.height_variable]) == expected_shape
    )
    topo_exists = (
        args.topo_variable in variables
        and parse_shape(variables[args.topo_variable]) == expected_shape
    )

    selected_variable: str | None = None
    if mode == "auto":
        if height_exists:
            mode = "height"
        elif topo_exists:
            units = read_string_attribute(
                reader, attributes, f"{args.topo_variable}/units"
            )
            normalized_units = (units or "").strip().lower()
            if normalized_units in {"index", "level", "levels"}:
                mode = "index"
            elif normalized_units in {"m", "meter", "meters", "metre", "metres"}:
                mode = "topo-height"
            else:
                raise ValueError(
                    f"Cannot determine whether {args.topo_variable!r} is index or height "
                    f"from units={units!r}; use --topography index or height"
                )
        else:
            mode = "flat"

    details: dict[str, Any] = {}
    if mode == "flat":
        height = np.full(expected_shape, args.flat_height, dtype=np.float64)
        details["flat_surface_height_m"] = float(args.flat_height)
        return height, "flat", details

    if mode == "qv-zero":
        height, details = surface_height_from_qv_zeros(
            reader, variables, expected_shape, z_mid, args
        )
        return height, "qv-zero", details

    if mode == "height":
        selected_variable = args.height_variable
        if not height_exists:
            raise ValueError(
                f"--topography=height requires {selected_variable!r} with shape {expected_shape}"
            )
        height = np.asarray(read_first_step(reader, selected_variable), dtype=np.float64)
        details["height_variable"] = selected_variable
        return height, "height", details

    if mode == "topo-height":
        selected_variable = args.topo_variable
        height = np.asarray(read_first_step(reader, selected_variable), dtype=np.float64)
        details["height_variable"] = selected_variable
        return height, "height", details

    if mode != "index":
        raise ValueError(f"Unsupported topography mode: {mode}")
    if not topo_exists:
        raise ValueError(
            f"--topography=index requires {args.topo_variable!r} with shape {expected_shape}"
        )

    topo = np.asarray(read_first_step(reader, args.topo_variable), dtype=np.float64)
    interfaces, interface_source = vertical_interfaces(reader, variables, z_mid, args)

    height = physical_height_from_index(topo, interfaces, args.topo_index_offset)
    details.update(
        {
            "topo_variable": args.topo_variable,
            "topo_index_offset": int(args.topo_index_offset),
            "z_interface_source": interface_source,
        }
    )
    return height, "index", details


def make_connectivity(nx: int, ny: int) -> np.ndarray:
    point_ids = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
    return np.stack(
        (
            point_ids[:-1, :-1],
            point_ids[:-1, 1:],
            point_ids[1:, 1:],
            point_ids[1:, :-1],
        ),
        axis=-1,
    ).reshape(-1)


def write_mesh(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    terrain_height: np.ndarray,
    source_path: Path,
    origin_mode: str,
    x_offset: float,
    y_offset: float,
    surface_mode: str,
    surface_details: dict[str, Any],
) -> None:
    xx, yy = np.meshgrid(x, y)
    arrays = {
        "x_centered": np.ascontiguousarray(x, dtype=np.float64),
        "y_centered": np.ascontiguousarray(y, dtype=np.float64),
        "terrain_x": np.ascontiguousarray(xx.ravel(), dtype=np.float64),
        "terrain_y": np.ascontiguousarray(yy.ravel(), dtype=np.float64),
        "terrain_height_m": np.ascontiguousarray(terrain_height.ravel(), dtype=np.float64),
        "terrain_connectivity": np.ascontiguousarray(make_connectivity(x.size, y.size)),
    }
    with adios2.Stream(str(output_path), "w") as stream:
        for name, data in arrays.items():
            shape = list(data.shape)
            stream.write(name, data, shape, [0] * data.ndim, shape)
        for name in (
            "x_centered",
            "y_centered",
            "terrain_x",
            "terrain_y",
            "terrain_height_m",
        ):
            stream.write_attribute("units", "m", name)
        stream.write_attribute(
            "long_name", "physical surface height above the declared vertical datum", "terrain_height_m"
        )
        stream.write_attribute("source_data", str(source_path))
        stream.write_attribute("horizontal_origin_mode", origin_mode)
        stream.write_attribute("source_x_offset_m", float(x_offset))
        stream.write_attribute("source_y_offset_m", float(y_offset))
        stream.write_attribute("surface_mode", surface_mode)
        for key, value in surface_details.items():
            stream.write_attribute(key, value)


def field_spec(name: str, data_source: str, static: bool = False) -> dict[str, Any]:
    array: dict[str, Any] = {
        "array_type": "basic",
        "data_source": data_source,
        "variable": name,
        # ParaView 6.2/Fides otherwise interprets the final dimension of a
        # rank-2 scalar as vector components and reads only ny tuples.
        "is_vector": "false",
    }
    if static:
        array["static"] = True
    return {"name": name, "association": "points", "array": array}


def data_source_spec(name: str, filename: str) -> dict[str, str]:
    return {
        "name": name,
        "filename_mode": "relative",
        "filename": filename,
    }


def add_time_information(model: dict[str, Any], time_variable: str | None) -> None:
    if time_variable is not None:
        model["step_information"] = {
            "data_source": "source",
            "variable": time_variable,
        }


def build_3d_json(
    source_filename: str,
    mesh_filename: str,
    fields: list[str],
    dimension_variable: str,
    z_variable: str,
    time_variable: str | None,
) -> dict[str, Any]:
    model: dict[str, Any] = {
        "data_sources": [
            data_source_spec("source", source_filename),
            data_source_spec("mesh", mesh_filename),
        ],
        "coordinate_system": {
            "array": {
                "array_type": "cartesian_product",
                "x_array": {
                    "array_type": "basic",
                    "data_source": "mesh",
                    "variable": "x_centered",
                    "static": True,
                },
                "y_array": {
                    "array_type": "basic",
                    "data_source": "mesh",
                    "variable": "y_centered",
                    "static": True,
                },
                "z_array": {
                    "array_type": "basic",
                    "data_source": "source",
                    "variable": z_variable,
                    "static": True,
                },
            }
        },
        "cell_set": {
            "cell_set_type": "structured",
            "dimensions": {
                "source": "variable_dimensions",
                "data_source": "source",
                "variable": dimension_variable,
            },
        },
        "fields": [field_spec(name, "source") for name in fields],
    }
    add_time_information(model, time_variable)
    return {"vvm_rectilinear": model}


def build_2d_json(
    source_filename: str,
    mesh_filename: str,
    fields: list[str],
    time_variable: str | None,
) -> dict[str, Any]:
    mesh_fields = ["terrain_height_m"]
    # Flat surfaces use the same composite schema: terrain_height_m simply
    # contains the same z coordinate for every point.
    model: dict[str, Any] = {
        "data_sources": [
            data_source_spec("source", source_filename),
            data_source_spec("mesh", mesh_filename),
        ],
        "coordinate_system": {
            "array": {
                "array_type": "composite",
                "x_array": {
                    "array_type": "basic",
                    "data_source": "mesh",
                    "variable": "terrain_x",
                    "static": True,
                },
                "y_array": {
                    "array_type": "basic",
                    "data_source": "mesh",
                    "variable": "terrain_y",
                    "static": True,
                },
                "z_array": {
                    "array_type": "basic",
                    "data_source": "mesh",
                    "variable": "terrain_height_m",
                    "static": True,
                },
            }
        },
        "cell_set": {
            "cell_set_type": "single_type",
            "cell_type": "quad",
            "data_source": "mesh",
            "variable": "terrain_connectivity",
            "static": True,
        },
        "fields": [field_spec(name, "source") for name in fields]
        + [field_spec(name, "mesh", static=True) for name in mesh_fields],
    }
    add_time_information(model, time_variable)
    return {"vvm_surface": model}


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def remove_existing(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def resolve_output(output_dir: Path, name: str, option: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or name in {"", ".", ".."}:
        raise ValueError(f"{option} must be a filename, not a path: {name!r}")
    return output_dir / relative


def resolve_input(input_path: Path, bp_name: str | None) -> tuple[Path, Path]:
    """Resolve either a BP path or a run directory containing a BP path."""
    resolved = input_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    # ADIOS BP5 data are normally directories whose names end in .bp.
    if not resolved.is_dir() or resolved.suffix == ".bp":
        return resolved, resolved.parent

    run_directory = resolved
    if bp_name is not None:
        candidate_name = Path(bp_name)
        if candidate_name.is_absolute() or len(candidate_name.parts) != 1:
            raise ValueError(f"--bp-name must be a filename, not a path: {bp_name!r}")
        candidate = run_directory / candidate_name
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate.resolve(), run_directory

    preferred = run_directory / "vvm_output.bp"
    if preferred.exists():
        return preferred.resolve(), run_directory

    candidates = sorted(run_directory.glob("*.bp"))
    if not candidates:
        raise FileNotFoundError(f"No *.bp data set found in {run_directory}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple BP data sets found in {run_directory}: {names}; use --bp-name"
        )
    return candidates[0].resolve(), run_directory


def choose_time_variable(
    requested: str | None, variables: dict[str, dict[str, str]]
) -> str | None:
    if requested is not None:
        if requested.lower() == "none":
            return None
        if requested not in variables:
            raise ValueError(f"Time variable {requested!r} does not exist")
        return requested
    for candidate in ("time", "model_time_s"):
        if candidate in variables:
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = ExampleArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s exp/vvm_output.bp
  %(prog)s exp/vvm_output.bp --topography qv-zero --force
""",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="VVM BP path, or a run directory containing vvm_output.bp",
    )
    parser.add_argument(
        "--bp-name",
        help="BP filename inside a run directory (default: vvm_output.bp or the only *.bp)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: directory containing the input BP)",
    )
    parser.add_argument("--mesh-name", default="vvm_mesh.bp")
    parser.add_argument("--json-2d-name", default="vvm_2d.json")
    parser.add_argument("--json-3d-name", default="vvm_3d.json")
    parser.add_argument("--x-variable", default=DEFAULT_X)
    parser.add_argument("--y-variable", default=DEFAULT_Y)
    parser.add_argument("--z-variable", default=DEFAULT_Z)
    parser.add_argument(
        "--horizontal-origin",
        choices=("center", "native"),
        default="center",
        help="subtract the middle-index x/y values, or preserve native coordinates",
    )
    parser.add_argument(
        "--topography",
        choices=("auto", "index", "height", "qv-zero", "flat"),
        default="auto",
        help=(
            "surface-height mode; qv-zero uses the leading qv == 0 levels in the "
            "second time step (auto prefers an existing physical-height variable)"
        ),
    )
    parser.add_argument("--topo-variable", default="topo")
    parser.add_argument("--height-variable", default="terrain_height_m")
    parser.add_argument(
        "--interface-variable",
        help="full nz+1 vertical-interface variable used for topo index lookup",
    )
    parser.add_argument(
        "--topo-index-offset",
        type=int,
        default=-1,
        help="interface index = round(topo) + offset (default: -1)",
    )
    parser.add_argument(
        "--lower-interface",
        type=float,
        default=0.0,
        help="lower interface in metres when interfaces must be reconstructed",
    )
    parser.add_argument(
        "--flat-height", type=float, default=0.0, help="surface height for flat mode"
    )
    parser.add_argument(
        "--time-variable",
        help="step-information variable; auto-detects time/model_time_s, or use 'none'",
    )
    parser.add_argument(
        "--dimension-variable",
        help="representative 3D variable used for structured dimensions",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing generated outputs"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path, default_output_dir = resolve_input(args.input, args.bp_name)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = resolve_output(output_dir, args.mesh_name, "--mesh-name")
    json_2d_path = resolve_output(output_dir, args.json_2d_name, "--json-2d-name")
    json_3d_path = resolve_output(output_dir, args.json_3d_name, "--json-3d-name")
    outputs = (mesh_path, json_2d_path, json_3d_path)
    if len(set(outputs)) != len(outputs):
        raise ValueError("Generated output filenames must be distinct")
    if data_path in outputs:
        raise ValueError("A generated output must not overwrite the input BP")

    reader = adios2.FileReader(str(data_path))
    try:
        variables = reader.available_variables()
        attributes = reader.available_attributes()
        for coordinate in (args.x_variable, args.y_variable, args.z_variable):
            if coordinate not in variables:
                raise ValueError(f"Required coordinate variable {coordinate!r} does not exist")

        x_raw = validate_axis(read_first_step(reader, args.x_variable), args.x_variable)
        y_raw = validate_axis(read_first_step(reader, args.y_variable), args.y_variable)
        z_mid = validate_axis(read_first_step(reader, args.z_variable), args.z_variable)
        nx, ny, nz = x_raw.size, y_raw.size, z_mid.size
        shape_2d = (ny, nx)
        shape_3d = (nz, ny, nx)

        fields_2d = sorted(
            name
            for name, metadata in variables.items()
            if parse_shape(metadata) == shape_2d
            and is_numeric(metadata)
            and name not in {args.height_variable, "tiff_valid"}
        )
        fields_3d = sorted(
            name
            for name, metadata in variables.items()
            if parse_shape(metadata) == shape_3d and is_numeric(metadata)
        )
        if not fields_3d:
            raise ValueError(f"No numeric 3D variables with shape {shape_3d} were found")

        dimension_variable = args.dimension_variable or (
            "qc" if "qc" in fields_3d else fields_3d[0]
        )
        if dimension_variable not in fields_3d:
            raise ValueError(
                f"Dimension variable {dimension_variable!r} is not a numeric {shape_3d} field"
            )
        time_variable = choose_time_variable(args.time_variable, variables)
        terrain_height, surface_mode, surface_details = select_surface_height(
            reader,
            variables,
            attributes,
            shape_2d,
            z_mid,
            args,
        )
    finally:
        reader.close()

    if terrain_height.shape != shape_2d or not np.all(np.isfinite(terrain_height)):
        raise ValueError(
            f"Surface height must be finite with shape {shape_2d}; got {terrain_height.shape}"
        )

    x, x_offset = visualization_axis(x_raw, args.horizontal_origin)
    y, y_offset = visualization_axis(y_raw, args.horizontal_origin)
    source_filename = Path(os.path.relpath(data_path, output_dir)).as_posix()
    mesh_filename = Path(os.path.relpath(mesh_path, output_dir)).as_posix()
    document_2d = build_2d_json(
        source_filename, mesh_filename, fields_2d, time_variable
    )
    document_3d = build_3d_json(
        source_filename,
        mesh_filename,
        fields_3d,
        dimension_variable,
        args.z_variable,
        time_variable,
    )

    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if existing and not args.force:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Generated output already exists; use --force to replace:\n  {joined}")
    for path in existing:
        remove_existing(path)

    write_mesh(
        mesh_path,
        x,
        y,
        terrain_height,
        data_path,
        args.horizontal_origin,
        x_offset,
        y_offset,
        surface_mode,
        surface_details,
    )
    write_json(json_2d_path, document_2d)
    write_json(json_3d_path, document_3d)

    print("Generated ParaView assets")
    print(f"  input:  {data_path}")
    print(f"  grid:   nx={nx}, ny={ny}, nz={nz}")
    print(f"  x:      {x[0]:g} .. {x[-1]:g} m ({args.horizontal_origin})")
    print(f"  y:      {y[0]:g} .. {y[-1]:g} m ({args.horizontal_origin})")
    print(f"  z_mid:  {z_mid[0]:g} .. {z_mid[-1]:g} m")
    print(
        f"  surface:{surface_mode}, {terrain_height.min():g} .. "
        f"{terrain_height.max():g} m"
    )
    print(f"  fields: {len(fields_2d)} two-dimensional, {len(fields_3d)} three-dimensional")
    print(f"  mesh:   {mesh_path}")
    print(f"  2D JSON:{json_2d_path}")
    print(f"  3D JSON:{json_3d_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
