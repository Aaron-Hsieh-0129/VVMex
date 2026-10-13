"""
Generate CLEO's input binaries from the VVMex JSON you are going to run.

  physics.microphysics.cleo.init_gbx_path     gridbox boundaries
  physics.microphysics.cleo.init_supers_path  initial superdroplets
  cleo_config.yaml                            beside them; cleopy reads nspacedims from it

The gridboxes are the model's physical cells: z_up[h-1 : nz+h] rebuilt as
Initializer.cpp does (including the dz1 stretching), and the dx/dy columns.
Only Cartesian grids are supported.

Usage:
    .venv/bin/python tools/cleo/gen_cleo_inputs.py -c <config.json>
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CLEO = "physics.microphysics.cleo"
DEFAULT_RCEMIP = "./rundata/initial_conditions/profiles/snd_rcemip_anal300_v3.txt"
_REQUIRED = object()


class NumberSampledRadiiGen:
    """Radii drawn from the number distribution, for a common multiplicity
    (Shima et al. 2009). Radii outside rspan are redrawn."""

    def __init__(self, geomeans, geosigs, scalefacs, rspan, seed):
        self.geomeans = np.asarray(geomeans, dtype=float)
        self.geosigs = np.asarray(geosigs, dtype=float)
        w = np.asarray(scalefacs, dtype=float)
        self.weights = w / w.sum()
        self.rspan = rspan
        self.rng = np.random.default_rng(seed)

    def __call__(self, nsupers):
        out = np.empty(int(nsupers))
        n = 0
        while n < out.size:
            need = out.size - n
            m = self.rng.choice(len(self.weights), size=need, p=self.weights)
            r = self.rng.lognormal(np.log(self.geomeans[m]), np.log(self.geosigs[m]))
            r = r[(r >= self.rspan[0]) & (r <= self.rspan[1])]
            take = min(r.size, need)
            out[n:n + take] = r[:take]
            n += take
        return out


class UniformProbDist:
    def __call__(self, radii, totxi):
        radii = np.asarray(radii)
        if radii.size == 0:
            return np.empty(0)
        return np.full(radii.shape, 1.0 / radii.size)


def die(msg):
    sys.exit(f"gen_cleo_inputs: {msg}")


def get(cfg: dict, path: str, default=_REQUIRED):
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            if default is _REQUIRED:
                die(f"{path} is missing from the configuration")
            return default
        node = node[k]
    return node


def resolve(p) -> Path:
    """Relative paths are taken from the repository root."""
    q = Path(str(p))
    return q if q.is_absolute() else (REPO / q).resolve()


def read_grid(cfg: dict) -> dict:
    g = get(cfg, "grid")
    if "horizontal" in g:
        if get(cfg, "grid.horizontal.geometry.kind") != "cartesian":
            die("only grid.horizontal.geometry.kind = 'cartesian' is supported")
        out = {"nx": int(get(cfg, "grid.horizontal.nx")),
               "ny": int(get(cfg, "grid.horizontal.ny")),
               "h": int(get(cfg, "grid.horizontal.n_halo_cells")),
               "dx": float(get(cfg, "grid.horizontal.geometry.dx")),
               "dy": float(get(cfg, "grid.horizontal.geometry.dy"))}
    else:
        out = {"nx": int(get(cfg, "grid.nx")), "ny": int(get(cfg, "grid.ny")),
               "h": int(get(cfg, "grid.n_halo_cells")),
               "dx": float(get(cfg, "grid.dx")), "dy": float(get(cfg, "grid.dy"))}
    if "vertical" in g:
        out.update(nz=int(get(cfg, "grid.vertical.nz")),
                   dz=float(get(cfg, "grid.vertical.dz")),
                   dz1=float(get(cfg, "grid.vertical.dz1")),
                   vtype=get(cfg, "grid.vertical.type", "default"),
                   rcemip_path=get(cfg, "grid.vertical.rcemip_grid_data_path", ""))
    else:
        out.update(nz=int(get(cfg, "grid.nz")), dz=float(get(cfg, "grid.dz")),
                   dz1=float(get(cfg, "grid.dz1")),
                   vtype=get(cfg, "grid.vertical_coordinate_type", "default"),
                   rcemip_path=get(cfg, "grid.rcemip_grid_data_path", DEFAULT_RCEMIP))
    return out


def parse_initializer_domain(initializer: Path) -> float:
    """DOMAIN (the stretching depth) is a constant in Initializer.cpp, not a config key."""
    m = re.search(r"\bDOMAIN\s*=\s*real\(\s*([0-9eE.+\-]+)\s*\)\s*;", initializer.read_text())
    if not m:
        die(f"could not find 'DOMAIN = real(...)' in {initializer}")
    return float(m.group(1))


def vertical_boundaries(g: dict, domain: float) -> np.ndarray:
    """The nz+1 physical level boundaries in metres, as Initializer.cpp builds them."""
    h, nz, dz, dz1 = g["h"], g["nz"], g["dz"], g["dz1"]
    ntot = nz + 2 * h
    z_up = np.zeros(ntot, dtype=float)

    if g["vtype"] == "default":
        for k in range(h, ntot):
            z_up[k] = z_up[k - 1] + dz
        cz2 = (dz - dz1) / (dz * (domain - dz))
        cz1 = 1.0 - cz2 * domain
        for k in range(h - 1, ntot):
            z_up[k] = z_up[k] * (cz1 + cz2 * z_up[k])
    elif g["vtype"] == "rcemip":
        src = resolve(g["rcemip_path"])
        if not src.is_file():
            die(f"rcemip profile not found: {src}")
        rows = src.read_text().splitlines()[1:]
        if len(rows) < nz:
            die(f"{src} has {len(rows)} data rows, need {nz}")
        for i, k in enumerate(range(h, ntot - h)):
            z_up[k] = float(rows[i].split()[0])
    else:
        die(f"vertical coordinate type '{g['vtype']}' is not supported; "
            "only 'default' and 'rcemip'")

    bounds = z_up[h - 1: nz + h]
    if not np.all(np.diff(bounds) > 0):
        die("vertical boundaries are not strictly increasing")
    return bounds


def model_grid(g: dict, domain: float) -> dict:
    nz, nx, ny = g["nz"], g["nx"], g["ny"]
    if nx == 1 and ny > 1:
        die("nx=1 with ny>1 has no CLEO equivalent (CLEO's 2-D domain is z-x)")
    # Always 3-D: a 2-D CLEO domain has infinite y bounds, so infinite gridbox areas.
    zgrid = vertical_boundaries(g, domain)
    xgrid = [0.0, nx * g["dx"], g["dx"]]
    ygrid = [0.0, ny * g["dy"], g["dy"]] if ny > 1 else np.array([0.0, g["dy"]])
    return {"nspacedims": 3, "zgrid": zgrid, "xgrid": xgrid, "ygrid": ygrid,
            "ngbxs": nz * nx * ny}


def read_settings(cfg: dict) -> dict:
    sd, ae = f"{CLEO}.superdroplets", "physics.microphysics.aerosol"
    zlim = get(cfg, f"{sd}.seed_below_z_m")
    s = {
        "gbx_path": resolve(get(cfg, f"{CLEO}.init_gbx_path")),
        "supers_path": resolve(get(cfg, f"{CLEO}.init_supers_path")),
        "npergbx": int(get(cfg, f"{sd}.per_gridbox")),
        # null seeds every gridbox; the test is on the gridbox upper bound
        "zlim": float("inf") if zlim is None else float(zlim),
        "rmin": float(get(cfg, f"{sd}.radius_min_m")),
        "rmax": float(get(cfg, f"{sd}.radius_max_m")),
        "numconc_tolerance": float(get(cfg, f"{sd}.numconc_tolerance")),
        "common_xi": bool(get(cfg, f"{sd}.common_multiplicity")),
        "wet_scale": float(get(cfg, f"{sd}.wet_scale")),
        "seed": int(get(cfg, f"{sd}.seed")),
        "geomeans": [float(v) for v in get(cfg, f"{ae}.geometric_mean_radius_m")],
        "geosigs": [float(v) for v in get(cfg, f"{ae}.geometric_stddev")],
        "scalefacs": [float(v) for v in get(cfg, f"{ae}.number_conc_per_m3")],
    }
    n = len(s["geomeans"])
    if n == 0 or not (len(s["geosigs"]) == len(s["scalefacs"]) == n):
        die(f"{ae}: the three mode arrays must be non-empty and the same length")
    return s


def write_grid(paths: dict, grid: dict, figures, figpath: Path) -> None:
    from cleopy import geninitconds
    geninitconds.generate_gridbox_boundaries(
        str(paths["grid"]), grid["zgrid"], grid["xgrid"], grid["ygrid"],
        str(paths["constants"]), isprintinfo=True, isfigures=figures,
        savefigpath=str(figpath))


def compute_nsupers(paths: dict, s: dict) -> dict:
    from cleopy.initsuperdropsbinary_src import crdgens
    nsupers = crdgens.nsupers_at_domain_base(
        str(paths["grid"]), str(paths["constants"]), s["npergbx"], s["zlim"])
    seeded = sum(v > 0 for v in nsupers.values())
    print(f"CLEO  : {sum(nsupers.values())} superdroplets in {seeded} of {len(nsupers)} gridboxes")
    return nsupers


def write_initsupers(paths: dict, s: dict, nsupers: dict, figures, figpath: Path) -> None:
    from cleopy import geninitconds
    from cleopy.initsuperdropsbinary_src import (attrsgen, crdgens, dryrgens,
                                                 probdists, rgens)

    # dryradius = radius / wet_scale; scale the spectrum too so the dry population is unchanged
    sf = s["wet_scale"]
    means = [g * sf for g in s["geomeans"]]
    span = [s["rmin"] * sf, s["rmax"] * sf]
    if s["common_xi"]:
        radiigen = NumberSampledRadiiGen(means, s["geosigs"], s["scalefacs"], span, s["seed"])
        xiprobdist = UniformProbDist()
    else:
        radiigen = rgens.SampleLog10RadiiGen(span)
        xiprobdist = probdists.LnNormal(means, s["geosigs"], s["scalefacs"])
    initattrsgen = attrsgen.AttrsGenerator(
        radiigen, dryrgens.ScaledRadiiGen(sf), xiprobdist,
        crdgens.SampleCoordGen(True), crdgens.SampleCoordGen(True), crdgens.SampleCoordGen(True))

    geninitconds.generate_initial_superdroplet_conditions(
        initattrsgen, str(paths["initsupers"]), str(paths["config"]),
        str(paths["constants"]), str(paths["grid"]), nsupers,
        np.sum(s["scalefacs"]), numconc_tolerance=s["numconc_tolerance"],
        isprintinfo=True, isfigures=figures, savefigpath=str(figpath), gbxs2plt=[0])


def report_domain_totals(paths: dict) -> None:
    """cleopy's own total sums xi as uint64 and overflows; print it in float64."""
    from cleopy.gbxboundariesbinary_src import read_gbxboundaries as rg
    from cleopy.initsuperdropsbinary_src import read_initsuperdrops as rs

    vols = np.asarray(rg.get_gbxvols_from_gridfile(
        str(paths["grid"]), constants_filename=str(paths["constants"]), isprint=False))
    attrs = rs.get_superdroplet_attributes(
        str(paths["config"]), str(paths["constants"]), str(paths["initsupers"]))
    xi = np.asarray(attrs.xi).flatten().astype(np.float64)
    print(f"\ndomain-mean droplet number concentration: {xi.sum() / vols.sum() / 1e6:.4g} /cm^3")


def cleo_from_cmake_cache():
    cache = REPO / "build" / "CMakeCache.txt"
    if not cache.is_file():
        return None
    m = re.search(r"^CLEO_SOURCE_DIR:\w+=(.+)$", cache.read_text(), re.M)
    return Path(m.group(1).strip()) if m else None


def find_cleo(explicit) -> Path:
    """The CLEO the build compiles against, so cleoconstants.hpp agrees with the model."""
    candidates = [explicit] if explicit is not None else [
        c for c in (cleo_from_cmake_cache(), REPO / "build" / "_deps" / "cleo-src",
                    REPO.parent / "CLEO") if c is not None]
    for c in candidates:
        if (c / "cleopy").is_dir() and (c / "libs" / "cleoconstants.hpp").is_file():
            return c.resolve()
    die("no CLEO checkout found in " + ", ".join(str(c) for c in candidates)
        + "; pass --path2cleo")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, required=True, help="the VVMex JSON")
    p.add_argument("--path2cleo", type=Path, default=None,
                   help="CLEO source root (default: CLEO_SOURCE_DIR from build/CMakeCache.txt)")
    p.add_argument("--save-figures", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        die(f"cannot read {config_path}: {e}")

    path2cleo = find_cleo(args.path2cleo)
    sys.path.insert(0, str(path2cleo))

    g = read_grid(cfg)
    grid = model_grid(g, parse_initializer_domain(REPO / "src" / "core" / "Initializer.cpp"))
    s = read_settings(cfg)

    print(f"config: {config_path}")
    print(f"model : nx={g['nx']} ny={g['ny']} nz={g['nz']} dx={g['dx']} dy={g['dy']} "
          f"dz={g['dz']} dz1={g['dz1']} vcoord={g['vtype']}")
    print(f"CLEO  : ngbxs={grid['ngbxs']}, z 0..{grid['zgrid'][-1]} m  (from {path2cleo})")

    share = s["gbx_path"].parent
    figpath = share / "figs"
    for d in (share, s["supers_path"].parent, figpath):
        d.mkdir(parents=True, exist_ok=True)

    paths = {"constants": path2cleo / "libs" / "cleoconstants.hpp",
             "config": share / "cleo_config.yaml",
             "grid": s["gbx_path"], "initsupers": s["supers_path"]}
    figures = [False, args.save_figures]
    for f in (paths["grid"], paths["initsupers"]):
        f.unlink(missing_ok=True)

    paths["config"].write_text(
        f"# generated by tools/cleo/gen_cleo_inputs.py for cleopy\n"
        f"domain:\n  nspacedims: {grid['nspacedims']}\n")
    write_grid(paths, grid, figures, figpath)
    nsupers = compute_nsupers(paths, s)
    if sum(nsupers.values()) == 0:
        die(f"no gridbox lies below {CLEO}.superdroplets.seed_below_z_m = {s['zlim']} m")

    # cleopy's radius and coordinate samplers use the global numpy RNG
    np.random.seed(s["seed"])
    write_initsupers(paths, s, nsupers, figures, figpath)
    report_domain_totals(paths)

    print("\nwrote:")
    for k in ("grid", "initsupers", "config"):
        print(f"  {paths[k]}")


if __name__ == "__main__":
    main()
