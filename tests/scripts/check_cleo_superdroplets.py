#!/usr/bin/env python3
"""Follow every superdroplet through CLEO's zarr after test_cleo_coupling advection.

The advection scenario blows a uniform wind (u, v) over the domain, w = 0,
condensation and collision off, so where each superdroplet must be is known
exactly. At every observation this checks that:

  1. the ragged arrays hold totnsupers superdroplets, each sdId once, in
     ascending order -- what CollectiveDataset promises and got wrong before;
  2. no superdroplet appears from nowhere, and one disappears only through the
     bottom of the domain;
  3. every superdroplet sits in the domain and in the gridbox its sdgbxindex
     names on the rank holding it (sdgbxindex is rank-local; decomposition.json,
     written by the test, says where each rank's gridboxes start);
  4. each moved by u dt and v dt, wrapped at the periodic edges, and sank no
     further than sedimentation takes it;
  5. the x columns and y rows hold as many superdroplets as each other: a rank
     edge that drops or misroutes superdroplets leaves a stripe.

With --compare-to it also requires the same superdroplets, with the same
attributes, as another run of the same case -- at another rank count, which is
the rank-invariance check. Collisions are off, so CLEO is deterministic and
anything beyond float32 rounding is a decomposition error.

    check_cleo_superdroplets.py --config build/cleo_tests/advection/config_r4.json \\
        --compare-to build/cleo_tests/advection/config.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Positions are float32 in units of 1 km, so about 1e-4 m is all two
# positions can agree to; 2e-3 m leaves room and is still far below the 40 m
# a superdroplet moves per step.
POSITION_TOL_M = 2e-3
# A haze droplet (wet radius < 1 um) falls at < 1e-4 m s-1 relative to the air.
SEDIMENTATION_M_S = 2e-4
# Leaving through the bottom needs a superdroplet this close to the ground the
# observation before.
BOTTOM_EXIT_M = 0.05
# 1-D marginals hold over a thousand superdroplets each; sampling noise is a
# few per cent.
STRIPE_TOL = 0.2

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        print(f"FAIL: {message}")
    return condition


class Case:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        c = json.loads(self.config_path.read_text())
        g = c["grid"]
        if g["dz"] != g["dz1"]:
            raise SystemExit("check_cleo_superdroplets.py assumes a uniform vertical grid")
        self.nx, self.ny, self.nz = g["nx"], g["ny"], g["nz"]
        self.dx, self.dy, self.dz = g["dx"], g["dy"], g["dz"]
        self.lx, self.ly, self.lz = self.nx * self.dx, self.ny * self.dy, self.nz * self.dz
        t = c["test_cleo"]
        self.u, self.v = t["u_m_s"], t["v_m_s"]
        output_dir = Path(c["output"]["output_dir"])
        self.zarr_path = output_dir / c["physics"]["microphysics"]["cleo"]["output"]["zarrbasedir"]
        decomposition = output_dir / "decomposition.json"
        if not decomposition.is_file():
            raise SystemExit(f"{decomposition} missing: did test_cleo_coupling advection run?")
        self.ranks = json.loads(decomposition.read_text())["ranks"]
        self._read_zarr()

    def _read_zarr(self):
        import zarr
        if not self.zarr_path.is_dir():
            raise SystemExit(f"no CLEO zarr at {self.zarr_path}")
        group = zarr.open_group(str(self.zarr_path), mode="r")

        def var(name, dtype=np.float64):
            a = group[name]
            data = np.asarray(a[...])
            if "scale_factor" in a.attrs:
                return data.astype(np.float64) * a.attrs["scale_factor"]
            return data.astype(dtype)

        self.time = var("time")
        self.count = var("raggedcount", np.int64)
        self.totnsupers = var("totnsupers", np.int64)
        self.nsupers = var("nsupers", np.int64) if "nsupers" in group else None
        # Raw float32 kept alongside, for the bitwise comparison between runs.
        self.raw = {name: np.asarray(group[name][...])
                    for name in ("sdId", "sdgbxindex", "coord1", "coord2", "coord3",
                                 "radius", "xi", "msol")}
        self.sdid = self.raw["sdId"].astype(np.int64)
        self.sdgbx = self.raw["sdgbxindex"].astype(np.int64)
        self.x = var("coord1")
        self.y = var("coord2")
        self.z = var("coord3")
        self.offsets = np.concatenate([[0], np.cumsum(self.count)])
        if self.offsets[-1] != self.sdid.size:
            raise SystemExit(f"raggedcount sums to {self.offsets[-1]} but the superdroplet arrays "
                             f"hold {self.sdid.size}")

    def at(self, n):
        return slice(self.offsets[n], self.offsets[n + 1])

    def local_index(self, x, y, z):
        """The rank-local gridbox index of each position, -1 if no rank holds it."""
        gi = np.floor(x / self.dx).astype(np.int64) % self.nx
        gj = np.floor(y / self.dy).astype(np.int64) % self.ny
        gk = np.clip(np.floor(z / self.dz).astype(np.int64), 0, self.nz - 1)
        out = np.full(x.shape, -1, dtype=np.int64)
        for r in self.ranks:
            mine = ((gi >= r["x0"]) & (gi < r["x0"] + r["nx"]) &
                    (gj >= r["y0"]) & (gj < r["y0"] + r["ny"]))
            out[mine] = gk[mine] + self.nz * ((gi[mine] - r["x0"]) + r["nx"] * (gj[mine] - r["y0"]))
        return out


def wrap(d, period):
    return (d + 0.5 * period) % period - 0.5 * period


def check_observation(c, n):
    s = c.at(n)
    t = c.time[n]
    count = int(c.count[n])
    check(count == c.totnsupers[n],
          f"t={t:g}: {count} superdroplets in the ragged arrays, totnsupers says {c.totnsupers[n]}")
    if c.nsupers is not None:
        check(count == c.nsupers[n].sum(),
              f"t={t:g}: {count} superdroplets in the ragged arrays, nsupers sums to "
              f"{c.nsupers[n].sum()}")
    ids = c.sdid[s]
    check(np.all(np.diff(ids) > 0),
          f"t={t:g}: sdId not strictly ascending ({np.sum(np.diff(ids) <= 0)} out-of-order "
          f"or repeated rows)")

    x, y, z = c.x[s], c.y[s], c.z[s]
    eps = POSITION_TOL_M
    inside = ((x >= -eps) & (x < c.lx + eps) & (y >= -eps) & (y < c.ly + eps) &
              (z >= -eps) & (z <= c.lz + eps))
    check(inside.all(), f"t={t:g}: {np.sum(~inside)} superdroplets outside the domain")

    # A position within float32 rounding of a face may land on either side.
    sdgbx = c.sdgbx[s]
    matched = np.zeros(ids.shape, dtype=bool)
    for ox in (-eps, 0.0, eps):
        for oy in (-eps, 0.0, eps):
            for oz in (-eps, 0.0, eps):
                matched |= c.local_index(x + ox, y + oy, z + oz) == sdgbx
    if not check(matched.all(),
                 f"t={t:g}: {np.sum(~matched)} superdroplets are not in the gridbox their "
                 f"sdgbxindex names"):
        m = np.argmax(~matched)
        print(f"      first: sdId {ids[m]} at ({x[m]:.3f}, {y[m]:.3f}, {z[m]:.3f}) m, "
              f"sdgbxindex {sdgbx[m]}, position says {c.local_index(x[m:m+1], y[m:m+1], z[m:m+1])[0]}")


def check_motion(c, n):
    """Observation n-1 -> n."""
    a, b = c.at(n - 1), c.at(n)
    t0, t1 = c.time[n - 1], c.time[n]
    dt = t1 - t0
    ids0, ids1 = c.sdid[a], c.sdid[b]
    appeared = np.setdiff1d(ids1, ids0)
    check(appeared.size == 0, f"t={t1:g}: {appeared.size} superdroplets appeared from nowhere "
                              f"(first sdId {appeared[:1]})")
    gone, i_gone, _ = np.intersect1d(ids0, np.setdiff1d(ids0, ids1), return_indices=True)
    if gone.size:
        near_ground = c.z[a][i_gone] < BOTTOM_EXIT_M
        check(near_ground.all(),
              f"t={t1:g}: {np.sum(~near_ground)} superdroplets vanished away from the bottom "
              f"(first sdId {gone[~near_ground][:1]}); they were lost, not precipitated")

    common, i0, i1 = np.intersect1d(ids0, ids1, return_indices=True)
    x0, y0, z0 = c.x[a][i0], c.y[a][i0], c.z[a][i0]
    x1, y1, z1 = c.x[b][i1], c.y[b][i1], c.z[b][i1]
    ex = np.abs(wrap(x1 - x0 - c.u * dt, c.lx))
    ey = np.abs(wrap(y1 - y0 - c.v * dt, c.ly))
    ez = z1 - z0
    for name, err, tol in (("x", ex, POSITION_TOL_M), ("y", ey, POSITION_TOL_M)):
        bad = err > tol
        if not check(not bad.any(), f"t={t1:g}: {bad.sum()} superdroplets did not move "
                                    f"{'u' if name == 'x' else 'v'} dt in {name}"):
            m = np.argmax(err)
            print(f"      worst: sdId {common[m]}, off by {err[m]:.4g} m")
    sink = SEDIMENTATION_M_S * dt + POSITION_TOL_M
    bad = (ez > POSITION_TOL_M) | (ez < -sink)
    if not check(not bad.any(), f"t={t1:g}: {bad.sum()} superdroplets moved vertically "
                                f"by more than sedimentation"):
        m = np.argmax(np.abs(ez))
        print(f"      worst: sdId {common[m]}: z {z0[m]:.4f} -> {z1[m]:.4f} m, "
              f"expected {z0[m]:.4f} m")


def check_stripes(c, n):
    s = c.at(n)
    gi = np.floor(c.x[s] / c.dx).astype(np.int64) % c.nx
    gj = np.floor(c.y[s] / c.dy).astype(np.int64) % c.ny
    for axis, index, size in (("x column", gi, c.nx), ("y row", gj, c.ny)):
        counts = np.bincount(index, minlength=size)
        worst = np.max(np.abs(counts / counts.mean() - 1.0))
        if not check(worst < STRIPE_TOL,
                     f"t={c.time[n]:g}: superdroplets per {axis} vary by {worst:.0%} "
                     f"(limit {STRIPE_TOL:.0%}): a stripe"):
            print(f"      counts per {axis}: {counts.tolist()}")


def compare(c, ref):
    """Same superdroplets with the same attributes, observation by observation."""
    check(c.count.size == ref.count.size,
          f"{c.count.size} observations against {ref.count.size} in the reference")
    for n in range(min(c.count.size, ref.count.size)):
        s, r = c.at(n), ref.at(n)
        t = c.time[n]
        if not check(c.count[n] == ref.count[n],
                     f"t={t:g}: {c.count[n]} superdroplets against {ref.count[n]} in the reference"):
            continue
        if not check(np.array_equal(c.sdid[s], ref.sdid[r]),
                     f"t={t:g}: a different set of sdIds from the reference"):
            continue
        # sdgbxindex is rank-local and so differs with the decomposition.
        for name in ("coord1", "coord2", "coord3", "radius", "xi", "msol"):
            got, want = c.raw[name][s], ref.raw[name][r]
            if got.dtype.kind == "f":
                # Two float32 ulps of slack; the arithmetic is the same per droplet.
                tol = 2 * np.spacing(np.abs(want).astype(want.dtype))
                bad = np.abs(got.astype(np.float64) - want.astype(np.float64)) > tol
            else:
                bad = got != want
            if not check(not bad.any(), f"t={t:g}: {name} differs from the reference for "
                                        f"{bad.sum()} superdroplets"):
                m = np.argmax(bad)
                print(f"      first: sdId {c.sdid[s][m]}: {got[m]!r} vs {want[m]!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="the case's config.json (or config_NAME.json)")
    p.add_argument("--compare-to", help="config of a reference run of the same case")
    a = p.parse_args()

    c = Case(a.config)
    print(f"{c.zarr_path}: {c.count.size} observations, {c.count[0]} superdroplets at "
          f"t={c.time[0]:g} s, {len(c.ranks)} rank(s); u={c.u} v={c.v} m/s")
    if not check(c.count.size >= 2, "fewer than two observations: nothing moved"):
        return 1
    for n in range(c.count.size):
        check_observation(c, n)
        if n > 0:
            check_motion(c, n)
    check_stripes(c, c.count.size - 1)

    crossings = abs(c.u) * (c.time[-1] - c.time[0]) / c.lx
    print(f"each superdroplet crossed the x domain {crossings:.2f} times; "
          f"{c.count[0] - c.count[-1]} fewer superdroplets at the end than at the start")

    if a.compare_to:
        ref = Case(a.compare_to)
        print(f"comparing with {ref.zarr_path} ({len(ref.ranks)} rank(s))")
        compare(c, ref)

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
