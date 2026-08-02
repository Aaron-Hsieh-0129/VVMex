#!/usr/bin/env python3
"""Bit-for-bit regression check against a stored per-variable digest.

Full baseline files do not scale past the 32^2 cases (rcemip output alone is
~1 GB), so the reference is a small JSON of SHA-256 digests over each variable's
raw bytes plus a few statistics for diagnostics. Any bit-level change fails.

    --update   (re)write the reference from this output
"""
import argparse, hashlib, json, os, sys

import h5py
import numpy as np


def digests(path):
    out = {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind == 'f' and obj.size > 1:
                arr = np.ascontiguousarray(obj[...])
                out[name] = {
                    'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
                    'shape': list(arr.shape),
                    'min': float(np.nanmin(arr)),
                    'max': float(np.nanmax(arr)),
                    'mean': float(np.nanmean(arr)),
                }
        f.visititems(visit)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--update', action='store_true')
    a = p.parse_args()

    if not os.path.isfile(a.output):
        print(f"FAIL: model produced no output at {a.output}", file=sys.stderr)
        return 1

    got = digests(a.output)
    if not got:
        print(f"FAIL: no comparable datasets in {a.output}", file=sys.stderr)
        return 1

    if a.update:
        os.makedirs(os.path.dirname(os.path.abspath(a.reference)), exist_ok=True)
        json.dump(got, open(a.reference, 'w'), indent=1, sort_keys=True)
        print(f"[check_output] wrote reference for {len(got)} variables -> {a.reference}")
        return 0

    if not os.path.isfile(a.reference):
        print(f"FAIL: missing reference {a.reference}. Regenerate with --update.", file=sys.stderr)
        return 1
    ref = json.load(open(a.reference))

    bad, missing = [], []
    for name, r in sorted(ref.items()):
        g = got.get(name)
        if g is None:
            missing.append(name)
        elif g['sha256'] != r['sha256']:
            bad.append((name, r, g))

    extra = sorted(set(got) - set(ref))
    if bad or missing:
        print(f"FAIL: {len(bad)} variable(s) differ, {len(missing)} missing", file=sys.stderr)
        for name, r, g in bad[:10]:
            print(f"  {name}: shape={r['shape']}\n"
                  f"      reference min/max/mean = {r['min']:.9g} / {r['max']:.9g} / {r['mean']:.9g}\n"
                  f"      actual    min/max/mean = {g['min']:.9g} / {g['max']:.9g} / {g['mean']:.9g}",
                  file=sys.stderr)
        for name in missing[:10]:
            print(f"  missing from output: {name}", file=sys.stderr)
        return 1

    note = f", {len(extra)} new variable(s) not in reference" if extra else ""
    print(f"PASS: {len(ref)} variables bit-for-bit{note}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
