#!/usr/bin/env python3
"""Bit-for-bit comparison of two model outputs.

Used for the rank-invariance tests: the same case run on different rank counts
must produce identical results, so no stored reference is needed. This is the
check that would have caught the initialiser bug, where every rank stamped its
own perturbation because the code used local instead of global indices.
"""
import argparse, sys

import h5py
import numpy as np


def datasets(f):
    out = {}
    f.visititems(lambda n, o: out.__setitem__(n, o)
                 if isinstance(o, h5py.Dataset) and o.dtype.kind == 'f' and o.size > 1 else None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--label-a', default='A')
    p.add_argument('--label-b', default='B')
    args = p.parse_args()

    for path in (args.a, args.b):
        try:
            open(path, 'rb').close()
        except OSError:
            print(f"FAIL: missing output {path}", file=sys.stderr)
            return 1

    A, B = h5py.File(args.a, 'r'), h5py.File(args.b, 'r')
    da, db = datasets(A), datasets(B)
    common = sorted(set(da) & set(db))
    if not common:
        print("FAIL: no comparable datasets", file=sys.stderr)
        return 1

    worst, worst_var, ndiff, total = 0.0, "", 0, 0
    for k in common:
        if da[k].shape != db[k].shape:
            print(f"FAIL: shape mismatch for {k}: {da[k].shape} vs {db[k].shape}", file=sys.stderr)
            return 1
        x = np.asarray(da[k][...], np.float64)
        y = np.asarray(db[k][...], np.float64)
        d = int((x != y).sum())
        ndiff += d
        total += x.size
        if d:
            m = float(np.nanmax(np.abs(x - y)))
            if m > worst:
                worst, worst_var = m, k

    if ndiff:
        print(f"FAIL: {args.label_a} vs {args.label_b} differ in {ndiff}/{total} values; "
              f"largest |diff| = {worst:.6e} in {worst_var}", file=sys.stderr)
        return 1
    print(f"PASS: {args.label_a} vs {args.label_b} bit-for-bit over {total} values "
          f"({len(common)} variables)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
