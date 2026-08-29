#!/usr/bin/env python3
"""Regression check against a stored per-variable digest.

Full baseline files do not scale past the 32^2 cases (rcemip output alone is
~1 GB), so the reference is a small JSON of SHA-256 digests over each variable's
values plus a few statistics for diagnostics. Native digests hash the raw bytes
and fail on any bit-level change. A float32 digest first narrows every value to
a canonical little-endian float32; this keeps a whole-field spatial check while
ignoring low FP64 mantissa bits that can vary across GPU architectures.

    --update --digest-dtype {native,float32}
        (re)write the reference from this output
"""
import argparse, hashlib, json, os, sys

import h5py
import numpy as np


VALID_DIGEST_DTYPES = ('native', 'float32')


def digest_values(arr, digest_dtype):
    if digest_dtype == 'native':
        return arr
    if digest_dtype == 'float32':
        # The byte order makes the digest portable beyond the machines used by
        # CTest. Normalize signed zero because it has no numerical distinction.
        narrowed = np.array(arr, dtype='<f4', order='C', copy=True)
        narrowed[narrowed == 0] = 0
        return narrowed
    raise ValueError(f"unsupported digest dtype: {digest_dtype}")


def digests(path, digest_dtype='native'):
    out = {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind == 'f' and obj.size > 1:
                arr = np.ascontiguousarray(obj[...])
                digest_arr = digest_values(arr, digest_dtype)
                record = {
                    'sha256': hashlib.sha256(digest_arr.tobytes()).hexdigest(),
                    'shape': list(arr.shape),
                    'min': float(np.nanmin(arr)),
                    'max': float(np.nanmax(arr)),
                    'mean': float(np.nanmean(arr)),
                }
                if digest_dtype != 'native':
                    record['digest_dtype'] = digest_dtype
                out[name] = record
        f.visititems(visit)
    return out


def reference_digest_dtype(ref):
    digest_dtypes = {record.get('digest_dtype', 'native')
                     for record in ref.values()}
    if len(digest_dtypes) != 1:
        raise ValueError('reference mixes digest dtypes')
    digest_dtype = digest_dtypes.pop()
    if digest_dtype not in VALID_DIGEST_DTYPES:
        raise ValueError(f"reference uses unsupported digest dtype: {digest_dtype}")
    return digest_dtype


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--update', action='store_true')
    p.add_argument('--digest-dtype', choices=('auto',) + VALID_DIGEST_DTYPES,
                   default='auto',
                   help='bytes to hash; auto reads the mode from the reference')
    a = p.parse_args()

    if not os.path.isfile(a.output):
        print(f"FAIL: model produced no output at {a.output}", file=sys.stderr)
        return 1

    if a.update:
        digest_dtype = 'native' if a.digest_dtype == 'auto' else a.digest_dtype
        got = digests(a.output, digest_dtype)
        if not got:
            print(f"FAIL: no comparable datasets in {a.output}", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(os.path.abspath(a.reference)), exist_ok=True)
        json.dump(got, open(a.reference, 'w'), indent=1, sort_keys=True)
        print(f"[check_output] wrote {digest_dtype} reference for {len(got)} "
              f"variables -> {a.reference}")
        return 0

    if not os.path.isfile(a.reference):
        print(f"FAIL: missing reference {a.reference}. Regenerate with --update.", file=sys.stderr)
        return 1
    ref = json.load(open(a.reference))
    try:
        reference_dtype = reference_digest_dtype(ref)
    except ValueError as error:
        print(f"FAIL: invalid reference: {error}", file=sys.stderr)
        return 1
    digest_dtype = reference_dtype if a.digest_dtype == 'auto' else a.digest_dtype
    if digest_dtype != reference_dtype:
        print(f"FAIL: requested {digest_dtype} digest but reference uses "
              f"{reference_dtype}", file=sys.stderr)
        return 1

    got = digests(a.output, digest_dtype)
    if not got:
        print(f"FAIL: no comparable datasets in {a.output}", file=sys.stderr)
        return 1

    bad, missing = [], []
    for name, r in sorted(ref.items()):
        g = got.get(name)
        if g is None:
            missing.append(name)
        elif g['shape'] != r['shape'] or g['sha256'] != r['sha256']:
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
    precision = ('bit-for-bit' if digest_dtype == 'native'
                 else 'at canonical float32 digest precision')
    print(f"PASS: {len(ref)} variables {precision}{note}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
