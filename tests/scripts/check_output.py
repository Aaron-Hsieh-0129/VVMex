#!/usr/bin/env python3
"""Regression check against a stored per-variable digest.

Full baseline files do not scale past the 32^2 cases (rcemip output alone is
~1 GB), so the reference is a small JSON of SHA-256 digests over each variable's
values plus a few statistics for diagnostics. Native digests hash the raw bytes
and fail on any bit-level change. A float32 digest first narrows every value to
a canonical little-endian float32; this keeps a whole-field spatial check while
ignoring low FP64 mantissa bits that can vary across GPU architectures. Float32
hashes can still differ at rounding midpoints or near zero. Explicit roundoff
annotations provide a whole-field, bounded-error fallback for those variables.

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


def make_roundoff_reference(arr, quantum, boundary_tolerance):
    """Hash every cell in fixed bins, retaining sparse bin-edge reference values.

    The sparse values make differences up to boundary_tolerance portable even
    when the reference lies on a rounding boundary. A matching digest bounds
    the error of every cell by quantum; it never compares only statistics.
    """
    values = np.asarray(arr, dtype=np.float64).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError('roundoff references require finite values')
    if (not np.isfinite(quantum) or not np.isfinite(boundary_tolerance) or
            not 0 < boundary_tolerance < quantum / 2):
        raise ValueError('require 0 < boundary_tolerance < quantum / 2')
    scaled = values / quantum
    if np.max(np.abs(scaled), initial=0) >= 2**52:
        raise ValueError('quantum is too small for reliable bin indices')
    bins = np.rint(scaled).astype('<i8')
    distance = np.abs(scaled - (np.floor(scaled) + 0.5)) * quantum
    indices = np.flatnonzero(distance <= boundary_tolerance)
    return {
        'sha256': hashlib.sha256(bins.tobytes()).hexdigest(),
        'quantum': float(quantum),
        'boundary_tolerance': float(boundary_tolerance),
        'boundary_indices': indices.tolist(),
        'boundary_values': values[indices].tolist(),
    }


def matches_roundoff_reference(arr, reference):
    values = np.asarray(arr, dtype=np.float64).ravel()
    quantum = reference['quantum']
    tolerance = reference['boundary_tolerance']
    if (not np.isfinite(quantum) or not np.isfinite(tolerance) or
            not 0 < tolerance < quantum / 2):
        raise ValueError('invalid roundoff reference tolerances')
    if not np.all(np.isfinite(values)):
        return False
    scaled = values / quantum
    if np.max(np.abs(scaled), initial=0) >= 2**52:
        return False
    bins = np.rint(scaled).astype('<i8')
    indices = np.asarray(reference['boundary_indices'], dtype=np.int64)
    expected = np.asarray(reference['boundary_values'], dtype=np.float64)
    if (indices.shape != expected.shape or np.any(indices < 0) or
            np.any(indices >= values.size) or np.any(np.diff(indices) <= 0) or
            not np.all(np.isfinite(expected))):
        raise ValueError('invalid roundoff reference boundary values')
    if not np.all(np.abs(values[indices] - expected) <= tolerance):
        return False
    bins[indices] = np.rint(expected / quantum).astype('<i8')
    return hashlib.sha256(bins.tobytes()).hexdigest() == reference['sha256']


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
    p.add_argument('--roundoff-variable', action='append', default=[],
                   help='with --update --digest-dtype float32, annotate this dataset '
                        'with a bounded roundoff digest (repeat for each dataset)')
    p.add_argument('--roundoff-relative-quantum', type=float, default=1e-8,
                   help='roundoff bin width relative to each reference field maximum')
    p.add_argument('--roundoff-relative-boundary-tolerance', type=float, default=1e-12,
                   help='guaranteed portability allowance at bin boundaries')
    a = p.parse_args()
    if a.roundoff_variable and (not a.update or a.digest_dtype != 'float32'):
        p.error('--roundoff-variable requires --update --digest-dtype float32')

    if not os.path.isfile(a.output):
        print(f"FAIL: model produced no output at {a.output}", file=sys.stderr)
        return 1

    if a.update:
        digest_dtype = 'native' if a.digest_dtype == 'auto' else a.digest_dtype
        got = digests(a.output, digest_dtype)
        if not got:
            print(f"FAIL: no comparable datasets in {a.output}", file=sys.stderr)
            return 1
        with h5py.File(a.output, 'r') as output:
            for name in a.roundoff_variable:
                if name not in got:
                    p.error(f'roundoff dataset not found: {name}')
                arr = output[name][...]
                scale = float(np.max(np.abs(arr)))
                quantum = a.roundoff_relative_quantum * scale
                tolerance = a.roundoff_relative_boundary_tolerance * scale
                try:
                    got[name]['roundoff'] = make_roundoff_reference(arr, quantum, tolerance)
                except ValueError as error:
                    p.error(f'invalid roundoff settings for {name}: {error}')
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

    bad, missing, roundoff_matches = [], [], []
    # Only explicitly annotated variables may use the roundoff fallback.
    # Native and ordinary float32 references retain their exact digest checks.
    output = h5py.File(a.output, 'r')
    for name, r in sorted(ref.items()):
        g = got.get(name)
        if g is None:
            missing.append(name)
        elif g['shape'] != r['shape']:
            bad.append((name, r, g))
        elif g['sha256'] != r['sha256']:
            try:
                portable = r.get('roundoff')
                matches = (reference_dtype == 'float32' and portable is not None and
                           matches_roundoff_reference(output[name][...], portable))
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                print(f"FAIL: invalid roundoff reference for {name}: {error}",
                      file=sys.stderr)
                output.close()
                return 1
            if matches:
                roundoff_matches.append(name)
            else:
                bad.append((name, r, g))
    output.close()

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
    if roundoff_matches:
        precision = f'checked ({len(roundoff_matches)} bounded roundoff fallbacks)'
    print(f"PASS: {len(ref)} variables {precision}{note}")
    if roundoff_matches:
        print('  Bounded roundoff digest: ' + ', '.join(roundoff_matches))
    return 0


if __name__ == '__main__':
    sys.exit(main())
