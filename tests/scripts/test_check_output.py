#!/usr/bin/env python3
"""Focused tests for exact and cross-machine-portable output digests."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

import check_output


class CheckOutputDigestTest(unittest.TestCase):
    def write_values(self, path, values):
        with h5py.File(path, 'w') as output:
            output.create_dataset('state', data=values)

    def test_float32_digest_ignores_low_fp64_mantissa_bits(self):
        values = np.array([0.1, -3.25, 1.0e-12, 1.0e10], dtype=np.float64)
        adjacent = np.nextafter(values, np.full(values.shape, np.inf))

        with tempfile.TemporaryDirectory() as directory:
            original_path = os.path.join(directory, 'original.h5')
            adjacent_path = os.path.join(directory, 'adjacent.h5')
            self.write_values(original_path, values)
            self.write_values(adjacent_path, adjacent)

            exact_original = check_output.digests(original_path)
            exact_adjacent = check_output.digests(adjacent_path)
            portable_original = check_output.digests(original_path, 'float32')
            portable_adjacent = check_output.digests(adjacent_path, 'float32')

        self.assertNotEqual(exact_original['state']['sha256'],
                            exact_adjacent['state']['sha256'])
        self.assertEqual(portable_original['state']['sha256'],
                         portable_adjacent['state']['sha256'])
        self.assertEqual(check_output.reference_digest_dtype(portable_original),
                         'float32')

    def test_roundoff_digest_handles_midpoints_and_near_zero(self):
        # Adjacent FP64 values straddle both an FP32 and a quantization midpoint.
        values = np.array([1.0 + 2**-24, -0.5, 1e-30, 0.0])
        adjacent = np.nextafter(values, np.inf)
        self.assertFalse(np.array_equal(values.astype('f4'), adjacent.astype('f4')))
        quantum = 2**-23
        reference = check_output.make_roundoff_reference(values, quantum, 1e-12)
        self.assertIn(0, reference['boundary_indices'])
        adjacent[2:] += 1e-14
        self.assertTrue(check_output.matches_roundoff_reference(values, reference))
        self.assertTrue(check_output.matches_roundoff_reference(adjacent, reference))

    def test_roundoff_digest_checks_every_cell(self):
        values = np.array([0.1, 0.2, 0.3, 0.4])
        reference = check_output.make_roundoff_reference(values, 1e-8, 1e-12)
        for index in range(values.size):
            changed = values.copy()
            changed[index] += 2e-8
            self.assertFalse(check_output.matches_roundoff_reference(changed, reference))
        # A permutation preserves min, max and mean but must fail the digest.
        self.assertFalse(check_output.matches_roundoff_reference(values[::-1], reference))

    def test_roundoff_digest_rejects_nonfinite_values_and_large_boundary_changes(self):
        values = np.array([0.5, 0.0])
        reference = check_output.make_roundoff_reference(values, 1.0, 1e-12)
        for invalid in (np.nan, np.inf, -np.inf, 0.5 + 1e-10):
            changed = values.copy()
            changed[0] = invalid
            self.assertFalse(check_output.matches_roundoff_reference(changed, reference))

    def test_roundoff_reference_rejects_invalid_parameters(self):
        values = np.array([1.0, 2.0])
        for quantum, tolerance in ((0, 1e-12), (1, 0), (1, 0.5), (np.inf, 1e-12)):
            with self.assertRaises(ValueError):
                check_output.make_roundoff_reference(values, quantum, tolerance)
        with self.assertRaises(ValueError):
            check_output.make_roundoff_reference(np.array([np.nan]), 1e-8, 1e-12)

    def test_cli_fallback_is_explicit_and_preserves_shape_and_native_checks(self):
        values = np.array([1.0 + 2**-24, 0.0, 3.0])
        adjacent = np.nextafter(values, np.inf)
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, 'output.h5')
            reference_path = os.path.join(directory, 'reference.json')
            self.write_values(output_path, values)
            reference = check_output.digests(output_path, 'float32')
            self.write_values(output_path, adjacent)

            def verify():
                with open(reference_path, 'w') as file:
                    json.dump(reference, file)
                return subprocess.run(
                    [sys.executable, check_output.__file__, '--output', output_path,
                     '--reference', reference_path], capture_output=True, text=True)

            self.assertEqual(verify().returncode, 1)
            reference['state']['roundoff'] = check_output.make_roundoff_reference(
                values, 2**-23, 1e-12)
            result = verify()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Bounded roundoff digest: state', result.stdout)

            self.write_values(output_path, adjacent.reshape((1, 3)))
            self.assertEqual(verify().returncode, 1)

            self.write_values(output_path, values)
            reference = check_output.digests(output_path, 'native')
            reference['state']['roundoff'] = check_output.make_roundoff_reference(
                values, 2**-23, 1e-12)
            self.write_values(output_path, adjacent)
            self.assertEqual(verify().returncode, 1)


if __name__ == '__main__':
    unittest.main()
