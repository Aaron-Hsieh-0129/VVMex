#!/usr/bin/env python3
"""Focused tests for exact and cross-machine-portable output digests."""
import os
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

    def test_approved_float32_variants_keep_exact_whole_field_checks(self):
        values = np.array([1.0 + 2**-24, 0.0, 3.0])
        adjacent = np.nextafter(values, np.inf)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'output.h5')
            self.write_values(path, values)
            original = check_output.digests(path, 'float32')['state']
            self.write_values(path, adjacent)
            variant = check_output.digests(path, 'float32')['state']
            self.assertFalse(check_output.matches_reference(variant, original))

            original['sha256_variants'] = {'validated_gpu': variant['sha256']}
            self.assertTrue(check_output.matches_reference(original, original))
            self.assertTrue(check_output.matches_reference(variant, original))

            self.write_values(path, adjacent.reshape((1, 3)))
            changed_shape = check_output.digests(path, 'float32')['state']
            self.assertFalse(check_output.matches_reference(changed_shape, original))

            for index in range(values.size):
                changed = adjacent.copy()
                changed[index] += 1e-6
                self.write_values(path, changed)
                unknown = check_output.digests(path, 'float32')['state']
                self.assertFalse(check_output.matches_reference(unknown, original))

    def test_native_reference_does_not_accept_variants(self):
        reference = {'sha256': 'original', 'shape': [3],
                     'sha256_variants': {'validated_gpu': 'variant'}}
        actual = {'sha256': 'variant', 'shape': [3]}
        self.assertFalse(check_output.matches_reference(actual, reference))


if __name__ == '__main__':
    unittest.main()
