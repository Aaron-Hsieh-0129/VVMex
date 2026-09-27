#!/usr/bin/env python3
"""Compare RLL BP5 terrain/surface categories against its original spatial input.

Read-only check; does not run the model. Accepts the generated run config path.
"""
import argparse
import json
from pathlib import Path

import adios2
import netCDF4
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('config', type=Path)
args = parser.parse_args()
config = json.loads(args.config.read_text())
path = Path(config['output']['output_dir'])/(config['output']['output_filename_prefix']+'.bp')
with netCDF4.Dataset(config['netcdf_reader']['source_file']) as source, adios2.FileReader(str(path)) as output:
    assert output.read_attribute('horizontal_geometry') == 'regular_latlon'
    expected = np.asarray(source['topo'][:]).copy()
    expected[expected == 0] = config['grid']['horizontal']['n_halo_cells']-1
    actual = np.asarray(output.read('topo', step_selection=[0, 1])).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected, err_msg='Original mountain was changed during RLL initialization')
    for name in ('sea_land_ice_mask', 'vegtype', 'soiltype', 'slopetype'):
        actual = np.asarray(output.read(name, step_selection=[0, 1])).reshape(expected.shape)
        np.testing.assert_array_equal(actual, source[name][:], err_msg=f'Spatial category changed: {name}')
    print('PASS: original grid-index mountain and land/surface categories preserved exactly.')
