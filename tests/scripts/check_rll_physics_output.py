#!/usr/bin/env python3
"""Basic checks for a completed profile-backed RLL BP5 physics smoke run."""
import argparse
import json
from pathlib import Path

import adios2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('config', type=Path)
args = parser.parse_args()
config = json.loads(args.config.read_text())
output = config['output']
path = Path(output['output_dir'])/(output['output_filename_prefix']+'.bp')
summary = {'output':str(path),'steps':[]}
with adios2.FileReader(str(path)) as reader:
    assert reader.read_attribute('horizontal_geometry') == 'regular_latlon'
    available = reader.available_variables()
    count = int(available['model_time_s']['AvailableStepsCount'])
    interval = config['simulation']['output_interval_s']
    end = config['simulation']['total_time_s']
    assert count == int(end/interval)+1, 'missing output steps'
    for step in range(count):
        data = {name:np.asarray(reader.read(name,step_selection=[step,1]))
                for name in output['fields_to_output']}
        for name,value in data.items():
            assert np.isfinite(value).all(), (step,name,'nonfinite')
        time = float(np.asarray(reader.read('model_time_s',step_selection=[step,1])).squeeze())
        assert time == step*interval
        for name in ('qv','qc','qr','qi','qp'):
            assert data[name].min() >= -1e-12, (step,name,'negative mixing ratio',data[name].min())
        np.testing.assert_allclose(data['qp'],data['qc']+data['qr']+data['qi'],rtol=0,atol=1e-12)
        assert data['pbar'].min() > 0 and np.ptp(data['pbar']) > 1000., 'not a pressure profile'
        assert data['rhobar'].min() > 0 and np.ptp(data['rhobar']) > .01, 'not a density profile'
        assert data['qv'].max() > .001, 'dry placeholder atmosphere'
        summary['steps'].append({'time_s':time,'ranges':{
            name:[float(value.min()),float(value.max())] for name,value in data.items()}})
    assert np.max(np.abs(data['lw_heating'])) > 0, 'radiation inactive'
    assert np.max(data['RKM']) > 0, 'turbulence inactive'
    assert np.max(np.abs(data['sfc_flux_qv'])) > 0, 'surface moisture flux inactive'
    for axis, key in [('x','longitude_bounds_deg'),('y','latitude_bounds_deg')]:
        coordinate = np.asarray(reader.read('coordinates/'+axis,step_selection=[0,1]))
        bounds = config['grid']['horizontal']['geometry'][key]
        assert coordinate.min() > bounds[0] and coordinate.max() < bounds[1]
args.config.with_name('basic_results.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: finite moist RLL physics, condensate consistency, active radiation/turbulence/surface, BP5 time and coordinates')
