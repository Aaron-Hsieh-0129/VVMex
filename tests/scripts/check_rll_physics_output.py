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
if not path.exists():
    parser.error(f'BP5 output not found: {path}. Run the model successfully before checking output.')
summary = {'output':str(path),'steps':[]}
with adios2.FileReader(str(path)) as reader:
    assert reader.read_attribute('horizontal_geometry') == 'regular_latlon'
    periodic = config['grid']['horizontal']['topology']['q2'] == 'periodic'
    if periodic:
        assert reader.read_attribute('latitude_topology') == 'experimental_periodic'
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
            minimum = float(data[name].min())
            if minimum < -1e-12:
                index = np.unravel_index(np.argmin(data[name]), data[name].shape)
                stages = {stage: {'at_failure': float(data[stage][index]),
                                  'minimum': float(data[stage].min())}
                          for stage in (name+'_before_p3', name+'_after_p3') if stage in data}
                raise AssertionError({'step': step, 'time_s': time, 'field': name,
                    'minimum': minimum, 'index': tuple(int(i) for i in index),
                    'stage_snapshots': stages,
                    'note': 'No clipping or tolerance relaxation applied.'})
        np.testing.assert_allclose(data['qp'],data['qc']+data['qr']+data['qi'],rtol=0,atol=1e-12)
        assert data['pbar'].min() > 0 and np.ptp(data['pbar']) > 1000., 'not a pressure profile'
        assert data['rhobar'].min() > 0 and np.ptp(data['rhobar']) > .01, 'not a density profile'
        assert data['qv'].max() > .001, 'dry placeholder atmosphere'
        summary['steps'].append({'time_s':time,'ranges':{
            name:[float(value.min()),float(value.max())] for name,value in data.items()}})
        if periodic:
            horizontal = config['grid']['horizontal']
            south, north = np.deg2rad(horizontal['geometry']['latitude_bounds_deg'])
            dphi = (north-south)/horizontal['ny']
            radius = horizontal['geometry']['earth_radius_m']
            phi_z = south+(np.arange(horizontal['ny'])+1)*dphi
            top_u = data['u'].reshape((-1, horizontal['ny'], horizontal['nx']))[-1]
            top_v = data['v'].reshape((-1, horizontal['ny'], horizontal['nx']))[-1]
            top_zeta = data['zeta'].reshape((-1, horizontal['ny'], horizontal['nx']))[-1]
            cycles = np.array([radius*np.cos(south+.5*dphi)*top_u[0].mean(),
                radius*top_v[:,0].sum()])
            if step == 0:
                initial_cycles = cycles.copy()
            # Fixed prescribed harmonic cycles are an explicit model constraint.
            # Scale roundoff by wind*radius*number of summed rows, not near-zero v.
            eps = np.finfo(top_u.dtype).eps
            scale = radius*max(1., np.abs(top_u).max(), np.abs(top_v).max())*horizontal['ny']
            np.testing.assert_allclose(cycles, initial_cycles, rtol=0., atol=100*eps*scale)
            summary['steps'][-1]['top_cycle_integrals'] = cycles.tolist()
            summary['steps'][-1]['area_weighted_top_zeta'] = float(
                np.sum(top_zeta*np.cos(phi_z)[:,None])/(horizontal['nx']*np.cos(phi_z).sum()))
    assert np.max(np.abs(data['lw_heating'])) > 0, 'radiation inactive'
    assert np.max(data['RKM']) > 0, 'turbulence inactive'
    assert np.max(np.abs(data['sfc_flux_qv'])) > 0, 'surface moisture flux inactive'
    for axis, key in [('x','longitude_bounds_deg'),('y','latitude_bounds_deg')]:
        coordinate = np.asarray(reader.read('coordinates/'+axis,step_selection=[0,1]))
        bounds = config['grid']['horizontal']['geometry'][key]
        assert coordinate.min() > bounds[0] and coordinate.max() < bounds[1]
args.config.with_name('basic_results.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: finite moist RLL physics, condensate consistency, active radiation/turbulence/surface, BP5 time and coordinates')
