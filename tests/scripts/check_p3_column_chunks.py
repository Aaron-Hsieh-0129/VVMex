#!/usr/bin/env python3
"""Check moist P3 output, history and precipitation across column chunk sizes."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np


def datasets(path):
    result = {}
    with h5py.File(path) as file:
        file.visititems(lambda name, obj: result.update({name: obj[...]})
                        if isinstance(obj, h5py.Dataset) else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--launcher', default='mpiexec')
    parser.add_argument('--wrapper', default='')
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--reference-executable', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.work.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='chunks-', dir=args.work.resolve()))
    base = json.loads((root / 'rundata/input_configs/default_cases/p3_bubble_shear.json').read_text())
    base['simulation'].update(total_time_s=4., output_interval_s=2.)
    base['physics']['turbulence']['enable_turbulence'] = False
    base['physics']['p3']['output_stage_diagnostics'] = True
    # Supersaturate the warm/moist sounding so these short runs exercise
    # condensation and latent heating, as well as the dry column path aloft.
    source = root / base['initial_conditions']['source_file']
    lines = source.read_text().splitlines()
    rows = [line.split() for line in lines[1:] if line.strip()]
    for row in rows:
        row[2] = str(1.5 * float(row[2]))
    profile = work / 'moist-profile.txt'
    profile.write_text(lines[0] + '\n' + '\n'.join(' '.join(row) for row in rows) + '\n')
    base['initial_conditions']['source_file'] = str(profile)
    fields = ['th', 'qv', 'qc', 'qr', 'qi', 'qm', 'nc', 'nr', 'ni', 'bm',
              'qp', 'T', 'diag_eff_radius_qc', 'diag_eff_radius_qi', 'diag_eff_radius_qr',
              'precip_liq_surf_mass', 'precip_ice_surf_mass',
              'precip_liq_surf_flux', 'precip_ice_surf_flux',
              'th_m_diag', 'qv_m_diag', 'qc_before_p3', 'qc_after_p3', 'th_after_p3']
    base['output'].update(engine='HDF5', precision='float64',
                          fields_to_output=fields, output_initial_step=False)

    def run_model(config, exe, tag):
        config_path = work / f'config-{tag}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        cmd = [args.launcher, '--bind-to', 'none', '-n', str(args.ranks)]
        if args.wrapper:
            cmd.append(args.wrapper)
        cmd.extend([str(exe.resolve()), str(config_path)])
        result = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=180)
        (work / f'run-{tag}.log').write_text(result.stdout)
        assert result.returncode == 0, f'P3 run {tag} failed:\n{result.stdout}'
        return result.stdout

    # Exceed the former 16384-column cap without advancing or writing output.
    # A standard idealized case permits initialization without a spatial file.
    default = copy.deepcopy(base)
    default['grid']['ny'] = 40 * args.ranks
    default['simulation'].update(total_time_s=0., idealized_test='advection_u')
    default.pop('netcdf_reader')
    default['physics']['p3'].pop('column_chunk_size', None)
    default['output']['output_dir'] = str(work / 'default-capacity')
    log = run_model(default, args.executable, 'default-capacity')
    assert 'p3 columns per chunk = 20480 / 20480' in log, (
        'Default P3 chunk must include every local column, even above 16384')
    print(f'PASS: default uses all 20480 columns/rank, ranks={args.ranks}')

    # 8192 columns/rank for one rank. Neither small chunk divides the domain
    # or a horizontal row; both exercise a partially occupied final buffer.
    sizes = [None, base['grid']['nx'] * base['grid']['ny'], 257, 511]
    runs = [(args.executable, size) for size in sizes]
    if args.reference_executable:
        runs.insert(0, (args.reference_executable, sizes[1]))
    reference = None
    for index, (exe, size) in enumerate(runs):
        config = copy.deepcopy(base)
        if size is None:
            config['physics']['p3'].pop('column_chunk_size', None)
        else:
            config['physics']['p3']['column_chunk_size'] = size
        out = work / f'run-{index}'
        config['output']['output_dir'] = str(out)
        run_model(config, exe, str(index))
        files = sorted(out.glob('*.h5'))
        assert len(files) == 2, f'Expected two precipitation output intervals: {files}'
        snapshots = [datasets(path) for path in files]
        for snapshot in snapshots:
            assert set(fields).issubset({name.rsplit('/', 1)[-1] for name in snapshot}), (
                'Requested P3 diagnostics are missing')
            for name, data in snapshot.items():
                if data.dtype.kind == 'f':
                    assert np.isfinite(data).all(), f'Nonfinite {name}, chunk {size}'
        if reference is None:
            reference = snapshots
            # Exercise warm rain, ice production and surface precipitation,
            # rather than comparing inactive P3 fields filled with zeros.
            for field in ['qc', 'qr', 'qi', 'precip_liq_surf_mass']:
                assert any(name.rsplit('/', 1)[-1] == field and np.any(data > 0)
                           for name, data in snapshots[-1].items()), f'No {field} produced'
        else:
            for expected, actual in zip(reference, snapshots):
                assert expected.keys() == actual.keys(), 'Output schema changed'
                for name in expected:
                    before, after = expected[name], actual[name]
                    assert before.dtype == after.dtype and before.shape == after.shape, (
                        f'{name} layout differs with chunk {size}')
                    assert before.tobytes() == after.tobytes(), (
                        f'{name} differs bitwise with chunk {size}; '
                        f'max absolute difference={np.max(np.abs(before - after))}')
        print(f'PASS: {exe.name}, chunk={size}, ranks={args.ranks}; all datasets match bitwise')
    print(f'Artifacts: {work}')


if __name__ == '__main__':
    main()
