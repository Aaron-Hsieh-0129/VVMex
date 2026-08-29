#!/usr/bin/env python3
"""Derive a short, deterministic test config from a default_cases config.

Run length, output cadence, and output location are touched. Grid dimensions
stay unchanged unless --nx or --ny is explicitly provided for a
reduced-dimensional test.

random_perturbation is disabled by default. A test may explicitly enable it
when the perturbation is part of its contract; the mapping is
deterministic for a fixed seed, timestep, and global cell coordinate.
"""
import argparse, json, os, sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--source-dir', required=True,
                   help='directory holding <case>.json (rundata/input_configs/default_cases)')
    p.add_argument('--out-config', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seconds', type=float, default=120.0)
    p.add_argument('--engine', default='HDF5',
                   help='output engine; HDF5 keeps CTest free of IO-rank plumbing')
    p.add_argument('--nx', type=int,
                   help='override grid.nx (use 1 for a Y-Z model)')
    p.add_argument('--ny', type=int,
                   help='override grid.ny (use 1 for an X-Z model)')
    p.add_argument('--no-spatial-input', action='store_true',
                   help='omit netcdf_reader for idealized reduced-grid smoke tests')
    p.add_argument('--enable-random-perturbation', action='store_true',
                   help='enable the source case random perturbation settings')
    a = p.parse_args()

    src = os.path.join(a.source_dir, a.case + '.json')
    if not os.path.isfile(src):
        print(f"ERROR: no such case config: {src}", file=sys.stderr)
        return 2
    c = json.load(open(src))

    c['simulation']['total_time_s'] = a.seconds
    c['simulation']['output_interval_s'] = a.seconds
    c['output']['output_dir'] = a.out_dir
    c['output']['engine'] = a.engine
    if a.nx is not None:
        c['grid']['nx'] = a.nx
    if a.ny is not None:
        c['grid']['ny'] = a.ny
    if a.no_spatial_input:
        c.pop('netcdf_reader', None)

    rp = c.get('dynamics', {}).get('forcings', {}).get('random_perturbation')
    if a.enable_random_perturbation:
        if rp is None:
            p.error('--enable-random-perturbation requires source case settings')
        rp['enable'] = True
    elif rp is not None:
        rp['enable'] = False
    perturbation_state = 'absent' if rp is None else (
        'on' if rp.get('enable', False) else 'off')

    os.makedirs(os.path.dirname(os.path.abspath(a.out_config)), exist_ok=True)
    json.dump(c, open(a.out_config, 'w'), indent=2)
    print(f"[make_test_config] {a.case}: {a.seconds:.0f}s, engine={a.engine}, "
          f"random_perturbation={perturbation_state} -> {a.out_config}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
