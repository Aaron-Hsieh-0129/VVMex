#!/usr/bin/env python3
"""Derive a short, deterministic test config from a default_cases config.

Only run length, output cadence, output location, and the stochastic forcing are
touched. Grid, physics, numerics and input files stay exactly as the shipped case
defines them, so the test exercises the real configuration.

random_perturbation is disabled so changes to the pseudorandom mapping do not
obscure regressions in the model physics and numerics.
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

    rp = c.get('dynamics', {}).get('forcings', {}).get('random_perturbation')
    if rp is not None:
        rp['enable'] = False

    os.makedirs(os.path.dirname(os.path.abspath(a.out_config)), exist_ok=True)
    json.dump(c, open(a.out_config, 'w'), indent=2)
    print(f"[make_test_config] {a.case}: {a.seconds:.0f}s, engine={a.engine}, "
          f"random_perturbation=off -> {a.out_config}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
