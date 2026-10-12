#!/usr/bin/env python3
"""Prepare one CLEO coupling test case: configuration plus CLEO input binaries.

Applies one entry of the template's test_scenarios, points every path the case
reads or writes into --out-dir, and runs tools/cleo/gen_cleo_inputs.py on the
result -- the same generator a real run uses, so the test reads binaries made
the way the model's are.

    make_cleo_test_case.py --template tests/configs/cleo_coupling.json \\
        --scenario rain --out-dir build/cleo_tests/rain [--path2cleo /path/to/CLEO]

Writes <out-dir>/config.json, which is what test_cleo_coupling is given.
Each --extra-output NAME also writes <out-dir>/config_NAME.json: the same case
and the same binaries, output to <out-dir>/output_NAME/, so a second run (at
another rank count, say) can be compared superdroplet by superdroplet.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def set_dotted(config, dotted, value):
    node = config
    *parents, leaf = dotted.split(".")
    for key in parents:
        if not isinstance(node.get(key), dict):
            raise SystemExit(f"override {dotted}: '{key}' is not a section of the template")
        node = node[key]
    if leaf not in node:
        # An override for a key the template lacks is almost always a typo, and
        # would otherwise silently test the default instead.
        raise SystemExit(f"override {dotted}: the template has no '{leaf}' to override")
    node[leaf] = value


def build_config(template, scenario, out_dir):
    scenarios = template.get("test_scenarios", {})
    if scenario not in scenarios or scenario.startswith("__"):
        known = ", ".join(k for k in scenarios if not k.startswith("__"))
        raise SystemExit(f"unknown scenario '{scenario}' (known: {known})")

    config = copy.deepcopy(template)
    config.pop("test_scenarios")
    for dotted, value in scenarios[scenario].items():
        if not dotted.startswith("__"):
            set_dotted(config, dotted, value)

    # The C++ side opens output_dir + zarrbasedir, so zarrbasedir stays
    # relative; everything else is absolute so the working directory is moot.
    share = out_dir / "share"
    cleo = config["physics"]["microphysics"]["cleo"]
    config["output"]["output_dir"] = f"{out_dir / 'output'}/"
    cleo["init_gbx_path"] = str(share / "dimlessGBxboundaries.dat")
    cleo["init_supers_path"] = str(share / "dimlessSDsinit.dat")
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--path2cleo", type=Path, default=None,
                   help="CLEO source root, passed through to gen_cleo_inputs.py")
    p.add_argument("--extra-output", action="append", default=[], metavar="NAME",
                   help="also write config_NAME.json, identical but for its output directory")
    a = p.parse_args()

    out_dir = a.out_dir.resolve()
    config = build_config(json.loads(a.template.read_text()), a.scenario, out_dir)

    (out_dir / "output").mkdir(parents=True, exist_ok=True)
    (out_dir / "share").mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=4) + "\n")

    command = [sys.executable, str(ROOT / "tools" / "cleo" / "gen_cleo_inputs.py"),
               "-c", str(config_path)]
    if a.path2cleo is not None:
        command += ["--path2cleo", str(a.path2cleo)]
    print(" ".join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode

    print(f"{a.scenario}: {config_path}")

    for name in a.extra_output:
        extra = copy.deepcopy(config)
        extra["output"]["output_dir"] = f"{out_dir / f'output_{name}'}/"
        (out_dir / f"output_{name}").mkdir(parents=True, exist_ok=True)
        extra_path = out_dir / f"config_{name}.json"
        extra_path.write_text(json.dumps(extra, indent=4) + "\n")
        print(f"{a.scenario}: {extra_path} (same binaries, output_{name}/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
