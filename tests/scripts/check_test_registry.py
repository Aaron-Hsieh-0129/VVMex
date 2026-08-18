#!/usr/bin/env python3
"""Check tests/CMakeLists.txt against the files it names.

CMake resolves these paths at configure time and CTest only at run time, so a
case registered without its config, or a baseline left behind by a removed test,
shows up as a red test or as nothing at all -- long after the commit that caused
it. This needs no compiler, no MPI and no GPU, so CI can run it on every push.

Exit status is 1 for a missing file (a test that cannot pass) and 0 for an
orphan (data no test reads); orphans are reported either way.
"""
import argparse, pathlib, re, sys


def strip_comments(text):
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def calls(text, name):
    """Arguments of every name(...) call, as a list of tokens per call."""
    out = []
    for match in re.finditer(re.escape(name) + r"\s*\(([^)]*)\)", text):
        out.append(match.group(1).replace('"', " ").split())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None, help="project root (default: infer)")
    a = p.parse_args()

    root = pathlib.Path(a.root) if a.root else pathlib.Path(__file__).resolve().parents[2]
    tests = root / "tests"
    cases = root / "rundata" / "input_configs" / "default_cases"
    text = strip_comments((tests / "CMakeLists.txt").read_text())

    missing, orphans = [], []
    used_units, used_baselines, used_references = set(), set(), set()

    def need(path, why):
        if not path.exists():
            missing.append(f"{why}: {path.relative_to(root)}")

    # Unit tests: one source file each. The three registration helpers differ
    # only in what they link, so the source-file rule is the same for all.
    unit_names = [c[0] for c in calls(text, "add_vvm_unit_test")]
    unit_names += [c[0] for c in calls(text, "add_vvm_file_unit_test")]
    unit_names += [c[0] for c in calls(text, "add_vvm_device_unit_test")]
    # Hand-registered ones name their source directly.
    unit_names += [
        pathlib.Path(m).stem
        for m in re.findall(r"\$\{TEST_DIR\}/unit/(\w+\.cpp)", text)
    ]
    for name in unit_names:
        if name.startswith("$"):
            continue  # loop variable, expanded below
        used_units.add(name)
        need(tests / "unit" / f"{name}.cpp", f"unit test {name}")

    # foreach(VAR a b c) ... ${VAR}.cpp -- the BP5 unit-test loop.
    for var, body in re.findall(r"foreach\((\w+)\s+([^)]*)\)", text):
        if f"${{{var}}}.cpp" not in text:
            continue
        for name in body.split():
            if name.startswith("$"):
                continue
            used_units.add(name)
            need(tests / "unit" / f"{name}.cpp", f"unit test {name}")

    # add_vvm_test: a config under tests/configs plus a baseline per backend.
    for args in calls(text, "add_vvm_test"):
        name = args[0]
        used_baselines.add(name)
        need(tests / "configs" / f"{name}.json", f"baseline case {name} config")
        for backend in ("baselines", "baselines_cpu"):
            need(tests / backend / f"{name}.h5", f"baseline case {name} {backend}")

    # add_case_test / add_rank_invariance_test: a shipped default_cases config.
    # add_case_test also needs a digest; the GPU tier is the one that must have
    # it, since VVM_TEST_PHYSICS is forced off without CPU references.
    for args in calls(text, "add_case_test"):
        name = args[0]
        used_references.add(name)
        need(cases / f"{name}.json", f"case test {name} config")
        need(tests / "references" / f"{name}.json", f"case test {name} reference")
    for args in calls(text, "add_rank_invariance_test"):
        need(cases / f"{args[0]}.json", f"rank-invariance case {args[0]} config")

    # Model-level configs named by the BP5 tier.
    for name in re.findall(r"\$\{TEST_DIR\}/configs/(\w+)\.json", text):
        used_baselines.add(name)
        need(tests / "configs" / f"{name}.json", f"config {name}")
    for var, body in re.findall(r"foreach\((\w+)\s+([^)]*)\)", text):
        if f"configs/${{{var}}}.json" not in text:
            continue
        for name in body.split():
            if not name.startswith("$"):
                used_baselines.add(name)
                need(tests / "configs" / f"{name}.json", f"config {name}")

    # The reverse direction: data no registered test reads.
    for path in sorted((tests / "unit").glob("*.cpp")):
        if path.stem not in used_units:
            orphans.append(f"unit source read by no test: {path.relative_to(root)}")
    for backend in ("baselines", "baselines_cpu"):
        for path in sorted((tests / backend).glob("*.h5")):
            if path.stem not in used_baselines:
                orphans.append(f"baseline read by no test: {path.relative_to(root)}")
    for backend in ("references", "references_cpu"):
        directory = tests / backend
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.stem not in used_references:
                orphans.append(f"reference read by no test: {path.relative_to(root)}")
    for path in sorted((tests / "configs").glob("*.json")):
        if path.stem not in used_baselines:
            orphans.append(f"config read by no test: {path.relative_to(root)}")

    for line in orphans:
        print(f"orphan:  {line}")
    for line in missing:
        print(f"MISSING: {line}", file=sys.stderr)

    if missing:
        print(f"\n{len(missing)} file(s) named by tests/CMakeLists.txt do not exist.",
              file=sys.stderr)
        return 1
    print(f"tests/CMakeLists.txt: all named files present "
          f"({len(used_units)} unit sources, {len(used_baselines)} configs, "
          f"{len(used_references)} references); {len(orphans)} orphan(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
