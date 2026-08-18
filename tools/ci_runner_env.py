#!/usr/bin/env python3
"""Emit a GitHub Actions runner `.env` for one CMake preset.

A self-hosted runner started as a service does not read an interactive shell
profile, so it has none of the environment a VVMex build needs. Everything that
environment depends on is already derived from CMakePresets.json by
submit.py:setup_environment(), so this reuses that rather than keeping a second
copy of the lib/lib64 and HPC-X rules in sync with it.

    tools/ci_runner_env.py --preset blaze-cpu > /path/to/runner/.env
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys


def load_submit(vvm_root):
    path = os.path.join(vvm_root, "submit.py")
    spec = importlib.util.spec_from_file_location("vvm_submit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preset_entry(vvm_root, name):
    with open(os.path.join(vvm_root, "CMakePresets.json")) as f:
        data = json.load(f)
    for p in data.get("configurePresets", []):
        if p.get("name") == name:
            return p
    return None


def tool_dirs():
    """Directories holding the tools a build step shells out to.

    Captured from this machine rather than assumed: cmake and python are often
    in a conda prefix or a module, and the service PATH would otherwise miss
    them. git is included because actions/checkout needs it.
    """
    dirs = []
    for tool in ("cmake", "ctest", "python3", "git"):
        found = shutil.which(tool)
        if found:
            d = os.path.dirname(os.path.realpath(found))
            if d not in dirs:
                dirs.append(d)
        else:
            print(f"[warning] '{tool}' not found on PATH; the runner will not "
                  f"be able to use it either.", file=sys.stderr)
    return dirs


def toolchain_lib_dirs(entry, gpu):
    """Runtime library directories of the compiler stack named by the preset.

    submit.py deliberately does not add these -- an interactive run already has
    them from the user's shell. A runner started as a service has no shell, so
    without them the build links but nothing can load libmpi or libnvhpc at run
    time. Derived from the preset so this stays correct on a machine whose
    NVHPC lives somewhere else.
    """
    cache = entry.get("cacheVariables", {})
    dirs = []

    def add(path):
        if path and os.path.isdir(path) and path not in dirs:
            dirs.append(path)

    cxx = cache.get("CMAKE_CXX_COMPILER", "")
    if "/ompi/bin/" in cxx:
        hpcx = cxx.split("/ompi/bin/")[0]
        for sub in ("ompi/lib", "ucx/lib", "sharp/lib",
                    "nccl_rdma_sharp_plugin/lib"):
            add(os.path.join(hpcx, sub))

    nvhpc = cache.get("NVHPC_DIR", "")
    if nvhpc:
        add(os.path.join(nvhpc, "compilers", "lib"))
        add(os.path.join(nvhpc, "math_libs", "lib64"))
        # Not gated on `gpu`: HPC-X's OpenMPI is CUDA-aware, so even a CPU
        # build resolves libcudart.so.12 through it. Verified by running the
        # CPU binary in an environment containing only this file.
        cuda_root = os.path.join(nvhpc, "cuda")
        if os.path.isdir(cuda_root):
            for version in sorted(os.listdir(cuda_root), reverse=True):
                add(os.path.join(cuda_root, version, "lib64"))

    # --gcc-toolchain=<prefix> matters at run time too: NVHPC compiles against
    # one libstdc++ and would otherwise load whatever the system provides.
    for flag_key in ("CMAKE_CXX_FLAGS", "CMAKE_C_FLAGS"):
        for token in str(cache.get(flag_key, "")).split():
            if token.startswith("--gcc-toolchain="):
                prefix = token.split("=", 1)[1]
                add(os.path.join(prefix, "lib64"))
                add(os.path.join(prefix, "lib"))

    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--build-jobs", type=int, default=0,
                    help="compile parallelism (default: all cores)")
    ap.add_argument("--test-threads", type=int, default=0,
                    help="OMP_NUM_THREADS per rank for CPU tests "
                         "(default: half the cores, capped at 64)")
    ap.add_argument("--ctest-jobs", type=int, default=0,
                    help="CTest PROCESSORS budget (default: all cores)")
    a = ap.parse_args()

    vvm_root = os.environ.get("VVM_ROOT") or os.getcwd()
    if not os.path.isfile(os.path.join(vvm_root, "CMakePresets.json")):
        sys.exit(f"error: {vvm_root} does not look like a VVMex checkout "
                 f"(no CMakePresets.json). Export VVM_ROOT.")

    entry = preset_entry(vvm_root, a.preset)
    if entry is None:
        sys.exit(f"error: preset '{a.preset}' is not in CMakePresets.json")

    submit = load_submit(vvm_root)

    # setup_environment layers the preset's directories on top of the caller's
    # LD_LIBRARY_PATH. That is right for an interactive run and wrong here: the
    # shell that runs this setup may have a *different* backend's prefix loaded,
    # and a CPU Kokkos shares its SONAME with the CUDA one, so inheriting it
    # would hand the runner a stack that silently resolves to the wrong library.
    # Generate from the preset alone.
    scrubbed = {k: v for k, v in os.environ.items()
                if k not in ("LD_LIBRARY_PATH", "VVM_EXTRA_LD_LIBRARY_PATH")}
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(scrubbed)
    # setup_environment prints progress to stdout; the .env goes to stdout too,
    # so keep them apart.
    stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        env = submit.setup_environment(a.preset)
    finally:
        sys.stdout = stdout
        os.environ.clear()
        os.environ.update(saved)

    cores = os.cpu_count() or 8
    build_jobs = a.build_jobs or cores
    ctest_jobs = a.ctest_jobs or cores
    # Half the cores, capped at 64. The cap is measured, not arbitrary: on a
    # 224-core host the default tier takes 211 s at 64 threads per rank and
    # 311 s at 56, because CTest packs whole tests side by side and 64 lands
    # better on the core topology. More threads per rank is not automatically
    # better -- the default cases are 32x32x33 grids, so past this point a test
    # only takes cores away from the test that would have run beside it.
    test_threads = a.test_threads or min(64, max(4, cores // 2))

    cache = entry.get("cacheVariables", {})
    gpu = str(cache.get("VVM_ENABLE_GPU", "ON")).upper() != "OFF"
    binary_dir = entry.get("binaryDir", "${sourceDir}/build")
    binary_dir = binary_dir.replace("${sourceDir}/", "").replace("${sourceDir}", ".")

    # PATH: the preset's MPI bin first (setup_environment puts it there), then
    # this machine's tool directories, then a minimal system base. The caller's
    # full interactive PATH is deliberately not carried over -- it would pin the
    # runner to whatever happened to be loaded when this ran.
    path_parts = []
    for d in env.get("PATH", "").split(os.pathsep):
        if d and "/ompi/bin" in d and d not in path_parts:
            path_parts.append(d)
    for d in tool_dirs():
        if d not in path_parts:
            path_parts.append(d)
    for d in ("/usr/local/bin", "/usr/bin", "/bin"):
        if d not in path_parts:
            path_parts.append(d)

    lines = [
        f"# Generated by tools/ci_runner_env.py for preset '{a.preset}'.",
        f"# Regenerate after changing CMakePresets.json or moving the TPLs.",
        f"VVM_CI_PRESET={a.preset}",
        f"VVM_CI_BUILD_DIR={binary_dir}",
        f"VVM_CI_BACKEND={'gpu' if gpu else 'cpu'}",
        f"VVM_CI_BUILD_JOBS={build_jobs}",
        f"VVM_CI_CTEST_JOBS={ctest_jobs}",
        f"VVM_CI_TEST_THREADS={test_threads}",
        f"PATH={os.pathsep.join(path_parts)}",
    ]
    for key in ("LD_LIBRARY_PATH", "VVM_EXTRA_LD_LIBRARY_PATH", "HPCX_HOME"):
        value = env.get(key)
        if not value:
            continue
        if key.endswith("PATH"):
            parts = value.split(os.pathsep)
            if key == "LD_LIBRARY_PATH":
                # Preset TPLs stay in front -- CPU and CUDA Kokkos share a
                # SONAME, so whichever comes first is the one that loads.
                parts = parts + toolchain_lib_dirs(entry, gpu)
            seen, kept = set(), []
            for d in parts:
                if d and d not in seen:
                    seen.add(d)
                    kept.append(d)
            value = os.pathsep.join(kept)
        lines.append(f"{key}={value}")

    print("\n".join(lines))
    print(f"[ok] preset '{a.preset}' -> {binary_dir}, backend "
          f"{'GPU' if gpu else 'CPU'}, {cores} cores detected", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
