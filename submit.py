#!/usr/bin/env python3

import argparse
import fnmatch
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys

try:
    import readline
except ImportError:
    readline = None


# ==============================================================================
# Defaults
# ==============================================================================

DEFAULT_CONFIG = "rundata/input_configs/default_cases/advection_u.json"
DEFAULT_COMPUTE = 1
DEFAULT_IO = None                 # None means infer from output.engine
DEFAULT_NODES = 1
DEFAULT_GPUS = None               # None means infer from compute ranks per node
DEFAULT_CPUS = None               # SLURM fills the node; local mode falls back to 1
DEFAULT_OMP_THREADS = None        # Defaults to allocated CPUs per task
DEFAULT_IO_CPUS = 1               # IO ranks are host-only and effectively single-threaded
DEFAULT_TIME = "24:00:00"
DEFAULT_OUT = None                # None means <output.output_dir>/%j.out
DEFAULT_ERR = None                # None means <output.output_dir>/%j.err
DEFAULT_LOG_OUT_NAME = "%j.out"
DEFAULT_LOG_ERR_NAME = "%j.err"
DEFAULT_JOB_NAME = "VVMex"
DEFAULT_ACCOUNT = "MST114418"
DEFAULT_PARTITION = "normal"
DEFAULT_EXPORT = "ALL"
DEFAULT_EXCLUSIVE = True


# ==============================================================================
# Path / environment helpers
# ==============================================================================

def get_vvm_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_available_presets(vvm_root):
    preset_file = os.path.join(vvm_root, "CMakePresets.json")
    if not os.path.exists(preset_file):
        return []
    try:
        with open(preset_file, "r") as f:
            presets_data = json.load(f)
        return [
            p.get("name")
            for p in presets_data.get("configurePresets", [])
            if p.get("name")
        ]
    except Exception:
        return []


def _append_lib_dirs(lib_dirs, key, val):
    """Add the shared-library directories of one dependency prefix to lib_dirs."""
    # A value may be an install prefix (HDF5_DIR=/opt/foo) or a CMake package
    # directory (ADIOS2_DIR=/opt/foo/lib/cmake/adios2). Joining "lib" onto the
    # latter yields a path that does not exist, and the library then silently
    # resolves from whatever else is on LD_LIBRARY_PATH.
    marker = os.sep + "cmake" + os.sep
    if marker in val:
        cand = val.split(marker)[0]
        # A prefix may name lib/cmake/<pkg> while the install actually put the
        # .so files in lib64 (CMake finds either). Landing on the empty one is
        # silent: the base stack further down LD_LIBRARY_PATH then supplies the
        # library, which is exactly the override this list exists to prevent.
        #
        # Take the sibling as well, not just as a fallback: a prefix can split
        # itself across both, as VVMex_libs does with ADIOS2 in lib64 and the
        # libfabric that its RDMA data plane needs in lib.
        head, tail = os.path.split(cand)
        sibling = os.path.join(head, "lib64" if tail == "lib" else "lib")
        found = [d for d in (cand, sibling) if os.path.isdir(d)]
        if not found:
            print(f"[Warning] {key}: neither {cand} nor {sibling} exists.")
        for d in found:
            append_unique(lib_dirs, d)
    else:
        append_unique(lib_dirs, os.path.join(val, "lib"))
        append_unique(lib_dirs, os.path.join(val, "lib64"))


def _read_cmake_cache(binary_dir):
    """Read NAME:TYPE=VALUE entries out of a configured tree's CMakeCache.txt."""
    cache_file = os.path.join(binary_dir, "CMakeCache.txt")
    values = {}
    try:
        with open(cache_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "//")):
                    continue
                name, sep, value = line.partition("=")
                if not sep or ":" not in name:
                    continue
                values[name.split(":", 1)[0]] = value
    except OSError:
        pass
    return values


def preset_uses_gpu(vvm_root, preset_name):
    """Return whether a CMake preset selects the GPU execution backend."""
    preset_file = os.path.join(vvm_root, "CMakePresets.json")
    try:
        with open(preset_file, "r") as f:
            presets_data = json.load(f)
        for preset in presets_data.get("configurePresets", []):
            if preset.get("name") == preset_name:
                value = preset.get("cacheVariables", {}).get("VVM_ENABLE_GPU", "ON")
                return str(value).strip().upper() not in ("OFF", "0", "FALSE", "NO")
    except Exception:
        pass
    # Existing presets predate VVM_ENABLE_GPU and are GPU builds by default.
    return True


def append_unique(paths, path):
    if not path:
        return

    if not os.path.isdir(path):
        return

    if path not in paths:
        paths.append(path)


def setup_environment(preset_name):
    """
    Load controlled build-time dependency paths from CMakePresets.json.

    Important:
    - Do not stage/copy libfabric.
    - Do not prepend .vvm_runtime_libs.
    - Do not manually add /usr/lib*, /lib*, etc. into LD_LIBRARY_PATH.

    Forcing broad system paths into LD_LIBRARY_PATH can mix incompatible
    MPI / UCX / UCC / libfabric / HDF5 / ADIOS2 runtime stacks.
    """
    env = os.environ.copy()
    vvm_root = get_vvm_root()
    env["VVM_ROOT"] = vvm_root

    preset_file = os.path.join(vvm_root, "CMakePresets.json")
    if not os.path.exists(preset_file):
        print(f"[Warning] {preset_file} not found. Environment may be incomplete.")
        return env

    try:
        with open(preset_file, "r") as f:
            presets_data = json.load(f)

        cache_vars = None
        binary_dir_raw = "${sourceDir}/build"
        for p in presets_data.get("configurePresets", []):
            if p.get("name") == preset_name:
                cache_vars = dict(p.get("cacheVariables", {}))
                binary_dir_raw = p.get("binaryDir", binary_dir_raw)
                break

        if cache_vars is None:
            print(f"[Warning] Preset '{preset_name}' not found in CMakePresets.json.")
            return env

        # A preset names only what is machine-specific; CMake derives the compiler
        # wrappers and the individual dependency prefixes from it. The configured
        # tree's CMakeCache holds what the binary was actually linked against, so
        # it wins over the preset wherever both have an entry.
        configured = _read_cmake_cache(binary_dir_raw.replace("${sourceDir}", vvm_root))
        if configured:
            cache_vars.update({k: v for k, v in configured.items() if v})

        print(f"[Info] Loaded environment from CMake preset: '{preset_name}'")

        # ----------------------------------------------------------------------
        # Execution backend
        # ----------------------------------------------------------------------
        # Both the backend and the binary path are read from the same preset that
        # built the code, so the launcher cannot disagree with the binary: a CPU
        # build is never handed a GPU mapping, and vice versa.
        # ----------------------------------------------------------------------
        gpu_enabled = str(cache_vars.get("VVM_ENABLE_GPU", "ON")).strip().upper() \
            not in ("OFF", "0", "FALSE", "NO")
        env["VVM_BACKEND"] = "gpu" if gpu_enabled else "cpu"

        binary_dir = binary_dir_raw.replace("${sourceDir}", vvm_root)
        env["VVM_BINARY"] = os.path.join(binary_dir, "vvm")
        print(f"[Info] Execution backend: {env['VVM_BACKEND']}  binary: {env['VVM_BINARY']}")

        # ----------------------------------------------------------------------
        # HPCX / MPI environment metadata
        # ----------------------------------------------------------------------
        # Only set metadata and PATH here. Do not force libfabric / UCX / UCC
        # paths through submit.py. core_run.sh may source hpcx-init.sh, and the
        # runtime stack should remain internally consistent.
        # ----------------------------------------------------------------------
        cxx_compiler = cache_vars.get("CMAKE_CXX_COMPILER", "")
        mpi_wrapper_names = {"mpic++", "mpicxx", "mpiCC"}
        if os.path.basename(cxx_compiler) in mpi_wrapper_names:
            mpi_bin = os.path.dirname(cxx_compiler)
            if os.path.isdir(mpi_bin):
                # Compile and launch with the same MPI implementation.  This is
                # needed for ordinary Open MPI prefixes as well as HPC-X; an
                # inherited Intel MPI mpirun cannot launch this binary.
                env["PATH"] = mpi_bin + os.pathsep + env.get("PATH", "")

        if "/ompi/bin/" in cxx_compiler:
            hpcx_home = cxx_compiler.split("/ompi/bin/")[0]
            my_plugin_path = f"{hpcx_home}/nccl_rdma_sharp_plugin/lib"
            sharp_lib_path = f"{hpcx_home}/sharp/lib"

            env["HPCX_HOME"] = hpcx_home
            env["MY_PLUGIN_PATH"] = my_plugin_path
            env["SHARP_LIB_PATH"] = sharp_lib_path
            env["VVM_ENV_SCRIPT"] = f"{hpcx_home}/hpcx-init.sh"
            env["VVM_PRE_RUN_CMD"] = f"source {hpcx_home}/hpcx-init.sh"

        # ----------------------------------------------------------------------
        # Explicitly configured project dependencies
        # ----------------------------------------------------------------------
        # These paths come from the same CMake preset used to build the code.
        # Avoid adding broad system library paths here.
        # ----------------------------------------------------------------------
        lib_dirs = []

        # Kokkos_DIR first: dependency prefixes often contain another Kokkos with
        # the same SONAMEs. If an ADIOS2/HDF5 base stack wins here, it silently
        # replaces the exact Kokkos backend and build options selected by CMake.
        #
        # ADIOS2_DIR remains ahead of the HDF5/NetCDF base stack so an explicit
        # ADIOS2 override still wins over libadios2 copies in that base prefix.
        for key in [
            "Kokkos_DIR",
            "ADIOS2_DIR",
            "HDF5_DIR",
            "NETCDF_C_DIR",
            "NETCDF_Fortran_DIR",
            "PNETCDF_DIR",
        ]:
            raw = cache_vars.get(key, "")
            if not raw:
                continue
            _append_lib_dirs(lib_dirs, key, str(raw))

        if lib_dirs:
            extra_ld = ":".join(lib_dirs)

            # Respect a caller-supplied VVM_EXTRA_LD_LIBRARY_PATH instead of
            # discarding it. It goes first, so an operator can override one library
            # of the preset's stack -- e.g. point at an ADIOS2 built without Kokkos
            # -- without editing CMakePresets.json.
            caller_extra = os.environ.get("VVM_EXTRA_LD_LIBRARY_PATH", "")
            if caller_extra:
                extra_ld = caller_extra + ":" + extra_ld
                print(f"[Info] Honouring caller VVM_EXTRA_LD_LIBRARY_PATH: {caller_extra}")

            env["VVM_EXTRA_LD_LIBRARY_PATH"] = extra_ld

            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = extra_ld + (":" + old_ld if old_ld else "")

    except Exception as e:
        print(f"[Warning] Error parsing CMakePresets.json: {e}")

    return env


# ==============================================================================
# Config helpers
# ==============================================================================

def ask(prompt_text, default_val):
    ans = input(f"{prompt_text} [{default_val}]: ").strip()
    return ans if ans else default_val


def ask_path(prompt_text, default_val):
    """
    Prompt for a file/path with TAB completion.

    Works in normal Linux terminals. If readline is unavailable, falls back to
    normal input().
    """
    if readline is None:
        return ask(prompt_text, default_val)

    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()

    def path_completer(text, state):
        expanded_text = os.path.expanduser(text)
        matches = glob.glob(expanded_text + "*")

        results = []
        for match in matches:
            if os.path.isdir(match):
                match += os.sep
            results.append(match)

        results = sorted(results)

        try:
            return results[state]
        except IndexError:
            return None

    try:
        readline.set_completer(path_completer)

        # Do not treat "/" as a word separator.
        readline.set_completer_delims(" \t\n")

        readline.parse_and_bind("tab: complete")

        ans = input(f"{prompt_text} [{default_val}]: ").strip()
        return ans if ans else default_val

    finally:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


def read_config(config_path):
    config_path_abs = os.path.abspath(os.path.expanduser(config_path))

    if not os.path.isfile(config_path_abs):
        raise FileNotFoundError(f"Configuration file not found: {config_path_abs}")

    with open(config_path_abs, "r") as f:
        return json.load(f), config_path_abs


def peek_io_engine(config_path):
    try:
        config_data, _ = read_config(config_path)
        return config_data.get("output", {}).get("engine", "HDF5")
    except Exception:
        return "HDF5"


def peek_output_dir(config_path):
    try:
        config_data, _ = read_config(config_path)
        return config_data.get("output", {}).get("output_dir", "")
    except Exception:
        return ""


def default_log_path(out_dir, name, root=None):
    if not out_dir:
        return name
    base = out_dir if os.path.isabs(out_dir) else os.path.join(root or os.getcwd(), out_dir)
    return os.path.join(os.path.normpath(base), name)


def infer_io_tasks(io_engine, compute_tasks, io_tasks):
    """
    IO rank default policy:
    - If user explicitly gives --io N, use it.
    - If SST and --io is omitted, use io = compute.
    - Otherwise use io = 0.
    """
    if io_tasks is not None:
        return io_tasks

    if io_engine == "SST":
        return compute_tasks

    return 0


def infer_gpus_per_node(compute_tasks, nodes, gpus):
    """
    GPU request policy:
    - GPUs are requested for compute ranks only.
    - IO ranks do not increase the GPU request by default.

    Example:
      compute = 64, io = 64, nodes = 8
      compute_per_node = 8
      total_tasks_per_node = 16
      gpus_per_node = 8
    """
    if gpus is not None:
        return gpus

    return max(1, math.ceil(compute_tasks / nodes))


def _slurm_query(cmd, timeout=15):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def query_partitions():
    out = _slurm_query(["sinfo", "-h", "-o", "%P"])
    if not out:
        return [], None

    names, default = [], None
    for entry in out.split():
        name = entry.rstrip("*")
        names.append(name)
        if entry.endswith("*"):
            default = name
    return names, default


def resolve_partition(partition):
    """
    Fall back to the cluster's default partition when the requested one does not
    exist. The shipped default is one site's name for it, and a partition SLURM
    does not know silently defeats every query the CPU sizing depends on.
    """
    names, default = query_partitions()
    if not names:
        return partition, None

    if partition in names:
        return partition, None

    if default:
        return default, f"partition {partition!r} not found; using default partition {default!r}"

    return partition, f"partition {partition!r} not found and SLURM reports no default"


def _parse_gpu_count(detail):
    for key in ("Gres", "CfgTRES"):
        m = re.search(rf"\b{key}=(\S+)", detail)
        if not m:
            continue
        for entry in m.group(1).split(","):
            entry = re.sub(r"\(.*?\)", "", entry)
            if not entry.startswith(("gpu:", "gres/gpu")):
                continue
            tail = entry.split("=")[-1].split(":")[-1]
            if tail.isdigit() and int(tail) > 0:
                return int(tail)
    return None


def query_node_resources(partition):
    """
    What one node of `partition` offers, or None if SLURM cannot say.

    CPUEfctv, not CPUTot: a node can advertise more CPUs than a job is allowed
    to hold, and asking for the difference gets the job rejected rather than
    scheduled. sinfo has no field for the effective count, so this takes a node
    name from the partition and asks scontrol about it.
    """
    out = _slurm_query(["sinfo", "-h", "-p", partition, "-o", "%n"])
    if not out or not out.split():
        return None

    detail = _slurm_query(["scontrol", "show", "node", out.split()[0]])
    if not detail:
        return None

    resources = {"cpus": None, "gpus": _parse_gpu_count(detail)}

    for key in ("CPUEfctv", "CPUTot"):
        m = re.search(rf"\b{key}=(\d+)", detail)
        if m and int(m.group(1)) > 0:
            resources["cpus"] = int(m.group(1))
            break

    part = _slurm_query(["scontrol", "show", "partition", partition]) or ""

    m = re.search(r"\bDefCpuPerGPU=(\d+)", part)
    resources["def_cpu_per_gpu"] = int(m.group(1)) if m else None

    m = re.search(r"\bMaxCPUsPerNode=(\d+)", part)
    resources["max_cpus_per_node"] = int(m.group(1)) if m else None

    return resources


def query_cpus_per_node(partition):
    resources = query_node_resources(partition)
    return resources["cpus"] if resources else None


def infer_cpus_per_task(cpus, partition, tasks_per_node, gpus_per_node=0,
                        compute_per_node=0, io_per_node=0, io_cpus=1):
    """
    CPU request policy when --cpus is not given.

    --cpus-per-task is the only thing that decides how much of a node the job
    actually holds: SLURM allocates it times the tasks placed there. Defaulting
    it to 1 quietly hands back most of the node -- on a 104-CPU node running 16
    ranks that is 16 CPUs used and 88 idle, which starves the host side of every
    compute rank and shows up as low GPU utilization.

    So fill the node, but no further than the CPUs the requested GPUs entitle
    the job to. Where a site rations cores per GPU, dividing the whole node by
    this job's task count over-requests whenever the job uses fewer GPUs than
    the node has, and the job is rejected or held rather than scheduled.
    """
    if cpus is not None:
        return cpus, "explicit", []

    if not partition or tasks_per_node <= 0:
        return 1, "fallback (no partition to query)", []

    resources = query_node_resources(partition)
    if not resources or not resources["cpus"]:
        return 1, "fallback (SLURM did not report CPUs per node)", []

    budget = resources["cpus"]
    reason = f"{budget} usable CPUs/node"

    max_per_node = resources.get("max_cpus_per_node")
    if max_per_node and max_per_node < budget:
        budget = max_per_node
        reason = f"{budget} CPUs/node allowed by the partition"

    cpus_per_gpu = resources.get("def_cpu_per_gpu")
    if not cpus_per_gpu and resources.get("gpus"):
        cpus_per_gpu = resources["cpus"] // resources["gpus"]

    if gpus_per_node > 0 and cpus_per_gpu:
        gpu_budget = gpus_per_node * cpus_per_gpu
        if gpu_budget < budget:
            budget = gpu_budget
            reason = f"{gpus_per_node} GPUs/node x {cpus_per_gpu} CPUs/GPU"

    inferred = max(1, budget // tasks_per_node)
    warnings = []

    if inferred * tasks_per_node > budget:
        warnings.append(
            f"{tasks_per_node} tasks/node x {inferred} CPU needs "
            f"{inferred * tasks_per_node} CPUs, above the {budget} this job is "
            "entitled to. Reduce --io, or set --cpus explicitly if the site allows it."
        )

    if compute_per_node > 0:
        allocated = inferred * tasks_per_node
        per_compute = (allocated - io_per_node * io_cpus) // compute_per_node
        if per_compute < 2:
            warnings.append(
                f"about {max(per_compute, 0)} core(s) per compute rank: CPU affinity "
                "stays off below 2, and the host side of each GPU will be tight."
            )

    return inferred, f"{reason} / {tasks_per_node} tasks/node", warnings


# ==============================================================================
# Snapshot
# ==============================================================================

SNAPSHOT_IGNORE_NAMES = [
    ".git",
    ".vvm_runtime_libs",
    "build",
    "build_cpu",
    "log",
    "rundata",
    "tests",
    "docs",
    "externals",
    "tags",
    "*.o",
    "output",
]

SUBMIT_COMMAND_FILE = "submit_command.sh"

SNAPSHOT_VERSION_FILE = "ORIGINAL_GIT_VERSION.txt"
SNAPSHOT_PATCH_FILE = "ORIGINAL_GIT_UNCOMMITTED.patch"


def git_text(repo_root, args):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.rstrip("\n")


def git_bytes(repo_root, args):
    result = subprocess.run(["git"] + args, cwd=repo_root, capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout


def collect_git_version_info(repo_root):
    head = git_text(repo_root, ["rev-parse", "HEAD"])
    if not head:
        return None

    branch = git_text(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch == "HEAD":
        branch = "(detached HEAD)"

    status = git_text(repo_root, ["status", "--porcelain"])

    return {
        "commit": head,
        "short": git_text(repo_root, ["rev-parse", "--short", "HEAD"]),
        "branch": branch,
        "describe": git_text(repo_root, ["describe", "--tags", "--always", "--dirty"]),
        "author_date": git_text(repo_root, ["log", "-1", "--format=%aI"]),
        "author": git_text(repo_root, ["log", "-1", "--format=%an <%ae>"]),
        "subject": git_text(repo_root, ["log", "-1", "--format=%s"]),
        "remote": git_text(repo_root, ["config", "--get", "remote.origin.url"]),
        "upstream": git_text(repo_root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]),
        "status": status,
        "dirty": bool(status),
    }


def write_git_version_file(snapshot_dir, info, repo_root):
    path = os.path.join(snapshot_dir, SNAPSHOT_VERSION_FILE)

    lines = [
        f"repository       : {repo_root}",
        f"commit           : {info['commit']}",
        f"short_commit     : {info['short']}",
        f"branch           : {info['branch']}",
        f"describe         : {info['describe']}",
        f"upstream         : {info['upstream'] or '(none)'}",
        f"remote_origin    : {info['remote'] or '(none)'}",
        f"commit_author    : {info['author']}",
        f"commit_date      : {info['author_date']}",
        f"commit_subject   : {info['subject']}",
        f"working_tree     : {'modified' if info['dirty'] else 'clean'}",
        "",
        "checkout command:",
        f"  git checkout {info['commit']}",
        "",
    ]

    if info["dirty"]:
        lines.append("uncommitted changes at submit time (git status --porcelain):")
        lines.extend("  " + line for line in info["status"].splitlines())
        lines.append("")
        lines.append(f"the same changes are applied in this snapshot and saved in {SNAPSHOT_PATCH_FILE}")
    else:
        lines.append("no uncommitted changes at submit time")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_git_patch_file(snapshot_dir, repo_root, pathspecs):
    args = ["diff", "HEAD", "--binary"]
    if pathspecs:
        args += ["--"] + pathspecs

    patch = git_bytes(repo_root, args)
    if not patch:
        return

    with open(os.path.join(snapshot_dir, SNAPSHOT_PATCH_FILE), "wb") as f:
        f.write(patch)


def snapshot_ignored(rel_path, ignore_names):
    for part in rel_path.split("/"):
        for pattern in ignore_names:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def snapshot_tracked_paths(repo_root, ignore_names):
    listing = git_text(repo_root, ["ls-files", "-z"])
    paths = [p for p in listing.split("\0") if p]
    return [p for p in paths if not snapshot_ignored(p, ignore_names)]


def create_baseline_history(repo_root, snapshot_dir, info, ignore_names):
    tracked = snapshot_tracked_paths(repo_root, ignore_names)
    if not tracked:
        return False

    modified_state = {}

    for rel in tracked:
        dst = os.path.join(snapshot_dir, rel)
        current = None
        if os.path.isfile(dst) and not os.path.islink(dst):
            with open(dst, "rb") as f:
                current = f.read()
        elif os.path.lexists(dst):
            continue

        original = git_bytes(repo_root, ["show", f"HEAD:{rel}"])
        if original is None:
            continue

        modified_state[rel] = (current, os.stat(dst).st_mode if current is not None else None)

        os.makedirs(os.path.dirname(dst) or snapshot_dir, exist_ok=True)
        with open(dst, "wb") as f:
            f.write(original)

    restored = sorted(modified_state.keys())
    if not restored:
        return False

    subprocess.run(["git", "init", "-q"], cwd=snapshot_dir, check=True)

    exclude_lines = [SNAPSHOT_VERSION_FILE, SNAPSHOT_PATCH_FILE] + [
        n for n in ignore_names if n != ".git"
    ]
    with open(os.path.join(snapshot_dir, ".git", "info", "exclude"), "a") as f:
        f.write("\n" + "\n".join(exclude_lines) + "\n")

    for start in range(0, len(restored), 200):
        subprocess.run(
            ["git", "add", "-f", "--"] + restored[start:start + 200],
            cwd=snapshot_dir,
            check=True,
        )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Snapshot",
            "-c",
            "user.email=snap@local",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            f"Snapshot base of {info['short']} ({info['branch']})",
            "-m",
            f"Original commit: {info['commit']}\nOriginal subject: {info['subject']}",
        ],
        cwd=snapshot_dir,
        check=True,
    )

    for rel, (content, mode) in modified_state.items():
        dst = os.path.join(snapshot_dir, rel)
        if content is None:
            if os.path.exists(dst):
                os.remove(dst)
            continue
        with open(dst, "wb") as f:
            f.write(content)
        if mode is not None:
            os.chmod(dst, mode)

    return True


def write_submit_command(out_dir_abs, vvm_root, args):
    script = os.path.abspath(sys.argv[0])

    command = getattr(args, "equivalent_command", None)
    if command:
        if not os.path.isabs(sys.argv[0]):
            command = command.replace(sys.argv[0], script, 1)
    else:
        command = shlex.join([script] + sys.argv[1:])

    path = os.path.join(out_dir_abs, SUBMIT_COMMAND_FILE)
    with open(path, "w") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            f"export VVM_ROOT={shlex.quote(vvm_root)}\n"
            'cd "$VVM_ROOT"\n'
            "\n"
            f"{command}\n"
        )
    os.chmod(path, 0o755)

    print(f"[Info] Submit command written to: {path}")


def create_code_snapshot(repo_root, snapshot_dir, config_path, prof_path, spat_path, out_dir_raw):
    print(f"\n[Info] Creating code snapshot at: {snapshot_dir}")

    if os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir)

    out_base = os.path.normpath(out_dir_raw).split(os.sep)[0]
    if out_base in [".", "..", ""]:
        out_base = "output"

    ignore_names = SNAPSHOT_IGNORE_NAMES + [out_base]
    ignore_patterns = shutil.ignore_patterns(*ignore_names)

    shutil.copytree(repo_root, snapshot_dir, ignore=ignore_patterns)
    shutil.copy2(config_path, snapshot_dir)

    if prof_path and os.path.isfile(prof_path):
        shutil.copy2(prof_path, snapshot_dir)

    if spat_path and os.path.isfile(spat_path):
        shutil.copy2(spat_path, snapshot_dir)

    info = collect_git_version_info(repo_root)

    if info is None:
        print("[Warning] Source tree is not a git repository; snapshot has no baseline history")
        try:
            subprocess.run(["git", "init", "-q"], cwd=snapshot_dir, check=True)
            with open(os.path.join(snapshot_dir, ".git", "info", "exclude"), "a") as f:
                f.write("\n" + "\n".join(ignore_names) + "\n")
            subprocess.run(["git", "add", "-A"], cwd=snapshot_dir, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Snapshot",
                    "-c",
                    "user.email=snap@local",
                    "commit",
                    "-q",
                    "-m",
                    "Auto Snapshot",
                ],
                cwd=snapshot_dir,
                check=True,
            )
        except Exception as e:
            print(f"[Warning] Git snapshot commit failed; ignored: {e}")
        return

    write_git_version_file(snapshot_dir, info, repo_root)
    write_git_patch_file(snapshot_dir, repo_root, snapshot_tracked_paths(repo_root, ignore_names))

    try:
        if create_baseline_history(repo_root, snapshot_dir, info, ignore_names):
            state = "with local edits kept as uncommitted changes" if info["dirty"] else "clean"
            print(f"[Info] Snapshot git baseline: {info['short']} ({info['branch']}) {state}")
    except Exception as e:
        print(f"[Warning] Git snapshot baseline failed; ignored: {e}")


# ==============================================================================
# Interactive wizard
# ==============================================================================

def interactive_wizard():
    vvm_root = get_vvm_root()
    presets = get_available_presets(vvm_root)
    default_preset = presets[0] if presets else "unknown"

    print("====================================================================")
    print(" GVVM Interactive Setup Wizard")
    print("====================================================================")
    print(" Note: All relative paths are based on the auto-detected $VVM_ROOT.")
    print("")
    print(" This wizard asks only the common options needed for a normal run.")
    print(" Other options are still available from the command line; see below.")
    print(" If you are not sure what to enter, run ./submit.py and follow")
    print(" the prompts step by step. Press Enter to accept a shown default.")
    print("--------------------------------------------------------------------")
    print(" Prompted Options")
    print("--------------------------------------------------------------------")
    print(f" --local          : Run locally without SLURM (default: False)")
    print(f" -c, --config     : Configuration file (default: {DEFAULT_CONFIG})")
    print("                    TAB completion is enabled for this path prompt.")
    print(f" --preset         : CMake preset to load environment from")
    print(f" --compute        : Compute MPI ranks (default: {DEFAULT_COMPUTE})")
    print(f" --nodes          : Number of SLURM nodes (default: {DEFAULT_NODES})")
    print(f" -t, --time       : Wall time limit (default: {DEFAULT_TIME})")
    print(f" --out            : Standard output log file (default: <output.output_dir>/{DEFAULT_LOG_OUT_NAME})")
    print(f" --err            : Standard error log file (default: <output.output_dir>/{DEFAULT_LOG_ERR_NAME})")
    print(f" --job-name       : SLURM job name (default: {DEFAULT_JOB_NAME})")
    print(f" -A, --account    : SLURM account (default: {DEFAULT_ACCOUNT})")
    print(f" -p, --partition  : SLURM partition (default: {DEFAULT_PARTITION})")
    print("--------------------------------------------------------------------")
    print(" Automatic and Advanced Options")
    print("--------------------------------------------------------------------")
    print(" --cpus N         : Host CPUs allocated per MPI task; OpenMP defaults to this.")
    print("                    SLURM default: fill each node across its tasks, capped by the")
    print("                    CPUs the requested GPUs entitle the job to.")
    print("                    Local CLI default: 1, because no partition can be queried.")
    print("                    For a local CPU preset, this wizard prompts with a full-node suggestion.")
    print("")
    print(" --io N           : Number of additional IO-server MPI ranks/processes.")
    print("                    This is a rank count, not a CPU count.")
    print("                    SST default: --io equals --compute; HDF5 default: 0.")
    print("                    IO ranks are only valid with the SST output engine.")
    print("")
    print(f" --io-cpus N      : Host CPUs/threads reserved for each IO rank (default: {DEFAULT_IO_CPUS}).")
    print("                    This does not create IO ranks and has no effect at --io 0.")
    print("                    It exists for both presets; normally leave it at 1 unless tuning SST IO.")
    print("")
    print(" --gpus N         : GPUs requested per node by GPU presets only.")
    print("                    Default: ceil(compute ranks / nodes). IO ranks add no GPUs.")
    print("                    CPU presets request no GPUs and ignore VVM_GPU_LIST.")
    print("")
    print(" --ntasks         : compute ranks + IO ranks")
    print(" --ntasks-per-node: ceil((compute ranks + IO ranks) / nodes)")
    print("--------------------------------------------------------------------")
    print(" Local GPU Selection (GPU presets only)")
    print("--------------------------------------------------------------------")
    print(" Set VVM_GPU_LIST before a local GPU run to choose physical devices:")
    print('   VVM_GPU_LIST=0,1,2,3,4,5,6,7 ./submit.py --local -c "rundata/input_configs/default_cases/taiwanvvm_2048.json" --preset blaze --compute 8 --nodes 1')
    print(" Without it, GPU ranks map by local rank modulo GPUs/node.")
    print(" CPU presets ignore VVM_GPU_LIST.")
    print("--------------------------------------------------------------------")
    print(" Advanced SLURM Options: command-line only")
    print("--------------------------------------------------------------------")
    print(f" --export SPEC    : SLURM export option (default: {DEFAULT_EXPORT})")
    print(" --exclusive      : Request exclusive nodes. This is the default.")
    print(" --no-exclusive   : Do not request exclusive nodes.")
    print(" --exclude LIST   : Exclude specific nodes.")
    print("                    Example: --exclude 25a-hgpn062")
    print(" --nodelist LIST  : Request specific nodes.")
    print("                    Example: --nodelist 25a-hgpn001,25a-hgpn002")
    print(" --contiguous     : Request contiguous nodes.")
    print(" --slurm-arg ARG  : Append raw sbatch argument. Can be repeated.")
    print("                    Example: --slurm-arg '--qos=debug'")
    print("====================================================================")

    class Args:
        pass

    args = Args()

    args.local = ask("Run locally without SLURM? (y/N)", "N").upper() == "Y"
    args.config = ask_path("Configuration file", DEFAULT_CONFIG)

    if presets:
        print("\nAvailable CMake presets:")
        for i, p in enumerate(presets):
            print(f" [{i + 1}] {p}")

        preset_choice = ask(f"Select preset (1-{len(presets)} or type name)", "1")
        if preset_choice.isdigit() and 1 <= int(preset_choice) <= len(presets):
            args.preset = presets[int(preset_choice) - 1]
        else:
            args.preset = preset_choice
    else:
        args.preset = ask("CMake preset environment", default_preset)

    gpu_preset = preset_uses_gpu(vvm_root, args.preset)

    args.compute = int(ask("\nCompute tasks / MPI ranks", DEFAULT_COMPUTE))

    # Derived or prompted below as appropriate.
    args.io = DEFAULT_IO
    args.io_cpus = DEFAULT_IO_CPUS
    args.cpus = DEFAULT_CPUS
    args.omp_threads = DEFAULT_OMP_THREADS
    args.gpus = DEFAULT_GPUS

    if not args.local:
        args.nodes = int(ask("\nNumber of nodes", DEFAULT_NODES))
    else:
        args.nodes = 1

    io_engine = peek_io_engine(args.config)
    peeked_out_dir = peek_output_dir(args.config)
    args.io = infer_io_tasks(io_engine, args.compute, args.io)
    inferred_gpus = infer_gpus_per_node(args.compute, args.nodes, args.gpus) if gpu_preset else 0
    if args.local and not gpu_preset:
        logical_cpus = os.cpu_count() or 1
        compute_per_node = math.ceil(args.compute / args.nodes)
        io_per_node = math.ceil(args.io / args.nodes) if args.io > 0 else 0
        io_threads = io_per_node * args.io_cpus
        full_node_cpus = max(1, (logical_cpus - io_threads) // compute_per_node)
        print(f"\n[Info] Full-node CPU suggestion: --cpus {full_node_cpus}")
        print(f"       ({logical_cpus} logical CPUs - {io_per_node} IO ranks x {args.io_cpus}) / {compute_per_node} compute ranks")
        args.cpus = int(ask("CPUs/threads per compute rank (--cpus)", full_node_cpus))

    print("")
    backend_name = "GPU" if gpu_preset else "CPU"
    print(f"[Info] Selected backend: {backend_name} ({args.preset})")
    print(f"[Info] Detected output engine: {io_engine}")
    io_reason = "SST default: same as compute ranks" if io_engine == "SST" else "non-SST default: no IO servers"
    print(f"[Info] IO MPI ranks (--io): {args.io}  ({io_reason})")
    print(f"[Info] CPUs per IO rank (--io-cpus): {args.io_cpus}")
    print("       --io-cpus sizes each IO rank; it does not change the IO rank count.")
    if args.local:
        if args.cpus is None:
            print("[Info] CPUs per task (--cpus) default: 1 in local GPU mode.")
        else:
            print(f"[Info] CPUs per compute rank (--cpus): {args.cpus}  (wizard selection)")
    else:
        print("[Info] CPUs per task (--cpus) default: fill the SLURM node.")
        print("       The final value is queried from the selected partition before submit.")
    if gpu_preset:
        print(f"[Info] GPUs per node default: {inferred_gpus}")
        print("       Derived from compute ranks only; IO ranks add no GPU requests.")
        if args.local:
            gpu_list = os.environ.get("VVM_GPU_LIST", "")
            if gpu_list:
                print(f"       Current VVM_GPU_LIST={gpu_list}")
            else:
                print("       VVM_GPU_LIST is unset; ranks map modulo GPUs/node.")
    else:
        print("[Info] GPUs per node: none; CPU presets ignore VVM_GPU_LIST.")

    if not args.local:
        args.time = ask("\nTime limit", DEFAULT_TIME)
        args.job_name = ask("Job name", DEFAULT_JOB_NAME)
        args.account = ask("SLURM account", DEFAULT_ACCOUNT)
        args.partition = ask("SLURM partition", DEFAULT_PARTITION)
        args.out = ask("Standard output log", default_log_path(peeked_out_dir, DEFAULT_LOG_OUT_NAME, vvm_root))
        args.err = ask("Standard error log", default_log_path(peeked_out_dir, DEFAULT_LOG_ERR_NAME, vvm_root))
    else:
        args.time = None
        args.job_name = DEFAULT_JOB_NAME
        args.account = None
        args.partition = None
        args.out = DEFAULT_OUT
        args.err = DEFAULT_ERR

    # Advanced command-line-only options.
    args.exclude = None
    args.nodelist = None
    args.contiguous = False
    args.export = DEFAULT_EXPORT
    args.exclusive = DEFAULT_EXCLUSIVE
    args.slurm_arg = []

    cmd_parts = [sys.argv[0]]

    if args.local and gpu_preset and os.environ.get("VVM_GPU_LIST"):
        cmd_parts.insert(0, f'VVM_GPU_LIST={os.environ["VVM_GPU_LIST"]}')

    if args.local:
        cmd_parts.append("--local")

    cmd_parts.append(f'-c "{args.config}"')
    cmd_parts.append(f'--preset "{args.preset}"')
    cmd_parts.append(f"--compute {args.compute}")
    cmd_parts.append(f"--io {args.io}")
    cmd_parts.append(f"--io-cpus {args.io_cpus}")
    if args.cpus is not None:
        cmd_parts.append(f"--cpus {args.cpus}")
    if args.omp_threads is not None:
        cmd_parts.append(f"--omp-threads {args.omp_threads}")
    cmd_parts.append(f"--nodes {args.nodes}")

    if not args.local:
        cmd_parts.append(f'-t "{args.time}"')
        cmd_parts.append(f'--out "{args.out}"')
        cmd_parts.append(f'--err "{args.err}"')
        cmd_parts.append(f'--job-name "{args.job_name}"')
        cmd_parts.append(f'-A "{args.account}"')
        cmd_parts.append(f'-p "{args.partition}"')

    args.equivalent_command = " ".join(cmd_parts)

    print("\n--- Setup Complete ---")
    print("\nEquivalent command:\n")
    print(" " + args.equivalent_command)
    print("\nUse --help to see advanced command-line-only options.\n")

    return args


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VVM C++ job submission wrapper",
        epilog=(
            "If you do not know which inputs to provide, run ./submit.py with no "
            "arguments and follow the prompts. For local runs that need specific "
            "physical GPUs, prefix the command with VVM_GPU_LIST, for example: "
            "VVM_GPU_LIST=0,1,2,3 ./submit.py --local -c <config> --preset <preset> --compute 4 --nodes 1"
        ),
    )

    parser.add_argument("-c", "--config", help="Path to JSON configuration file")
    parser.add_argument("--preset", type=str, help="CMake preset name to load environment from")
    parser.add_argument("--local", action="store_true", help="Run locally without SLURM")

    parser.add_argument("--compute", type=int, default=DEFAULT_COMPUTE, help="Compute MPI ranks")
    parser.add_argument(
        "--io",
        type=int,
        default=DEFAULT_IO,
        help="IO MPI ranks. Default: same as compute for SST; 0 otherwise.",
    )
    parser.add_argument(
        "--io-cpus", type=int, default=DEFAULT_IO_CPUS,
        help="Host CPUs/threads reserved per IO rank (default 1). This does not create IO ranks; "
             "IO ranks are host-only and do not consume a GPU.")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES, help="Number of nodes")
    parser.add_argument(
        "--gpus",
        type=int,
        default=DEFAULT_GPUS,
        help="GPUs per node. Default: ceil(compute ranks / nodes).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=DEFAULT_CPUS,
        help="CPUs allocated per MPI task. OpenMP threads default to this unless --omp-threads is set.",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=DEFAULT_OMP_THREADS,
        help="OpenMP/Kokkos threads per compute rank. Defaults to --cpus; may be lower when the scheduler requires allocation padding.",
    )

    parser.add_argument("-t", "--time", type=str, default=DEFAULT_TIME, help="Wall time limit")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="Standard output log (default: <output.output_dir>/"
                             + DEFAULT_LOG_OUT_NAME.replace("%", "%%") + ")")
    parser.add_argument("--err", type=str, default=DEFAULT_ERR,
                        help="Standard error log (default: <output.output_dir>/"
                             + DEFAULT_LOG_ERR_NAME.replace("%", "%%") + ")")
    parser.add_argument("--job-name", type=str, default=DEFAULT_JOB_NAME, help="SLURM job name")
    parser.add_argument("-A", "--account", type=str, default=DEFAULT_ACCOUNT, help="SLURM account")
    parser.add_argument("-p", "--partition", type=str, default=DEFAULT_PARTITION, help="SLURM partition")

    # Advanced SLURM options. CLI only; not prompted in wizard.
    parser.add_argument("--exclude", type=str, default=None, help="SLURM --exclude node list")
    parser.add_argument("--nodelist", type=str, default=None, help="SLURM --nodelist node list")
    parser.add_argument("--contiguous", action="store_true", help="Request contiguous SLURM nodes")
    parser.add_argument(
        "--export",
        type=str,
        default=DEFAULT_EXPORT,
        help="SLURM --export value. Default: ALL.",
    )
    parser.add_argument(
        "--exclusive",
        dest="exclusive",
        action="store_true",
        help="Request exclusive node allocation. Default.",
    )
    parser.add_argument(
        "--no-exclusive",
        dest="exclusive",
        action="store_false",
        help="Do not request exclusive node allocation.",
    )
    parser.set_defaults(exclusive=DEFAULT_EXCLUSIVE)

    parser.add_argument(
        "--slurm-arg",
        action="append",
        default=[],
        help="Append a raw sbatch argument. Repeatable. Example: --slurm-arg='--qos=debug'",
    )

    if len(sys.argv) == 1:
        return interactive_wizard()

    args = parser.parse_args()

    if not args.config:
        print("[Error] --config is required in command-line mode.")
        sys.exit(1)

    if not args.preset:
        presets = get_available_presets(get_vvm_root())
        args.preset = presets[0] if presets else "unknown"

    return args


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    env = setup_environment(args.preset)

    vvm_root = env.get("VVM_ROOT")
    if not vvm_root:
        print("[Error] Failed to detect VVM_ROOT.")
        sys.exit(1)

    os.chdir(vvm_root)

    try:
        config_data, config_path_user = read_config(args.config)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    output_info = config_data.get("output", {})
    io_engine = output_info.get("engine", "HDF5")
    out_dir_raw = output_info.get("output_dir", "")

    if not out_dir_raw:
        print("[Error] output.output_dir missing in JSON.")
        sys.exit(1)

    if args.nodes <= 0:
        print("[Error] --nodes must be positive.")
        sys.exit(1)

    if args.compute <= 0:
        print("[Error] --compute must be positive.")
        sys.exit(1)

    if args.io is not None and args.io < 0:
        print("[Error] --io cannot be negative.")
        sys.exit(1)

    if args.cpus is not None and args.cpus <= 0:
        print("[Error] --cpus must be positive.")
        sys.exit(1)

    if args.omp_threads is not None and args.omp_threads <= 0:
        print("[Error] --omp-threads must be positive.")
        sys.exit(1)

    if args.io_cpus <= 0:
        print("[Error] --io-cpus must be positive.")
        sys.exit(1)

    # Derived defaults.
    args.io = infer_io_tasks(io_engine, args.compute, args.io)

    if not args.local and args.partition:
        requested_partition = args.partition
        args.partition, partition_note = resolve_partition(args.partition)
        if partition_note:
            level = "Info" if args.partition != requested_partition else "Warning"
            print(f"[{level}] {partition_note}")

    cpu_backend = env.get("VVM_BACKEND", "gpu") == "cpu"
    if cpu_backend:
        # A CPU build has no device to request or map. Requesting GPUs from SLURM
        # would also make the job queue for resources it will never touch.
        if args.gpus:
            print("[Info] CPU backend: ignoring --gpus.")
        args.gpus = 0
    else:
        args.gpus = infer_gpus_per_node(args.compute, args.nodes, args.gpus)

        if args.gpus <= 0:
            print("[Error] --gpus must be positive.")
            sys.exit(1)

    total_tasks = args.compute + args.io

    if io_engine == "SST" and args.io == 0:
        print("[Error] SST engine requires IO ranks. Use --io N or omit --io to default to compute ranks.")
        sys.exit(1)
    if io_engine != "SST" and args.io > 0:
        print(f"[Error] output.engine={io_engine!r} does not use IO-server ranks. Omit --io or use --io 0.")
        sys.exit(1)

    prof_file = config_data.get("initial_conditions", {}).get("source_file", "")
    spat_file = config_data.get("netcdf_reader", {}).get("source_file", "")

    out_dir_abs = os.path.abspath(out_dir_raw)
    prof_path = os.path.abspath(prof_file) if prof_file else ""
    spat_path = os.path.abspath(spat_file) if spat_file else ""

    os.makedirs(out_dir_abs, exist_ok=True)

    if not args.out:
        args.out = default_log_path(out_dir_abs, DEFAULT_LOG_OUT_NAME)

    if not args.err:
        args.err = default_log_path(out_dir_abs, DEFAULT_LOG_ERR_NAME)

    if args.out:
        out_log_dir = os.path.dirname(os.path.abspath(args.out))
        if out_log_dir:
            os.makedirs(out_log_dir, exist_ok=True)

    if args.err:
        err_log_dir = os.path.dirname(os.path.abspath(args.err))
        if err_log_dir:
            os.makedirs(err_log_dir, exist_ok=True)

    write_submit_command(out_dir_abs, vvm_root, args)

    snapshot_dir = os.path.join(out_dir_abs, "code_snapshot")
    create_code_snapshot(
        vvm_root,
        snapshot_dir,
        config_path_user,
        prof_path,
        spat_path,
        out_dir_raw,
    )

    compute_per_node = math.ceil(args.compute / args.nodes)
    io_per_node = math.ceil(args.io / args.nodes) if args.io > 0 else 0

    # Important:
    # - total tasks include compute + IO ranks
    # - GPU request is based only on compute ranks
    tasks_per_node = math.ceil(total_tasks / args.nodes)

    # Needs tasks_per_node, so it cannot sit with the other derived defaults.
    args.cpus, cpus_origin, cpus_warnings = infer_cpus_per_task(
        args.cpus,
        None if args.local else args.partition,
        tasks_per_node,
        args.gpus,
        compute_per_node,
        io_per_node,
        args.io_cpus,
    )
    if cpus_origin != "explicit":
        print(f"[Info] CPUs per task: {args.cpus}  ({cpus_origin})")
    for warning in cpus_warnings:
        print(f"[Warning] {warning}")

    env["VVM_CONFIG_FILE"] = config_path_user
    env["VVM_COMPUTE_TASKS"] = str(args.compute)
    env["VVM_IO_TASKS"] = str(args.io)
    env["VVM_TOTAL_TASKS"] = str(total_tasks)
    env["VVM_COMPUTE_PER_NODE"] = str(compute_per_node)
    env["VVM_IO_PER_NODE"] = str(io_per_node)
    env["VVM_IO_ENGINE"] = io_engine
    env["VVM_OUTPUT_DIR"] = out_dir_abs
    env["OMP_NUM_THREADS"] = str(args.cpus)
    if args.omp_threads is not None:
        env["VVM_OMP_THREADS"] = str(args.omp_threads)
    env["VVM_GPUS"] = str(args.gpus)
    env["VVM_IO_CPUS"] = str(args.io_cpus)

    script_path = os.path.join(vvm_root, "tools", "core_run.sh")
    if not os.path.isfile(script_path):
        print(f"[Error] Core script missing at {script_path}.")
        sys.exit(1)

    print("\n=========================================")
    print(" VVM SUBMISSION SUMMARY")
    print("=========================================")
    print(f" Mode              : {'LOCAL' if args.local else 'SLURM'}")
    print(f" Config            : {config_path_user}")
    print(f" Preset            : {args.preset}")
    print(f" Output engine     : {io_engine}")
    if io_engine == "BP5":
        print(" IO path           : direct compute-rank BP5 (no IO-server ranks)")
    print(f" Compute ranks     : {args.compute}")
    print(f" IO ranks          : {args.io}")
    print(f" Total ranks       : {total_tasks}")
    print(f" Nodes             : {args.nodes}")
    print(f" Compute/node      : {compute_per_node}")
    print(f" IO/node           : {io_per_node}")
    print(f" Total tasks/node  : {tasks_per_node}")
    print(f" Backend           : {env.get('VVM_BACKEND', 'gpu')}")
    print(f" Binary            : {env.get('VVM_BINARY', '<default>')}")
    if cpu_backend:
        print(" GPUs/node         : none (CPU build)")
    else:
        print(f" GPUs/node         : {args.gpus}")
        if args.local:
            gpu_list = env.get("VVM_GPU_LIST", "")
            if gpu_list:
                print(f" Local GPU list    : {gpu_list}")
            else:
                print(" Local GPU list    : <unset; ranks map by local rank modulo GPUs/node>")
                print("                     Set VVM_GPU_LIST=0,1,... before ./submit.py --local to choose GPU IDs.")
    print(f" CPUs/compute rank : {args.cpus}")
    print(f" OpenMP threads/compute rank: {args.omp_threads or args.cpus}")
    print(f" Active compute cores: {args.compute * (args.omp_threads or args.cpus)}")
    if args.io > 0:
        print(f" CPUs/IO rank      : {args.io_cpus}  (host-only, no GPU)")
    if not args.local:
        print(f" Stdout log        : {args.out}")
        print(f" Stderr log        : {args.err}")
        print(f" Exclusive         : {args.exclusive}")
        print(f" Export            : {args.export}")
        if args.exclude:
            print(f" Exclude           : {args.exclude}")
        if args.nodelist:
            print(f" Nodelist          : {args.nodelist}")
        if args.contiguous:
            print(" Contiguous        : True")
        if args.slurm_arg:
            print(f" Extra sbatch args : {' '.join(args.slurm_arg)}")
    print("=========================================\n")

    if args.local:
        cmd = ["bash", script_path]

    else:
        cmd = [
            "sbatch",
            f"--job-name={args.job_name}",
            f"--nodes={args.nodes}",
            f"--ntasks={total_tasks}",
            f"--ntasks-per-node={tasks_per_node}",
        ]

        if not cpu_backend:
            cmd += [
                f"--gpus-per-node={args.gpus}",
                "--gpu-bind=none",
            ]

        cmd += [
            f"--cpus-per-task={args.cpus}",
            f"--export={args.export}",
            f"--time={args.time}",
            f"--output={os.path.abspath(args.out)}",
            f"--error={os.path.abspath(args.err)}",
        ]

        if args.exclusive:
            cmd.append("--exclusive")

        if args.account:
            cmd.append(f"--account={args.account}")

        if args.partition:
            cmd.append(f"--partition={args.partition}")

        if args.exclude:
            cmd.append(f"--exclude={args.exclude}")

        if args.nodelist:
            cmd.append(f"--nodelist={args.nodelist}")

        if args.contiguous:
            cmd.append("--contiguous")

        for extra_arg in args.slurm_arg:
            if extra_arg:
                cmd.append(extra_arg)

        cmd.append(script_path)

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Process failed with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
