#!/usr/bin/env bash
# Register this machine as a GitHub Actions runner for VVMex.
#
# Everything machine-specific comes from one CMake preset: which TPLs to load,
# where the build tree goes, whether this is a CPU or GPU host, and the labels
# the workflows select on. Adding a second or third development machine should
# therefore be this one command with a different --preset.
#
#   tools/setup_ci_runner.sh --preset blaze-cpu --token <registration-token>
#
# The token comes from Settings -> Actions -> Runners -> New self-hosted runner
# on the repository, and expires after an hour.
set -euo pipefail

# config.sh and svc.sh are .NET; see the note in ci_runner_env.py. This shell
# has the scientific stack on LD_LIBRARY_PATH, so without this the very first
# ./config.sh dies on an ICU version mismatch.
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

RUNNER_VERSION="2.330.0"
# Empty means "ask the checkout". A hardcoded URL goes stale the moment the
# repository is renamed, which is exactly what happened to the previous one.
REPO_URL=""
PRESET=""
TOKEN=""
RUNNER_DIR=""
WORK_DIR=""
RUNNER_NAME=""
INSTALL_SERVICE=1
USER_SERVICE=0
BUILD_JOBS=""
TEST_THREADS=""

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --preset <name>      CMake preset for this machine (required)
  --token <token>      runner registration token (required unless --env-only)
  --url <url>          repository URL (default: the VVMex repository)
  --dir <path>         where to install the runner (default: <parent of VVM_ROOT>/actions-runner)
  --work <path>        runner work directory (default: <dir>/_work)
  --name <name>        runner name shown on GitHub (default: <hostname>-<preset>)
  --build-jobs <n>     compile parallelism (default: all cores)
  --test-threads <n>   OMP threads per rank for CPU tests (default: cores/4)
  --runner-version <v> actions/runner release (default: RUNNER_VERSION above)
  --user-service       install a systemd --user unit instead of a system one
                       (no root needed; use this when you have no sudo)
  --no-service         configure but do not install any service
  --env-only           only regenerate .env in an existing runner directory
USAGE
}

ENV_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --preset)         PRESET="$2"; shift 2 ;;
        --token)          TOKEN="$2"; shift 2 ;;
        --url)            REPO_URL="$2"; shift 2 ;;
        --dir)            RUNNER_DIR="$2"; shift 2 ;;
        --work)           WORK_DIR="$2"; shift 2 ;;
        --name)           RUNNER_NAME="$2"; shift 2 ;;
        --build-jobs)     BUILD_JOBS="$2"; shift 2 ;;
        --test-threads)   TEST_THREADS="$2"; shift 2 ;;
        --runner-version) RUNNER_VERSION="$2"; shift 2 ;;
        --no-service)     INSTALL_SERVICE=0; shift ;;
        --user-service)   USER_SERVICE=1; shift ;;
        --env-only)       ENV_ONLY=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

: "${VVM_ROOT:?export VVM_ROOT=/path/to/VVMex first}"
[ -f "$VVM_ROOT/CMakePresets.json" ] || { echo "VVM_ROOT does not contain CMakePresets.json" >&2; exit 2; }
[ -n "$PRESET" ] || { echo "--preset is required" >&2; usage >&2; exit 2; }

# config.sh wants the web URL. Accept whichever form the checkout uses --
# git@github.com:owner/repo.git and https://github.com/owner/repo.git both
# normalise to https://github.com/owner/repo.
if [ -z "$REPO_URL" ]; then
    ORIGIN="$(git -C "$VVM_ROOT" remote get-url origin 2>/dev/null || true)"
    [ -n "$ORIGIN" ] || { echo "no 'origin' remote in $VVM_ROOT; pass --url" >&2; exit 2; }
    case "$ORIGIN" in
        git@*:*) _rest="${ORIGIN#git@}"
                 # Split host from path on the FIRST colon, before adding the
                 # scheme -- substituting afterwards hits the one in "https:".
                 REPO_URL="https://${_rest%%:*}/${_rest#*:}" ;;
        ssh://git@*) REPO_URL="https://${ORIGIN#ssh://git@}" ;;
        *)       REPO_URL="$ORIGIN" ;;
    esac
    REPO_URL="${REPO_URL%.git}"
    echo "==> repository  : $REPO_URL (from origin)"
fi

# Fail early and by name if the preset is not in this checkout, rather than
# after downloading 200 MB of runner.
python3 - "$VVM_ROOT" "$PRESET" <<'PY'
import json, sys
root, preset = sys.argv[1], sys.argv[2]
with open(f"{root}/CMakePresets.json") as f:
    names = [p["name"] for p in json.load(f).get("configurePresets", [])]
if preset not in names:
    sys.exit(f"error: preset '{preset}' not found. Available: {', '.join(names)}")
PY

BACKEND="$(python3 - "$VVM_ROOT" "$PRESET" <<'PY'
import json, sys
root, preset = sys.argv[1], sys.argv[2]
with open(f"{root}/CMakePresets.json") as f:
    for p in json.load(f).get("configurePresets", []):
        if p["name"] == preset:
            gpu = str(p.get("cacheVariables", {}).get("VVM_ENABLE_GPU", "ON")).upper()
            print("cpu" if gpu == "OFF" else "gpu")
            break
PY
)"

[ -n "$RUNNER_DIR" ]  || RUNNER_DIR="$(dirname "$VVM_ROOT")/actions-runner-$PRESET"
[ -n "$WORK_DIR" ]    || WORK_DIR="$RUNNER_DIR/_work"
# <host>-<preset> reads badly when the preset is already named after the host:
# blaze + blaze-cpu became "blaze-blaze-cpu". Use the preset alone in that case.
if [ -z "$RUNNER_NAME" ]; then
    _host="$(hostname -s)"
    case "$PRESET" in
        "$_host"|"$_host"-*) RUNNER_NAME="$PRESET" ;;
        *)                   RUNNER_NAME="$_host-$PRESET" ;;
    esac
fi

LABELS="vvmex,vvmex-$BACKEND,preset-$PRESET,$(hostname -s)"

echo "==> preset      : $PRESET ($BACKEND backend)"
echo "==> runner dir  : $RUNNER_DIR"
echo "==> work dir    : $WORK_DIR"
echo "==> runner name : $RUNNER_NAME"
echo "==> labels      : $LABELS"

# The work directory holds a full checkout plus a build tree per run: ~300 MB
# for a CPU build, ~9 GB for a GPU one. Warn rather than fail -- only the
# operator knows what else lives on that filesystem -- but warn loudly, because
# filling the root filesystem takes down more than CI.
_parent="$(dirname "$RUNNER_DIR")"
mkdir -p "$_parent"
_avail_gb=$(( $(df -Pk "$_parent" | awk 'NR==2{print $4}') / 1024 / 1024 ))
_want_gb=$([ "$BACKEND" = gpu ] && echo 25 || echo 10)
echo "==> free space  : ${_avail_gb} GB on $(df -Ph "$_parent" | awk 'NR==2{print $6}')"
if [ "$_avail_gb" -lt "$_want_gb" ]; then
    echo
    echo "WARNING: a $BACKEND runner wants roughly ${_want_gb} GB for checkouts and build"
    echo "         trees, and only ${_avail_gb} GB is free here. Put it on a bigger"
    echo "         filesystem instead:"
    echo "           --dir /path/with/space/actions-runner-$PRESET"
    echo
    printf "Continue anyway? [y/N] "
    read -r _reply
    case "$_reply" in [yY]*) ;; *) echo "aborted."; exit 1 ;; esac
fi
echo

write_env() {
    local args=(--preset "$PRESET")
    [ -n "$BUILD_JOBS" ]   && args+=(--build-jobs "$BUILD_JOBS")
    [ -n "$TEST_THREADS" ] && args+=(--test-threads "$TEST_THREADS")
    python3 "$VVM_ROOT/tools/ci_runner_env.py" "${args[@]}" > "$RUNNER_DIR/.env"
    echo "==> wrote $RUNNER_DIR/.env"
}

if [ "$ENV_ONLY" -eq 1 ]; then
    [ -d "$RUNNER_DIR" ] || { echo "no runner at $RUNNER_DIR" >&2; exit 2; }
    write_env
    echo "Restart the runner for the new environment to take effect:"
    echo "  cd $RUNNER_DIR && sudo ./svc.sh stop && sudo ./svc.sh start"
    exit 0
fi

[ -n "$TOKEN" ] || { echo "--token is required (Settings -> Actions -> Runners -> New self-hosted runner)" >&2; exit 2; }

mkdir -p "$RUNNER_DIR" "$WORK_DIR"

if [ ! -x "$RUNNER_DIR/config.sh" ]; then
    TARBALL="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
    echo "==> downloading actions/runner $RUNNER_VERSION"
    curl -fsSL -o "$RUNNER_DIR/$TARBALL" \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${TARBALL}"
    tar xzf "$RUNNER_DIR/$TARBALL" -C "$RUNNER_DIR"
    rm -f "$RUNNER_DIR/$TARBALL"
else
    echo "==> runner already unpacked, reusing it"
fi

echo "==> configuring"
(
    cd "$RUNNER_DIR"
    ./config.sh --unattended --replace \
        --url "$REPO_URL" --token "$TOKEN" \
        --name "$RUNNER_NAME" --labels "$LABELS" --work "$WORK_DIR"
)

write_env

if [ "$INSTALL_SERVICE" -eq 1 ] && [ "$USER_SERVICE" -eq 1 ]; then
    # No root required. `loginctl enable-linger` is what keeps a --user unit
    # running after the last session ends and starts it again at boot; on many
    # systems a user may enable it for themselves.
    UNIT="vvmex-runner-$PRESET"
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$HOME/.config/systemd/user/$UNIT.service" <<UNITFILE
[Unit]
Description=GitHub Actions runner for VVMex ($PRESET)
After=network-online.target

[Service]
ExecStart=$RUNNER_DIR/run.sh
WorkingDirectory=$RUNNER_DIR
Restart=always
RestartSec=10
# Leave KillMode at its default (control-group) so SIGTERM reaches
# Runner.Listener itself. KillMode=process -- which GitHub's own template uses
# -- signals only the ExecStart process, and that is run.sh, which does not
# forward signals to its child. The listener then survives a restart, keeps the
# GitHub session, and the replacement loops forever on "A session for this
# runner already exists". GitHub gets away with it because svc.sh installs a
# runsvc.sh wrapper that does forward; we start run.sh directly.
# The cost is that a restart also stops a running job, which is the right
# trade: jobs are re-runnable, orphaned listeners are not self-healing.
KillSignal=SIGTERM
TimeoutStopSec=3min

[Install]
WantedBy=default.target
UNITFILE
    loginctl enable-linger "$(id -un)" 2>/dev/null || \
        echo "WARNING: could not enable linger; the runner will stop when you log out."
    systemctl --user daemon-reload
    systemctl --user enable --now "$UNIT.service"
    echo "==> running as a user service: $UNIT"
    echo "    status:  systemctl --user status $UNIT"
    echo "    logs:    journalctl --user -u $UNIT -f"
    echo "    stop:    systemctl --user stop $UNIT"
elif [ "$INSTALL_SERVICE" -eq 1 ]; then
    echo "==> installing system service as $(id -un) (needs sudo)"
    if ! sudo -n true 2>/dev/null && ! sudo -v 2>/dev/null; then
        echo
        echo "ERROR: no usable sudo. Re-run with --user-service to install a" >&2
        echo "       systemd --user unit instead, which needs no root." >&2
        exit 1
    fi
    (cd "$RUNNER_DIR" && sudo ./svc.sh install "$(id -un)" && sudo ./svc.sh start)
    echo "==> running"
else
    echo "==> not installing a service (--no-service). Start it manually with:"
    echo "      cd $RUNNER_DIR && ./run.sh"
fi

# Which variable this backend's workflows read. Printing VVMEX_RUNNER for a GPU
# runner would be actively misleading: the operator sets the one variable they
# were told about, gpu-tests.yml stays inert, and nothing says why.
if [ "$BACKEND" = gpu ]; then
    VAR_NAME="VVMEX_GPU_RUNNER"
    VAR_GATES="gpu-tests.yml and the nightly GPU job"
    CHECK_NAME="Build and test (GPU)"
else
    VAR_NAME="VVMEX_RUNNER"
    VAR_GATES="cpu-tests.yml and the nightly CPU job"
    CHECK_NAME="Build and test (CPU)"
fi

cat <<DONE

Done. This machine now accepts jobs labelled: vvmex-$BACKEND

Remaining, once per repository (not per machine):
  1. Set repository variable  $VAR_NAME = true
     Settings -> Secrets and variables -> Actions -> Variables
     It gates $VAR_GATES. Each backend has its own
     variable, because a job dispatched to a label no runner carries does not
     skip -- it queues for 24 hours and then fails.
  2. Add "$CHECK_NAME" to the required status checks
     under Settings -> Branches, once it has run green at least once.

Re-run with --env-only after moving the TPLs or editing CMakePresets.json.
DONE
