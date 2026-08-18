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
  --no-service         configure but do not install the systemd service
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
[ -n "$RUNNER_NAME" ] || RUNNER_NAME="$(hostname -s)-$PRESET"

LABELS="vvmex,vvmex-$BACKEND,preset-$PRESET,$(hostname -s)"

echo "==> preset      : $PRESET ($BACKEND backend)"
echo "==> runner dir  : $RUNNER_DIR"
echo "==> work dir    : $WORK_DIR"
echo "==> runner name : $RUNNER_NAME"
echo "==> labels      : $LABELS"
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

if [ "$INSTALL_SERVICE" -eq 1 ]; then
    echo "==> installing service as $(id -un)"
    (cd "$RUNNER_DIR" && sudo ./svc.sh install "$(id -un)" && sudo ./svc.sh start)
    echo "==> running"
else
    echo "==> not installing a service (--no-service). Start it manually with:"
    echo "      cd $RUNNER_DIR && ./run.sh"
fi

cat <<DONE

Done. This machine now accepts jobs labelled: vvmex-$BACKEND

Remaining, once per repository (not per machine):
  1. Set repository variable  VVMEX_RUNNER = true
  2. Add the required status checks under Settings -> Branches

Re-run with --env-only after moving the TPLs or editing CMakePresets.json.
DONE
