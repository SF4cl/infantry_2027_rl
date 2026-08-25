#!/usr/bin/env bash
set -Eeuo pipefail

# Launch the formal Flat-Compatible-v1 run from random network weights.
# The launcher exits after Isaac Lab creates the run directory; training keeps
# running under nohup.  It deliberately does not start terrain training so the
# flat result can be reviewed first.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
RUN_ROOT="${PROJECT_DIR}/logs/rsl_rl/infantry_2027_v1_flat_compatible"
AUTOMATION_ROOT="${PROJECT_DIR}/logs/automation/flat_scratch_v1"
CURRENT_SESSION_FILE="${AUTOMATION_ROOT}/current_session.txt"
ASSET_FILE="${WORKSPACE_DIR}/assets/infantry_2027_v0/isaac/infantry_2027_v0.usdc"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python executable is unavailable: ${PYTHON_BIN}"
[[ -f "${ASSET_FILE}" ]] || fail "Isaac asset is missing: ${ASSET_FILE}"
[[ -f "${PROJECT_DIR}/scripts/rsl_rl/train.py" ]] || fail "train.py is missing"

if [[ -f "${CURRENT_SESSION_FILE}" ]]; then
    previous_session="$(<"${CURRENT_SESSION_FILE}")"
    previous_pid_file="${previous_session}/train.pid"
    if [[ -f "${previous_pid_file}" ]]; then
        previous_pid="$(<"${previous_pid_file}")"
        if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
            fail "A flat training process is already alive: PID ${previous_pid} (${previous_session})"
        fi
    fi
fi

cd "${WORKSPACE_DIR}"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    fail "Git working tree is dirty. Commit and synchronize code before formal training."
fi
GIT_COMMIT="$(git rev-parse HEAD)"

mkdir -p "${RUN_ROOT}" "${AUTOMATION_ROOT}"
STAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
SESSION_DIR="${AUTOMATION_ROOT}/${STAMP}"
RUN_NAME="flat_compatible_v1_4096x2000_server_scratch_${STAMP}"
mkdir -p "${SESSION_DIR}"

printf '%s\n' "${SESSION_DIR}" > "${CURRENT_SESSION_FILE}"
printf '%s\n' "${GIT_COMMIT}" > "${SESSION_DIR}/git_commit.txt"
printf '%s\n' "${RUN_NAME}" > "${SESSION_DIR}/run_name.txt"
printf '%s\n' "${PYTHON_BIN}" > "${SESSION_DIR}/python.txt"

echo "[INFO] Project       : ${PROJECT_DIR}"
echo "[INFO] Asset         : ${ASSET_FILE}"
echo "[INFO] Git commit    : ${GIT_COMMIT}"
echo "[INFO] Python        : ${PYTHON_BIN}"
echo "[INFO] Environments  : ${NUM_ENVS}"
echo "[INFO] Target updates: ${MAX_ITERATIONS}"
"${PYTHON_BIN}" -c \
    "import importlib.metadata as m; print('[INFO] Isaac Sim     :', m.version('isaacsim')); print('[INFO] Isaac Lab     :', m.version('isaaclab')); print('[INFO] PyTorch       :', m.version('torch')); print('[INFO] RSL-RL        :', m.version('rsl-rl-lib'))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${PROJECT_DIR}"
nohup "${PYTHON_BIN}" -u scripts/rsl_rl/train.py \
    --task Infantry-2027-Flat-Compatible-v1 \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless \
    --run_name "${RUN_NAME}" \
    > "${SESSION_DIR}/train_stdout.log" \
    2> "${SESSION_DIR}/train_stderr.log" \
    < /dev/null &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${SESSION_DIR}/train.pid"

echo "[INFO] Training PID  : ${TRAIN_PID}"
echo "[INFO] Session       : ${SESSION_DIR}"

deadline=$((SECONDS + 180))
RUN_DIR=""
while (( SECONDS < deadline )); do
    RUN_DIR="$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -print | sort | tail -n 1)"
    if [[ -n "${RUN_DIR}" ]]; then
        break
    fi
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "[ERROR] Training exited before creating its run directory." >&2
        tail -n 100 "${SESSION_DIR}/train_stderr.log" >&2 || true
        exit 2
    fi
    sleep 3
done

if [[ -z "${RUN_DIR}" ]]; then
    echo "[WARN] Training is alive but the run directory was not found within 180 seconds." >&2
    echo "[WARN] Inspect ${SESSION_DIR}/train_stdout.log and train_stderr.log." >&2
    exit 3
fi

RUN_DIR="$(realpath "${RUN_DIR}")"
printf '%s\n' "${RUN_DIR}" > "${SESSION_DIR}/run_dir.txt"

echo "[OK] Flat training started from scratch."
echo "[OK] Run directory  : ${RUN_DIR}"
echo "[OK] Live log       : tail -f ${SESSION_DIR}/train_stdout.log"
echo "[OK] Status command : bash scripts/automation/status_flat_server.sh"
