#!/usr/bin/env bash
set -Eeuo pipefail

# Run the formal Flat-Compatible-v1 training in the foreground. The script is
# intended to be executed inside tmux so train.py retains direct terminal IO.
# It deliberately starts from random network weights and stops after flat
# training, leaving terrain training for a separate reviewed stage.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
ASSET_FILE="${WORKSPACE_DIR}/assets/infantry_2027_v0/isaac/infantry_2027_v0.usdc"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python executable is unavailable: ${PYTHON_BIN}"
[[ -f "${ASSET_FILE}" ]] || fail "Isaac asset is missing: ${ASSET_FILE}"
[[ -f "${PROJECT_DIR}/scripts/rsl_rl/train.py" ]] || fail "train.py is missing"

TRAIN_PATTERN='[p]ython.*scripts/rsl_rl/train.py.*Infantry-2027-Flat-Compatible-v1'
if pgrep -f "${TRAIN_PATTERN}" >/dev/null 2>&1; then
    echo "[ERROR] Another Flat-Compatible-v1 training process is already running:" >&2
    pgrep -af "${TRAIN_PATTERN}" >&2 || true
    exit 1
fi

cd "${WORKSPACE_DIR}"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    fail "Git working tree is dirty. Commit and synchronize code before formal training."
fi

GIT_COMMIT="$(git rev-parse HEAD)"
STAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
RUN_NAME="flat_compatible_v1_4096x2000_server_scratch_${STAMP}"

echo "[INFO] Project       : ${PROJECT_DIR}"
echo "[INFO] Asset         : ${ASSET_FILE}"
echo "[INFO] Git commit    : ${GIT_COMMIT}"
echo "[INFO] Python        : ${PYTHON_BIN}"
echo "[INFO] Environments  : ${NUM_ENVS}"
echo "[INFO] Target updates: ${MAX_ITERATIONS}"
echo "[INFO] Run name      : ${RUN_NAME}"
"${PYTHON_BIN}" -c \
    "import importlib.metadata as m; print('[INFO] Isaac Sim     :', m.version('isaacsim')); print('[INFO] Isaac Lab     :', m.version('isaaclab')); print('[INFO] PyTorch       :', m.version('torch')); print('[INFO] RSL-RL        :', m.version('rsl-rl-lib'))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "[INFO] Starting train.py in the foreground."
echo "[INFO] Detach tmux with Ctrl-b d; attach with: tmux attach -t flat_v1"

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" -u scripts/rsl_rl/train.py \
    --task Infantry-2027-Flat-Compatible-v1 \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless \
    --run_name "${RUN_NAME}"
