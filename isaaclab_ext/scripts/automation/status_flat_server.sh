#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
AUTOMATION_ROOT="${PROJECT_DIR}/logs/automation/flat_scratch_v1"
CURRENT_SESSION_FILE="${AUTOMATION_ROOT}/current_session.txt"

if [[ ! -f "${CURRENT_SESSION_FILE}" ]]; then
    echo "[ERROR] No flat training session has been launched." >&2
    exit 1
fi

SESSION_DIR="$(<"${CURRENT_SESSION_FILE}")"
PID_FILE="${SESSION_DIR}/train.pid"
RUN_FILE="${SESSION_DIR}/run_dir.txt"

echo "Session: ${SESSION_DIR}"
if [[ -f "${PID_FILE}" ]]; then
    TRAIN_PID="$(<"${PID_FILE}")"
    if [[ "${TRAIN_PID}" =~ ^[0-9]+$ ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "Process: running (PID ${TRAIN_PID})"
    else
        echo "Process: stopped (last PID ${TRAIN_PID})"
    fi
else
    echo "Process: PID file missing"
fi

if [[ -f "${RUN_FILE}" ]]; then
    RUN_DIR="$(<"${RUN_FILE}")"
    echo "Run: ${RUN_DIR}"
    latest_checkpoint="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)"
    echo "Latest checkpoint: ${latest_checkpoint:-none}"
fi

echo
echo "Latest iteration:"
grep 'Learning iteration' "${SESSION_DIR}/train_stdout.log" | tail -n 1 || true
echo
echo "Latest core metrics:"
grep -E 'Mean action noise std:|Mean reward:|Mean episode length:|Metrics/motion/error_forward_velocity:|Episode_Termination/time_out:|Episode_Termination/bad_orientation:|Episode_Termination/non_finite:' \
    "${SESSION_DIR}/train_stdout.log" | tail -n 7 || true
echo
echo "Errors:"
tail -n 20 "${SESSION_DIR}/train_stderr.log" || true
