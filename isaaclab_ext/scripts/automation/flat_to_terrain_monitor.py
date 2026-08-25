"""Wait for the v1 flat run, gate it, then launch and monitor terrain training.

This helper is intentionally independent of the Isaac Sim process.  It writes
an atomic JSON status file so progress survives terminal/session disconnects.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROCESS_SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INFINITE = 0xFFFFFFFF


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def wait_for_process(pid: int) -> None:
    """Wait for a process that is not necessarily a child of this monitor."""
    if pid <= 0:
        raise ValueError(f"Process id must be positive, got {pid}")

    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            # The training process may already have exited before the monitor
            # finished starting.  Continue to checkpoint validation in that case.
            return
        try:
            kernel32.WaitForSingleObject(handle, INFINITE)
        finally:
            kernel32.CloseHandle(handle)
        return

    # On POSIX the training process is normally launched by a shell and is not
    # a child of this monitor, so waitpid() cannot be used.  Signal 0 performs a
    # read-only existence check and does not deliver a signal to the process.
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            # It exists but belongs to another user.  Keep waiting; validation
            # will still be driven by the final checkpoint after it exits.
            pass
        time.sleep(2.0)


def scalar_window(run_dir: Path, window: int = 20) -> dict[str, float]:
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    result: dict[str, float] = {}
    for tag in tags:
        values = accumulator.Scalars(tag)[-window:]
        if values:
            result[tag] = sum(value.value for value in values) / len(values)
            result[f"{tag}__last_step"] = float(values[-1].step)
            result[f"{tag}__last"] = float(values[-1].value)
    return result


def conditional_mae(metrics: dict[str, float], prefix: str) -> float:
    numerator = metrics[f"Metrics/motion/{prefix}_forward_error_sum"]
    denominator = metrics[f"Metrics/motion/{prefix}_fraction"]
    return numerator / max(denominator, 1.0e-8)


def validate_checkpoint(run_dir: Path, iteration: int, apply_flat_gate: bool) -> dict:
    checkpoint = run_dir / f"model_{iteration}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed = int(payload.get("completed_iterations", payload.get("iter", -1)))
    if completed != iteration:
        raise RuntimeError(
            f"Checkpoint iteration mismatch: completed={completed}, expected={iteration}"
        )
    required = (
        "model_state_dict",
        "optimizer_state_dict",
        "estimator_optimizer_state_dict",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise RuntimeError(f"Checkpoint is missing training state: {missing}")

    metrics = scalar_window(run_dir)
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "completed_iterations": completed,
        "mean_episode_length_20": metrics.get("Train/mean_episode_length", math.nan),
        "mean_reward_20": metrics.get("Train/mean_reward", math.nan),
        "non_finite_20": metrics.get("Episode_Termination/non_finite", math.nan),
        "estimator_loss_20": metrics.get("Loss/estimator", math.nan),
    }
    for prefix in ("positive", "negative", "standing"):
        report[f"{prefix}_forward_mae_20_mps"] = conditional_mae(metrics, prefix)

    if apply_flat_gate:
        finite = all(math.isfinite(value) for value in report.values() if isinstance(value, float))
        asymmetry = abs(
            report["positive_forward_mae_20_mps"]
            - report["negative_forward_mae_20_mps"]
        )
        checks = {
            "all_metrics_finite": finite,
            "mean_episode_length_at_least_1200": report["mean_episode_length_20"] >= 1200.0,
            "non_finite_zero": report["non_finite_20"] <= 1.0e-8,
            "estimator_loss_below_2": report["estimator_loss_20"] < 2.0,
            "positive_mae_below_1": report["positive_forward_mae_20_mps"] < 1.0,
            "negative_mae_below_1": report["negative_forward_mae_20_mps"] < 1.0,
            "standing_mae_below_0_4": report["standing_forward_mae_20_mps"] < 0.4,
            "direction_asymmetry_below_0_4": asymmetry < 0.4,
        }
        report["gate_checks"] = checks
        report["gate_passed"] = all(checks.values())
    return report


def find_new_run(root: Path, suffix: str, started_at: float, timeout_s: float = 180.0) -> Path:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        candidates = [
            path
            for path in root.glob(f"*_{suffix}")
            if path.is_dir() and path.stat().st_mtime >= started_at - 5.0
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
        time.sleep(2.0)
    raise TimeoutError(f"Terrain run directory was not created below {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-pid", type=int, required=True)
    parser.add_argument("--flat-run", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--flat-iterations", type=int, default=2000)
    parser.add_argument("--terrain-iterations", type=int, default=5000)
    parser.add_argument("--terrain-envs", type=int, default=1024)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    flat_run = args.flat_run.resolve()
    status_path = args.status.resolve()
    status = {
        "phase": "waiting_for_flat",
        "updated_at": now(),
        "flat_pid": args.flat_pid,
        "flat_run": str(flat_run),
    }
    atomic_json(status_path, status)

    try:
        wait_for_process(args.flat_pid)
        status.update(phase="validating_flat", updated_at=now())
        atomic_json(status_path, status)
        flat_report = validate_checkpoint(flat_run, args.flat_iterations, apply_flat_gate=True)
        status["flat_validation"] = flat_report
        if not flat_report["gate_passed"]:
            status.update(phase="flat_gate_failed", updated_at=now())
            atomic_json(status_path, status)
            return 2

        terrain_root = project / "logs" / "rsl_rl" / "infantry_2027_v1_terrain"
        terrain_root.mkdir(parents=True, exist_ok=True)
        automation_dir = status_path.parent
        terrain_stdout = automation_dir / "terrain_v1_from_flat_stdout.log"
        suffix = "terrain_v1_from_flat_2000"
        command = [
            str(args.python.resolve()),
            "scripts/rsl_rl/train.py",
            "--task",
            "Infantry-2027-Terrain-v1",
            "--num_envs",
            str(args.terrain_envs),
            "--max_iterations",
            str(args.terrain_iterations),
            "--headless",
            "--run_name",
            suffix,
            "--resume_path",
            flat_report["checkpoint"],
        ]
        started_at = time.time()
        output = terrain_stdout.open("w", encoding="utf-8", buffering=1)
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        terrain_process = subprocess.Popen(
            command,
            cwd=project,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        terrain_run = find_new_run(terrain_root, suffix, started_at)
        status.update(
            phase="terrain_training",
            updated_at=now(),
            terrain_pid=terrain_process.pid,
            terrain_run=str(terrain_run.resolve()),
            terrain_stdout=str(terrain_stdout.resolve()),
            terrain_target_iterations=args.terrain_iterations,
        )
        atomic_json(status_path, status)
        return_code = terrain_process.wait()
        output.close()
        status.update(terrain_return_code=return_code, updated_at=now())
        if return_code != 0:
            status["phase"] = "terrain_process_failed"
            atomic_json(status_path, status)
            return return_code or 3

        status["terrain_validation"] = validate_checkpoint(
            terrain_run, args.terrain_iterations, apply_flat_gate=False
        )
        status.update(phase="complete", updated_at=now())
        atomic_json(status_path, status)
        return 0
    except Exception as exception:
        status.update(
            phase="monitor_failed",
            updated_at=now(),
            error=f"{type(exception).__name__}: {exception}",
        )
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    sys.exit(main())
