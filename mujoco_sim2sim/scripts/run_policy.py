"""Run an exported infantry_2027 policy in MuJoCo, headless or with the viewer."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from runtime import DEFAULT_MODEL, DEFAULT_POLICY, Runtime, VmcGains


SCENARIOS = {
    "stand": (0.0, 0.0, 0.215),
    "forward": (0.8, 0.0, 0.215),
    "backward": (-0.8, 0.0, 0.215),
    "forward_left": (0.8, 0.8, 0.215),
    "forward_right": (0.8, -0.8, 0.215),
}

# Windows virtual-key codes. Numeric keypad avoids MuJoCo's built-in shortcuts.
KEYS = {
    "forward": 0x68, "backward": 0x62, "left": 0x64, "right": 0x66,
    "up": 0x67, "down": 0x61, "stop": 0x65, "height_reset": 0x60,
}


class KeypadCommand:
    """Match the latest Isaac Lab held-key command state machine.

    MuJoCo's passive-viewer callback reports only a key code, not press/release
    state. Polling GetAsyncKeyState is therefore required for correct release
    behavior on Windows.
    """

    def __init__(
        self,
        *,
        forward_speed: float,
        yaw_acceleration: float,
        moving_yaw_limit: float,
        point_yaw_limit: float,
        base_height: float,
        height_step: float,
        key_reader: Callable[[int], bool] | None = None,
    ):
        if key_reader is None:
            if sys.platform != "win32":
                raise RuntimeError("Held-key control currently requires Windows GetAsyncKeyState")
            get_state = ctypes.windll.user32.GetAsyncKeyState
            key_reader = lambda code: bool(get_state(code) & 0x8000)
        self._key_reader = key_reader
        self.forward_speed = forward_speed
        self.yaw_acceleration = yaw_acceleration
        self.moving_yaw_limit = moving_yaw_limit
        self.point_yaw_limit = point_yaw_limit
        self.base_height = base_height
        self.initial_base_height = base_height
        self.height_step = height_step
        self.yaw_rate = 0.0
        self._previous: set[str] = set()
        self._last_report = None

    def advance(self, dt: float) -> np.ndarray:
        pressed = {name for name, code in KEYS.items() if self._key_reader(code)}
        edges = pressed - self._previous
        if "up" in edges:
            self.base_height = min(0.318, self.base_height + self.height_step)
        if "down" in edges:
            self.base_height = max(0.148, self.base_height - self.height_step)
        if "height_reset" in edges:
            self.base_height = self.initial_base_height

        forward_axis = float("forward" in pressed) - float("backward" in pressed)
        turn_axis = float("left" in pressed) - float("right" in pressed)
        forward = self.forward_speed * forward_axis
        if "stop" in pressed:
            forward = 0.0
            self.yaw_rate = 0.0
        elif turn_axis == 0.0:
            # Latest play behavior: releasing both turn keys immediately zeros yaw.
            self.yaw_rate = 0.0
        else:
            yaw_limit = self.point_yaw_limit if forward_axis == 0.0 else self.moving_yaw_limit
            self.yaw_rate = float(np.clip(
                self.yaw_rate + self.yaw_acceleration * turn_axis * dt, -yaw_limit, yaw_limit
            ))
        command = np.array((forward, self.yaw_rate, self.base_height), dtype=np.float64)
        report = (tuple(sorted(pressed)), *np.round(command, 4))
        if report != self._last_report and (edges or pressed != self._previous):
            print(
                f"[KEYPAD] held={','.join(sorted(pressed)) or '-':<18} "
                f"vx={forward:+.2f} yaw={self.yaw_rate:+.2f} height={self.base_height:.3f}",
                flush=True,
            )
            self._last_report = report
        self._previous = pressed
        return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="stand")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "YAW", "HEIGHT"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--forward-speed", type=float, default=2.3)
    parser.add_argument("--yaw-acceleration", type=float, default=10.0)
    parser.add_argument("--moving-yaw-limit", type=float, default=4.0)
    parser.add_argument("--point-yaw-limit", type=float, default=10.0)
    parser.add_argument("--base-height", type=float, default=0.233)
    parser.add_argument("--height-step", type=float, default=0.02)
    parser.add_argument("--kp-length", type=float, default=900.0)
    parser.add_argument("--kd-length", type=float, default=20.0)
    parser.add_argument("--kp-angle", type=float, default=50.0)
    parser.add_argument("--kd-angle", type=float, default=3.0)
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    parser.add_argument(
        "--report-interval", type=float, default=0.0,
        help="Print policy targets and measured VMC state every N seconds (0 disables)",
    )
    args = parser.parse_args()
    if args.keyboard and not args.viewer:
        parser.error("--keyboard requires --viewer")
    if not 0.0 < args.forward_speed <= 2.3:
        parser.error("--forward-speed must be in (0, 2.3]")
    if args.yaw_acceleration <= 0.0:
        parser.error("--yaw-acceleration must be positive")
    if not 0.0 < args.moving_yaw_limit <= 4.0:
        parser.error("--moving-yaw-limit must be in (0, 4] rad/s")
    if not args.moving_yaw_limit <= args.point_yaw_limit <= 10.0:
        parser.error("--point-yaw-limit must be in [moving-yaw-limit, 10] rad/s")
    if not 0.148 <= args.base_height <= 0.318 or args.height_step <= 0.0:
        parser.error("invalid base-height or height-step")
    if args.report_interval < 0.0:
        parser.error("--report-interval must be non-negative")
    target = np.array(args.command if args.command else SCENARIOS[args.scenario], dtype=np.float64)
    initial_height = args.base_height if args.keyboard else 0.215
    gains = VmcGains(args.kp_length, args.kd_length, args.kp_angle, args.kd_angle)
    # Headless validation does not need the density-zero CAD render meshes.
    # The viewer path still loads the complete high-fidelity appearance.
    runtime = Runtime(
        args.policy, args.model, initial_height=initial_height, gains=gains,
        load_visuals=args.viewer,
    )
    keyboard = KeypadCommand(
        forward_speed=args.forward_speed,
        yaw_acceleration=args.yaw_acceleration,
        moving_yaw_limit=args.moving_yaw_limit,
        point_yaw_limit=args.point_yaw_limit,
        base_height=args.base_height,
        height_step=args.height_step,
    ) if args.keyboard else None

    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(runtime.model, runtime.data)
        viewer.cam.lookat[:] = (0.0, 0.0, 0.20)
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        if args.keyboard:
            print("[INFO] 小键盘 8/2=按住前进/后退，4/6=按住累加转向、松开归零")
            print("[INFO] 小键盘 7/1=高度瞬间加/减，5=停止，0=重置目标高度")

    rows = []
    next_report = 0.0
    start_wall = time.perf_counter()
    steps = math.ceil(args.duration / 0.01)
    try:
        for _ in range(steps):
            simulation_time = float(runtime.data.time)
            command = keyboard.advance(0.01) if keyboard is not None else (
                np.array((0.0, 0.0, 0.215)) if simulation_time < args.settle else target
            )
            runtime.set_command(command)
            metrics = runtime.step()
            rows.append(metrics)
            if args.report_interval > 0.0 and metrics["time"] + 1.0e-9 >= next_report:
                print(
                    f"[VMC] t={metrics['time']:7.2f}s "
                    f"target_L=({metrics['target_length'][0]:.4f},{metrics['target_length'][1]:.4f})m "
                    f"actual_L=({metrics['length'][0]:.4f},{metrics['length'][1]:.4f})m "
                    f"target_theta=({metrics['target_angle'][0]:+.3f},{metrics['target_angle'][1]:+.3f})rad "
                    f"actual_theta=({metrics['leg_angle'][0]:+.3f},{metrics['leg_angle'][1]:+.3f})rad "
                    f"inner_peak={metrics['inner_leg_effort_peak']:.2f}Nm",
                    flush=True,
                )
                next_report += args.report_interval
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                remaining = start_wall + float(runtime.data.time) - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
            if metrics["failed"]:
                break
    finally:
        if viewer is not None:
            viewer.close()

    settled = [row for row in rows if row["time"] >= args.settle] or rows
    summary = {
        "scenario": args.scenario, "target": target.tolist(), "duration_s": rows[-1]["time"],
        "survived": not any(row["failed"] for row in rows),
        "vx_mean_mps": float(np.mean([row["linear"][0] for row in settled])),
        "vx_mae_mps": float(np.mean([abs(row["linear"][0] - target[0]) for row in settled])),
        "yaw_mean_radps": float(np.mean([row["angular"][2] for row in settled])),
        "yaw_mae_radps": float(np.mean([abs(row["angular"][2] - target[1]) for row in settled])),
        "height_mean_m": float(np.mean([row["height"] for row in settled])),
        "tilt_p90_deg": float(np.degrees(np.quantile([row["tilt"] for row in settled], 0.9))),
        "max_closure_residual_m": float(max(row["closure"] for row in rows)),
        "max_leg_effort_nm": float(max(np.max(np.abs(row["leg_effort"])) for row in rows)),
        "max_wheel_effort_nm": float(max(np.max(np.abs(row["wheel_effort"])) for row in rows)),
        "vmc_gains": {
            "kp_length": gains.kp_length, "kd_length": gains.kd_length,
            "kp_angle": gains.kp_angle, "kd_angle": gains.kd_angle,
        },
        "target_length_mean_m": np.mean([row["target_length"] for row in settled], axis=0).tolist(),
        "target_length_range_m": [
            float(np.min([row["target_length"] for row in settled])),
            float(np.max([row["target_length"] for row in settled])),
        ],
        "target_length_tracking_rmse_m": float(np.sqrt(np.mean([
            np.mean((row["target_length"] - row["length"]) ** 2) for row in settled
        ]))),
        "checkpoint_iteration": int(runtime.policy.arrays["checkpoint_iteration"].item()),
        "checkpoint_sha256": str(runtime.policy.arrays["checkpoint_sha256"].item()),
        "model_sha256": runtime.model_sha256,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("SIM2SIM_RESULT", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
