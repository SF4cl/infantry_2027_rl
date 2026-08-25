"""Evaluate model_1600 with many VMC PD groups in full MuJoCo sim2sim."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runtime import DEFAULT_MODEL, DEFAULT_POLICY, Runtime, VmcGains
from vmc import wrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "policy_pd_sweep"

LENGTH_GAINS = (
    (600.0, 15.0), (600.0, 40.0),
    (750.0, 20.0), (750.0, 50.0),
    (900.0, 10.0), (900.0, 20.0), (900.0, 40.0), (900.0, 80.0),
    (1050.0, 30.0), (1050.0, 60.0),
    (1200.0, 50.0), (1200.0, 100.0),
)

ANGLE_GAINS = (
    (35.0, 2.0),
    (50.0, 1.5), (50.0, 3.0), (50.0, 5.0), (50.0, 8.0),
    (65.0, 3.0), (65.0, 5.0), (65.0, 8.0),
    (80.0, 6.0), (80.0, 10.0),
)

MAX_WORKERS = 8


def constant_profile(target: tuple[float, float, float]):
    def command(time_s: float) -> np.ndarray:
        return np.asarray((0.0, 0.0, 0.215) if time_s < 1.0 else target, dtype=np.float64)
    return command


def height_steps(time_s: float) -> np.ndarray:
    if time_s < 1.0:
        height = 0.215
    elif time_s < 3.5:
        height = 0.148
    elif time_s < 6.0:
        height = 0.318
    else:
        height = 0.215
    return np.asarray((0.0, 0.0, height), dtype=np.float64)


def motion_steps(time_s: float) -> np.ndarray:
    if time_s < 1.0:
        return np.asarray((0.0, 0.0, 0.215))
    if time_s < 3.5:
        return np.asarray((0.8, 0.0, 0.215))
    if time_s < 5.5:
        return np.asarray((0.0, 0.0, 0.215))
    if time_s < 8.0:
        return np.asarray((0.8, 0.8, 0.215))
    return np.asarray((0.0, 0.0, 0.215))


PROFILES = (
    ("stand", constant_profile((0.0, 0.0, 0.215)), 6.0),
    ("forward", constant_profile((0.8, 0.0, 0.215)), 6.0),
    ("backward", constant_profile((-0.8, 0.0, 0.215)), 6.0),
    ("forward_turn", constant_profile((0.8, 0.8, 0.215)), 6.0),
    ("minimum_height", constant_profile((0.0, 0.0, 0.148)), 6.0),
    ("maximum_height", constant_profile((0.0, 0.0, 0.318)), 6.0),
    ("height_steps", height_steps, 8.0),
    ("motion_steps", motion_steps, 10.0),
)


def run_profile(runtime: Runtime, command_fn, duration_s: float, retain_rows: bool = False):
    runtime.initialize(0.215)
    rows = []
    for _ in range(math.ceil(duration_s / 0.01)):
        runtime.set_command(command_fn(float(runtime.data.time)))
        metrics = runtime.step()
        rows.append(metrics)
        if metrics["failed"]:
            break

    evaluation = [row for row in rows if row["time"] >= 1.0] or rows
    command = np.asarray([row["command"] for row in evaluation])
    linear = np.asarray([row["linear"] for row in evaluation])
    angular = np.asarray([row["angular"] for row in evaluation])
    length = np.asarray([row["length"] for row in evaluation])
    target_length = np.asarray([row["target_length"] for row in evaluation])
    angle = np.asarray([row["leg_angle"] for row in evaluation])
    target_angle = np.asarray([row["target_angle"] for row in evaluation])
    action = np.asarray([row["action"] for row in evaluation])
    effort = np.asarray([row["leg_effort"] for row in evaluation])
    tilt = np.asarray([row["tilt"] for row in evaluation])
    height = np.asarray([row["height"] for row in evaluation])
    length_delta = np.diff(target_length, axis=0) / 0.01 if len(target_length) > 1 else np.zeros((1, 2))
    angle_error = np.arctan2(np.sin(target_angle - angle), np.cos(target_angle - angle))
    metrics = {
        "survived": not any(row["failed"] for row in rows) and rows[-1]["time"] >= duration_s - 0.02,
        "duration_s": float(rows[-1]["time"]),
        "vx_mae_mps": float(np.mean(np.abs(linear[:, 0] - command[:, 0]))),
        "yaw_mae_radps": float(np.mean(np.abs(angular[:, 2] - command[:, 1]))),
        "height_mae_m": float(np.mean(np.abs(height - command[:, 2]))),
        "length_tracking_rmse_m": float(np.sqrt(np.mean((target_length - length) ** 2))),
        "angle_tracking_rmse_rad": float(np.sqrt(np.mean(angle_error**2))),
        "tilt_p90_deg": float(np.degrees(np.quantile(tilt, 0.9))),
        "tilt_max_deg": float(np.degrees(np.max(tilt))),
        "leg_saturation_fraction": float(np.mean([row["inner_leg_saturation_fraction"] for row in evaluation])),
        "peak_inner_leg_effort_nm": float(max(row["inner_leg_effort_peak"] for row in rows)),
        "max_closure_residual_m": float(max(row["closure"] for row in rows)),
        "target_length_min_m": float(np.min(target_length)),
        "target_length_max_m": float(np.max(target_length)),
        "target_length_std_m": float(np.std(target_length)),
        "target_length_rate_p95_mps": float(np.quantile(np.abs(length_delta), 0.95)),
        "target_length_total_variation_m": float(np.mean(np.sum(np.abs(np.diff(target_length, axis=0)), axis=0))),
        "raw_length_action_std": float(np.std(action[:, (1, 4)])),
    }
    return metrics, rows if retain_rows else None


def aggregate(gains: VmcGains, profiles: dict[str, dict]) -> dict:
    values = list(profiles.values())
    survived = all(value["survived"] for value in values)
    mean = lambda key: float(np.mean([value[key] for value in values]))
    maximum = lambda key: float(max(value[key] for value in values))
    score = (
        mean("vx_mae_mps") / 0.35
        + mean("yaw_mae_radps") / 0.30
        + mean("height_mae_m") / 0.025
        + mean("length_tracking_rmse_m") / 0.012
        + mean("angle_tracking_rmse_rad") / 0.12
        + mean("tilt_p90_deg") / 12.0
        + mean("leg_saturation_fraction") / 0.10
    )
    if not survived:
        score += 100.0 + 10.0 * sum(not value["survived"] for value in values)
    return {
        "kp_length": gains.kp_length,
        "kd_length": gains.kd_length,
        "kp_angle": gains.kp_angle,
        "kd_angle": gains.kd_angle,
        "score": score,
        "survived_all": survived,
        "mean_vx_mae_mps": mean("vx_mae_mps"),
        "mean_yaw_mae_radps": mean("yaw_mae_radps"),
        "mean_height_mae_m": mean("height_mae_m"),
        "mean_length_tracking_rmse_m": mean("length_tracking_rmse_m"),
        "mean_angle_tracking_rmse_rad": mean("angle_tracking_rmse_rad"),
        "mean_tilt_p90_deg": mean("tilt_p90_deg"),
        "max_tilt_deg": maximum("tilt_max_deg"),
        "mean_leg_saturation_fraction": mean("leg_saturation_fraction"),
        "max_closure_residual_m": maximum("max_closure_residual_m"),
        "profiles": profiles,
    }


def evaluate_group(values: tuple[float, float, float, float]) -> dict:
    """Worker entry point; each process owns its MuJoCo model and policy state."""
    gains = VmcGains(*values)
    runtime = Runtime(DEFAULT_POLICY, DEFAULT_MODEL, gains=gains, load_visuals=False)
    profiles = {}
    for name, command_fn, duration in PROFILES:
        metrics, _ = run_profile(runtime, command_fn, duration)
        profiles[name] = metrics
    return aggregate(gains, profiles)


def diagnostic_command(time_s: float) -> np.ndarray:
    if time_s < 2.0:
        return np.asarray((0.0, 0.0, 0.215))
    if time_s < 5.0:
        return np.asarray((0.8, 0.0, 0.215))
    if time_s < 7.0:
        return np.asarray((0.0, 0.0, 0.215))
    if time_s < 10.0:
        return np.asarray((0.8, 0.8, 0.215))
    if time_s < 12.0:
        return np.asarray((0.0, 0.0, 0.215))
    if time_s < 15.0:
        return np.asarray((0.0, 0.0, 0.148))
    if time_s < 18.0:
        return np.asarray((0.0, 0.0, 0.318))
    return np.asarray((0.0, 0.0, 0.215))


def rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    keys = (
        "time", "command", "linear", "angular", "height", "length", "target_length",
        "leg_angle", "target_angle", "action", "leg_effort", "closure", "tilt",
    )
    return {key: np.asarray([row[key] for row in rows]) for key in keys}


def plot_heatmap(rows: list[dict]) -> None:
    matrix = np.full((len(LENGTH_GAINS), len(ANGLE_GAINS)), np.nan)
    for row in rows:
        i = LENGTH_GAINS.index((row["kp_length"], row["kd_length"]))
        j = ANGLE_GAINS.index((row["kp_angle"], row["kd_angle"]))
        matrix[i, j] = row["score"]
    fig, ax = plt.subplots(figsize=(15, 8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ANGLE_GAINS)), [f"{kp:g}/{kd:g}" for kp, kd in ANGLE_GAINS], rotation=45, ha="right")
    ax.set_yticks(range(len(LENGTH_GAINS)), [f"{kp:g}/{kd:g}" for kp, kd in LENGTH_GAINS])
    ax.set_xlabel("angle Kp/Kd")
    ax.set_ylabel("length Kp/Kd")
    ax.set_title("model_1600 full sim2sim VMC PD sweep (lower is better)")
    fig.colorbar(image, ax=ax, label="composite score")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_heatmap.png", dpi=180)
    plt.close(fig)


def plot_diagnostic(name: str, arrays: dict[str, np.ndarray], output: Path) -> None:
    time = arrays["time"]
    fig, axes = plt.subplots(6, 1, figsize=(15, 18), sharex=True)
    axes[0].plot(time, arrays["command"][:, 2], "k--", label="commanded base height")
    axes[0].plot(time, arrays["height"], label="actual base height")
    axes[0].set_ylabel("base height [m]")
    axes[1].plot(time, arrays["target_length"][:, 0], "C0--", label="left target")
    axes[1].plot(time, arrays["length"][:, 0], "C0", label="left actual")
    axes[1].plot(time, arrays["target_length"][:, 1], "C1--", label="right target")
    axes[1].plot(time, arrays["length"][:, 1], "C1", label="right actual")
    axes[1].set_ylabel("leg length [m]")
    axes[2].plot(time, arrays["target_angle"][:, 0], "C0--", label="left target")
    axes[2].plot(time, arrays["leg_angle"][:, 0], "C0", label="left actual")
    axes[2].plot(time, arrays["target_angle"][:, 1], "C1--", label="right target")
    axes[2].plot(time, arrays["leg_angle"][:, 1], "C1", label="right actual")
    axes[2].set_ylabel("leg angle [rad]")
    axes[3].plot(time, arrays["command"][:, 0], "k--", label="vx command")
    axes[3].plot(time, arrays["linear"][:, 0], label="actual vx")
    axes[3].plot(time, arrays["command"][:, 1], "C3--", label="yaw command")
    axes[3].plot(time, arrays["angular"][:, 2], "C3", label="actual yaw")
    axes[3].set_ylabel("velocity")
    for index, label in zip((1, 4), ("left length action", "right length action")):
        axes[4].plot(time, arrays["action"][:, index], label=label)
    axes[4].set_ylabel("raw policy action")
    axes[5].plot(time, np.max(np.abs(arrays["leg_effort"]), axis=1), label="peak leg effort")
    axes[5].axhline(45.0, color="k", linestyle="--", label="45 Nm limit")
    axes[5].set_ylabel("effort [Nm]")
    axes[5].set_xlabel("simulation time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    groups = [
        (kp_length, kd_length, kp_angle, kd_angle)
        for kp_length, kd_length in LENGTH_GAINS
        for kp_angle, kd_angle in ANGLE_GAINS
    ]
    total = len(groups)
    workers = min(MAX_WORKERS, os.cpu_count() or 1, total)
    print(f"Evaluating {total} groups with {workers} worker processes", flush=True)
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for count, result in enumerate(pool.imap_unordered(evaluate_group, groups, chunksize=1), 1):
            results.append(result)
            print(
                f"[{count:3d}/{total}] L={result['kp_length']:g}/{result['kd_length']:g} "
                f"A={result['kp_angle']:g}/{result['kd_angle']:g} "
                f"score={result['score']:.4f} survived={result['survived_all']}",
                flush=True,
            )

    results.sort(key=lambda item: item["score"])
    for rank, result in enumerate(results, 1):
        result["rank"] = rank
    baseline = next(
        item for item in results
        if (item["kp_length"], item["kd_length"], item["kp_angle"], item["kd_angle"])
        == (900.0, 20.0, 50.0, 3.0)
    )
    summary = {
        "policy": str(DEFAULT_POLICY),
        "model": str(DEFAULT_MODEL),
        "tested_parameter_groups": total,
        "physics_hz": 500,
        "policy_hz": 100,
        "baseline_rank": baseline["rank"],
        "ranked_results": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fields = [
        "rank", "score", "survived_all", "kp_length", "kd_length", "kp_angle", "kd_angle",
        "mean_vx_mae_mps", "mean_yaw_mae_radps", "mean_height_mae_m",
        "mean_length_tracking_rmse_m", "mean_angle_tracking_rmse_rad", "mean_tilt_p90_deg",
        "max_tilt_deg", "mean_leg_saturation_fraction", "max_closure_residual_m",
    ]
    with (OUTPUT_DIR / "ranking.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    plot_heatmap(results)

    selected = (("best", results[0]), ("baseline", baseline))
    for name, result in selected:
        gains = VmcGains(result["kp_length"], result["kd_length"], result["kp_angle"], result["kd_angle"])
        runtime = Runtime(DEFAULT_POLICY, DEFAULT_MODEL, gains=gains, load_visuals=False)
        _, rows = run_profile(runtime, diagnostic_command, 20.0, retain_rows=True)
        arrays = rows_to_arrays(rows)
        np.savez_compressed(OUTPUT_DIR / f"{name}_diagnostic_trace.npz", **arrays)
        plot_diagnostic(
            f"{name}: L={gains.kp_length:g}/{gains.kd_length:g}, A={gains.kp_angle:g}/{gains.kd_angle:g}",
            arrays,
            OUTPUT_DIR / f"{name}_diagnostic.png",
        )

    print("\nTop 10:")
    for result in results[:10]:
        print(
            f"#{result['rank']:2d} score={result['score']:.4f} "
            f"L={result['kp_length']:g}/{result['kd_length']:g} "
            f"A={result['kp_angle']:g}/{result['kd_angle']:g} "
            f"vx={result['mean_vx_mae_mps']:.3f} yaw={result['mean_yaw_mae_radps']:.3f} "
            f"height={1000*result['mean_height_mae_m']:.1f}mm "
            f"Ltrack={1000*result['mean_length_tracking_rmse_m']:.1f}mm"
        )
    print(f"Baseline rank: {baseline['rank']} / {total}, score={baseline['score']:.4f}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
