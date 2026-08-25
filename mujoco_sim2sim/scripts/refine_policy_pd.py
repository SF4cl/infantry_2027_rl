"""Second-stage policy PD sweep beyond the best edge of the coarse grid."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runtime import DEFAULT_MODEL, DEFAULT_POLICY, Runtime, VmcGains
from sweep_policy_pd import (
    MAX_WORKERS,
    OUTPUT_DIR,
    diagnostic_command,
    evaluate_group,
    plot_diagnostic,
    rows_to_arrays,
    run_profile,
)


LENGTH_GAINS = (
    (1200.0, 150.0),
    (1350.0, 60.0), (1350.0, 100.0), (1350.0, 140.0),
    (1500.0, 80.0), (1500.0, 120.0), (1500.0, 180.0),
    (1800.0, 180.0),
)

ANGLE_GAINS = (
    (80.0, 4.0), (80.0, 8.0),
    (90.0, 5.0), (90.0, 7.0),
    (100.0, 6.0), (100.0, 8.0), (100.0, 10.0),
    (120.0, 8.0), (120.0, 12.0), (120.0, 16.0),
)


def main() -> None:
    coarse_path = OUTPUT_DIR / "summary.json"
    if not coarse_path.exists():
        raise FileNotFoundError(f"Run sweep_policy_pd.py first: {coarse_path}")
    coarse = json.loads(coarse_path.read_text(encoding="utf-8"))
    groups = [
        (kp_length, kd_length, kp_angle, kd_angle)
        for kp_length, kd_length in LENGTH_GAINS
        for kp_angle, kd_angle in ANGLE_GAINS
    ]
    workers = min(MAX_WORKERS, os.cpu_count() or 1, len(groups))
    refined = []
    print(f"Evaluating {len(groups)} refinement groups with {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for count, result in enumerate(pool.imap_unordered(evaluate_group, groups, chunksize=1), 1):
            refined.append(result)
            print(
                f"[{count:2d}/{len(groups)}] L={result['kp_length']:g}/{result['kd_length']:g} "
                f"A={result['kp_angle']:g}/{result['kd_angle']:g} score={result['score']:.4f} "
                f"survived={result['survived_all']}",
                flush=True,
            )

    all_results = coarse["ranked_results"] + refined
    all_results.sort(key=lambda item: item["score"])
    for rank, result in enumerate(all_results, 1):
        result["rank"] = rank
    baseline = next(
        item for item in all_results
        if (item["kp_length"], item["kd_length"], item["kp_angle"], item["kd_angle"])
        == (900.0, 20.0, 50.0, 3.0)
    )
    summary = {
        "policy": str(DEFAULT_POLICY),
        "model": str(DEFAULT_MODEL),
        "coarse_groups": coarse["tested_parameter_groups"],
        "refinement_groups": len(groups),
        "total_groups": len(all_results),
        "baseline_rank": baseline["rank"],
        "ranked_results": all_results,
    }
    (OUTPUT_DIR / "refined_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fields = [
        "rank", "score", "survived_all", "kp_length", "kd_length", "kp_angle", "kd_angle",
        "mean_vx_mae_mps", "mean_yaw_mae_radps", "mean_height_mae_m",
        "mean_length_tracking_rmse_m", "mean_angle_tracking_rmse_rad", "mean_tilt_p90_deg",
        "max_tilt_deg", "mean_leg_saturation_fraction", "max_closure_residual_m",
    ]
    with (OUTPUT_DIR / "refined_ranking.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    matrix = [[0.0 for _ in ANGLE_GAINS] for _ in LENGTH_GAINS]
    for result in refined:
        i = LENGTH_GAINS.index((result["kp_length"], result["kd_length"]))
        j = ANGLE_GAINS.index((result["kp_angle"], result["kd_angle"]))
        matrix[i][j] = result["score"]
    fig, ax = plt.subplots(figsize=(15, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ANGLE_GAINS)), [f"{kp:g}/{kd:g}" for kp, kd in ANGLE_GAINS], rotation=45, ha="right")
    ax.set_yticks(range(len(LENGTH_GAINS)), [f"{kp:g}/{kd:g}" for kp, kd in LENGTH_GAINS])
    ax.set_xlabel("angle Kp/Kd")
    ax.set_ylabel("length Kp/Kd")
    ax.set_title("model_1600 policy PD refinement (lower is better)")
    fig.colorbar(image, ax=ax, label="composite score")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "refinement_heatmap.png", dpi=180)
    plt.close(fig)

    best = all_results[0]
    gains = VmcGains(best["kp_length"], best["kd_length"], best["kp_angle"], best["kd_angle"])
    runtime = Runtime(DEFAULT_POLICY, DEFAULT_MODEL, gains=gains, load_visuals=False)
    _, rows = run_profile(runtime, diagnostic_command, 20.0, retain_rows=True)
    arrays = rows_to_arrays(rows)
    plot_diagnostic(
        f"refined best: L={gains.kp_length:g}/{gains.kd_length:g}, A={gains.kp_angle:g}/{gains.kd_angle:g}",
        arrays,
        OUTPUT_DIR / "refined_best_diagnostic.png",
    )

    print("\nCombined top 10:")
    for result in all_results[:10]:
        print(
            f"#{result['rank']:2d} score={result['score']:.4f} "
            f"L={result['kp_length']:g}/{result['kd_length']:g} "
            f"A={result['kp_angle']:g}/{result['kd_angle']:g} "
            f"sat={100*result['mean_leg_saturation_fraction']:.2f}% "
            f"tilt={result['mean_tilt_p90_deg']:.2f}deg"
        )
    print(f"Baseline rank: {baseline['rank']} / {len(all_results)}")


if __name__ == "__main__":
    main()
