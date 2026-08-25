"""Create a command/target/response diagnostic plot for one exported policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from runtime import DEFAULT_MODEL, Runtime, VmcGains
from sweep_policy_pd import diagnostic_command, plot_diagnostic, rows_to_arrays, run_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="policy")
    parser.add_argument("--kp-length", type=float, default=900.0)
    parser.add_argument("--kd-length", type=float, default=20.0)
    parser.add_argument("--kp-angle", type=float, default=50.0)
    parser.add_argument("--kd-angle", type=float, default=3.0)
    args = parser.parse_args()

    gains = VmcGains(args.kp_length, args.kd_length, args.kp_angle, args.kd_angle)
    runtime = Runtime(args.policy, args.model, gains=gains, load_visuals=False)
    metrics, rows = run_profile(runtime, diagnostic_command, 20.0, retain_rows=True)
    arrays = rows_to_arrays(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / f"{args.name}_trace.npz", **arrays)
    plot_diagnostic(
        f"{args.name}: L={gains.kp_length:g}/{gains.kd_length:g}, A={gains.kp_angle:g}/{gains.kd_angle:g}",
        arrays,
        args.output_dir / f"{args.name}_diagnostic.png",
    )
    report = {
        "policy": str(args.policy.resolve()),
        "checkpoint_iteration": int(runtime.policy.arrays["checkpoint_iteration"].item()),
        "gains": gains.__dict__,
        "profile": "stand -> forward -> stop -> forward+turn -> stop -> minimum -> maximum -> nominal height",
        **metrics,
    }
    (args.output_dir / f"{args.name}_diagnostic.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
