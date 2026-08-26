"""Export an RSL-RL checkpoint into a portable NumPy policy bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from policy import CONTRACT, LEGACY_ACTION_CONTRACT, SCHEMA, STABLE_V2_ACTION_CONTRACT


ENCODER_LAYERS = (0, 2, 4)
ACTOR_LAYERS = (0, 2, 4, 6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def mlp(value: torch.Tensor, state: dict, prefix: str, layers: tuple[int, ...]) -> torch.Tensor:
    result = value
    for index, layer in enumerate(layers):
        result = F.linear(result, state[f"{prefix}.{layer}.weight"], state[f"{prefix}.{layer}.bias"])
        if index + 1 < len(layers):
            result = F.elu(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action-contract",
        choices=(LEGACY_ACTION_CONTRACT, STABLE_V2_ACTION_CONTRACT),
        default=LEGACY_ACTION_CONTRACT,
        help="Physical action decoder used by the checkpoint's training task.",
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema") != SCHEMA:
        raise ValueError(f"Unexpected schema: {checkpoint.get('checkpoint_schema')!r}")
    state = checkpoint["model_state_dict"]
    arrays: dict[str, np.ndarray] = {}
    for prefix, layers in (("encoder", ENCODER_LAYERS), ("actor", ACTOR_LAYERS)):
        for layer in layers:
            for field in ("weight", "bias"):
                name = f"{prefix}.{layer}.{field}"
                arrays[name.replace(".", "_")] = state[name].detach().float().numpy()
    history = torch.linspace(-0.5, 0.5, 125)
    with torch.inference_mode():
        latent = mlp(history, state, "encoder", ENCODER_LAYERS)
        action = mlp(torch.cat((history[-25:], latent)), state, "actor", ACTOR_LAYERS)
    if args.action_contract == STABLE_V2_ACTION_CONTRACT:
        equilibrium_length_nodes = np.asarray((0.16, 0.22, 0.28, 0.33), dtype=np.float64)
        equilibrium_angle_nodes = np.asarray((0.0, 0.0, -0.005, -0.005), dtype=np.float64)
    else:
        equilibrium_length_nodes = np.empty(0, dtype=np.float64)
        equilibrium_angle_nodes = np.empty(0, dtype=np.float64)
    arrays.update({
        "test_history": history.numpy(), "test_latent": latent.numpy(), "test_action": action.numpy(),
        "checkpoint_schema": np.asarray(SCHEMA), "contract": np.asarray(CONTRACT),
        "action_contract": np.asarray(args.action_contract),
        "equilibrium_length_nodes": equilibrium_length_nodes,
        "equilibrium_angle_nodes": equilibrium_angle_nodes,
        "checkpoint_iteration": np.asarray(int(checkpoint["iter"])),
        "completed_iterations": np.asarray(int(checkpoint.get("completed_iterations", checkpoint["iter"]))),
        "checkpoint_sha256": np.asarray(sha256(checkpoint_path)),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    manifest = {
        "contract": CONTRACT, "action_contract": args.action_contract,
        "equilibrium_length_nodes": equilibrium_length_nodes.tolist(),
        "equilibrium_angle_nodes": equilibrium_angle_nodes.tolist(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_iteration": int(checkpoint["iter"]),
        "completed_iterations": int(checkpoint.get("completed_iterations", checkpoint["iter"])),
        "export": str(output_path), "export_sha256": sha256(output_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("POLICY_EXPORT_OK", json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
