"""Export an infantry_2027 estimator policy checkpoint to portable ONNX.

The exported graph has one float32 input and two float32 outputs:

    history [batch, 125] -> actions [batch, 6],
                            estimated_base_lin_vel_scaled [batch, 3]

The current 25-D frame is sliced from the end of ``history`` inside the graph,
so a deployment cannot accidentally pass inconsistent current/history inputs.
Only the supervised encoder and deterministic actor are exported.  Action
decoding, VMC, limits, and motor control intentionally remain in the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx.reference import ReferenceEvaluator


CHECKPOINT_SCHEMA = "infantry-2027-v0-fudan-estimator"
POLICY_CONTRACT = "infantry-2027-v0-flat-25d-v1"
HISTORY_LENGTH = 5
FRAME_DIM = 25
HISTORY_DIM = HISTORY_LENGTH * FRAME_DIM
LATENT_DIM = 3
ACTION_DIM = 6

EXPECTED_SHAPES = {
    "encoder.0.weight": (128, 125),
    "encoder.0.bias": (128,),
    "encoder.2.weight": (64, 128),
    "encoder.2.bias": (64,),
    "encoder.4.weight": (3, 64),
    "encoder.4.bias": (3,),
    "actor.0.weight": (128, 28),
    "actor.0.bias": (128,),
    "actor.2.weight": (64, 128),
    "actor.2.bias": (64,),
    "actor.4.weight": (32, 64),
    "actor.4.bias": (32,),
    "actor.6.weight": (6, 32),
    "actor.6.bias": (6,),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class InfantryPolicyOnnx(nn.Module):
    """Checkpoint-compatible encoder plus deterministic actor."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(HISTORY_DIM, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, LATENT_DIM),
        )
        self.actor = nn.Sequential(
            nn.Linear(FRAME_DIM + LATENT_DIM, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, ACTION_DIM),
        )

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(history)
        current_frame = history[:, -FRAME_DIM:]
        actions = self.actor(torch.cat((current_frame, latent), dim=-1))
        return actions, latent


def validate_checkpoint(checkpoint: dict, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    schema = checkpoint.get("checkpoint_schema")
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"{checkpoint_path} has checkpoint_schema={schema!r}; expected {CHECKPOINT_SCHEMA!r}."
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint_path} does not contain model_state_dict.")
    for name, expected_shape in EXPECTED_SHAPES.items():
        if name not in state:
            raise ValueError(f"Checkpoint is missing {name}.")
        actual_shape = tuple(state[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{name} has shape {actual_shape}; expected {expected_shape}.")
        if not torch.isfinite(state[name]).all():
            raise ValueError(f"{name} contains non-finite weights.")
    return {name: state[name].detach().cpu().float() for name in EXPECTED_SHAPES}


def add_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
    del model.metadata_props[:]
    for key, value in values.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def reference_outputs(model: nn.Module, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        actions, latent = model(torch.from_numpy(history))
    return actions.numpy(), latent.numpy()


def verify_onnx(
    onnx_path: Path,
    model: nn.Module,
    test_inputs: list[np.ndarray],
    atol: float,
    rtol: float,
) -> dict[str, float]:
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph, full_check=True)
    evaluator = ReferenceEvaluator(graph)
    maximum_action_error = 0.0
    maximum_latent_error = 0.0
    for history in test_inputs:
        expected_actions, expected_latent = reference_outputs(model, history)
        actual_actions, actual_latent = evaluator.run(None, {"history": history})
        np.testing.assert_allclose(actual_actions, expected_actions, atol=atol, rtol=rtol)
        np.testing.assert_allclose(actual_latent, expected_latent, atol=atol, rtol=rtol)
        maximum_action_error = max(
            maximum_action_error, float(np.max(np.abs(actual_actions - expected_actions)))
        )
        maximum_latent_error = max(
            maximum_latent_error, float(np.max(np.abs(actual_latent - expected_latent)))
        )
    return {
        "maximum_action_abs_error": maximum_action_error,
        "maximum_latent_abs_error": maximum_latent_error,
    }


def export(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"--output must end in .onnx: {output_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_encoder_state = validate_checkpoint(checkpoint, checkpoint_path)
    model = InfantryPolicyOnnx().cpu().eval()
    missing, unexpected = model.load_state_dict(actor_encoder_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch: missing={missing}, unexpected={unexpected}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_history = torch.zeros(1, HISTORY_DIM, dtype=torch.float32)
    # Match the PyTorch modules: the leading batch dimension is unrestricted.
    dynamic_axes = {
        "history": {0: "batch"},
        "actions": {0: "batch"},
        "estimated_base_lin_vel_scaled": {0: "batch"},
    }
    torch.onnx.export(
        model,
        dummy_history,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["history"],
        output_names=["actions", "estimated_base_lin_vel_scaled"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    checkpoint_hash = sha256(checkpoint_path)
    graph = onnx.load(output_path)
    metadata = {
        "policy_contract": POLICY_CONTRACT,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iteration": str(int(checkpoint.get("iter", -1))),
        "completed_iterations": str(
            int(checkpoint.get("completed_iterations", checkpoint.get("iter", -1)))
        ),
        "input": "history: float32[batch,125], five 25-D frames, oldest_to_newest",
        "output_actions": "float32[batch,6], raw actor actions, no clip or VMC decoding",
        "output_velocity": "float32[batch,3], estimated body linear velocity times 2.0",
        "frame_period_seconds": "0.01",
        "history_length": str(HISTORY_LENGTH),
        "frame_dim": str(FRAME_DIM),
        "action_dim": str(ACTION_DIM),
        "latent_dim": str(LATENT_DIM),
        "coordinate_convention": "+X forward, +Y left, +Z up, right-handed",
    }
    add_metadata(graph, metadata)
    onnx.save(graph, output_path)

    rng = np.random.default_rng(args.seed)
    golden_history = np.linspace(-0.5, 0.5, HISTORY_DIM, dtype=np.float32)[None, :]
    test_inputs = [
        np.zeros((1, HISTORY_DIM), dtype=np.float32),
        golden_history,
        rng.uniform(-1.0, 1.0, size=(1, HISTORY_DIM)).astype(np.float32),
    ]
    test_inputs.append(rng.uniform(-1.0, 1.0, size=(4, HISTORY_DIM)).astype(np.float32))
    verification = verify_onnx(output_path, model, test_inputs, args.atol, args.rtol)

    golden_actions, golden_latent = reference_outputs(model, golden_history)
    golden_path = output_path.with_suffix(".golden.json")
    golden = {
        "description": "Deterministic cross-runtime golden vector for the ONNX policy.",
        "input_name": "history",
        "output_names": ["actions", "estimated_base_lin_vel_scaled"],
        "history": golden_history[0].tolist(),
        "actions": golden_actions[0].tolist(),
        "estimated_base_lin_vel_scaled": golden_latent[0].tolist(),
        "atol": args.atol,
        "rtol": args.rtol,
    }
    golden_path.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_path = output_path.with_suffix(".onnx.json")
    manifest = {
        "policy_contract": POLICY_CONTRACT,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iteration": int(checkpoint.get("iter", -1)),
        "completed_iterations": int(
            checkpoint.get("completed_iterations", checkpoint.get("iter", -1))
        ),
        "onnx": str(output_path),
        "onnx_sha256": sha256(output_path),
        "onnx_opset": args.opset,
        "input": {"name": "history", "dtype": "float32", "shape": ["batch", HISTORY_DIM]},
        "outputs": [
            {"name": "actions", "dtype": "float32", "shape": ["batch", ACTION_DIM]},
            {
                "name": "estimated_base_lin_vel_scaled",
                "dtype": "float32",
                "shape": ["batch", LATENT_DIM],
            },
        ],
        "golden_vector": golden_path.name,
        "verification": {
            "backend": "onnx.reference.ReferenceEvaluator",
            "test_cases": len(test_inputs),
            "atol": args.atol,
            "rtol": args.rtol,
            **verification,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("POLICY_ONNX_EXPORT_OK")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export infantry_2027 Encoder+Actor checkpoint to ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input RSL-RL model_*.pt")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (default: 17)")
    parser.add_argument("--seed", type=int, default=2027, help="Verification random seed")
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
