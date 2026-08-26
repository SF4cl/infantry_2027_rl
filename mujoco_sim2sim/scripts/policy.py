"""NumPy-only inference for the infantry_2027_v0 Fudan estimator policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


HISTORY_LENGTH = 5
FRAME_DIM = 25
HISTORY_DIM = HISTORY_LENGTH * FRAME_DIM
ACTION_DIM = 6
SCHEMA = "infantry-2027-v0-fudan-estimator"
CONTRACT = "infantry-2027-v0-flat-25d-v1"
LEGACY_ACTION_CONTRACT = "infantry-2027-v0-vmc-action-v1"
STABLE_V2_ACTION_CONTRACT = "infantry-2027-flat-stable-v2-vmc-action-v1"


def _elu(value: np.ndarray) -> np.ndarray:
    return np.where(value > 0.0, value, np.expm1(value))


class NumpyPolicy:
    ENCODER_LAYERS = (0, 2, 4)
    ACTOR_LAYERS = (0, 2, 4, 6)

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as archive:
            self.arrays = {name: archive[name].copy() for name in archive.files}
        self._validate()

    def _mlp(self, value: np.ndarray, prefix: str, layers: tuple[int, ...]) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        for index, layer in enumerate(layers):
            result = self.arrays[f"{prefix}_{layer}_weight"] @ result + self.arrays[f"{prefix}_{layer}_bias"]
            if index + 1 < len(layers):
                result = _elu(result)
        return result.astype(np.float32, copy=False)

    def infer(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        history = np.asarray(history, dtype=np.float32)
        if history.shape != (HISTORY_DIM,) or not np.isfinite(history).all():
            raise ValueError(f"Expected one finite {HISTORY_DIM}-D history, got {history.shape}")
        latent = self._mlp(history, "encoder", self.ENCODER_LAYERS)
        action = self._mlp(np.concatenate((history[-FRAME_DIM:], latent)), "actor", self.ACTOR_LAYERS)
        if not np.isfinite(action).all():
            raise FloatingPointError("Policy produced non-finite actions")
        return latent, action

    def decode_action(self, action: np.ndarray, base_height: float) -> dict[str, np.ndarray]:
        return decode_action(
            action,
            base_height,
            self.arrays.get("equilibrium_length_nodes", np.empty(0)),
            self.arrays.get("equilibrium_angle_nodes", np.empty(0)),
        )

    def _validate(self) -> None:
        expected = {
            "encoder_0_weight": (128, 125), "encoder_0_bias": (128,),
            "encoder_2_weight": (64, 128), "encoder_2_bias": (64,),
            "encoder_4_weight": (3, 64), "encoder_4_bias": (3,),
            "actor_0_weight": (128, 28), "actor_0_bias": (128,),
            "actor_2_weight": (64, 128), "actor_2_bias": (64,),
            "actor_4_weight": (32, 64), "actor_4_bias": (32,),
            "actor_6_weight": (6, 32), "actor_6_bias": (6,),
            "test_history": (125,), "test_latent": (3,), "test_action": (6,),
        }
        for name, shape in expected.items():
            if name not in self.arrays or self.arrays[name].shape != shape:
                actual = None if name not in self.arrays else self.arrays[name].shape
                raise ValueError(f"Invalid export array {name}: {actual}, expected {shape}")
            if not np.isfinite(self.arrays[name]).all():
                raise ValueError(f"Non-finite values in {name}")
        if str(self.arrays["checkpoint_schema"].item()) != SCHEMA:
            raise ValueError("Checkpoint schema does not match infantry_2027_v0")
        if str(self.arrays["contract"].item()) != CONTRACT:
            raise ValueError("Policy observation contract does not match this runtime")
        action_contract = str(
            self.arrays.get("action_contract", np.asarray(LEGACY_ACTION_CONTRACT)).item()
        )
        if action_contract not in (LEGACY_ACTION_CONTRACT, STABLE_V2_ACTION_CONTRACT):
            raise ValueError(f"Unknown policy action contract: {action_contract}")
        length_nodes = self.arrays.get("equilibrium_length_nodes", np.empty(0))
        angle_nodes = self.arrays.get("equilibrium_angle_nodes", np.empty(0))
        if length_nodes.shape != angle_nodes.shape or length_nodes.ndim != 1:
            raise ValueError("Invalid equilibrium length/angle table shapes")
        if action_contract == STABLE_V2_ACTION_CONTRACT:
            if length_nodes.size < 2 or not np.all(np.diff(length_nodes) > 0.0):
                raise ValueError("Stable-v2 action contract requires increasing equilibrium nodes")
        elif length_nodes.size or angle_nodes.size:
            raise ValueError("Legacy action contract must not contain equilibrium nodes")
        latent, action = self.infer(self.arrays["test_history"])
        if np.max(np.abs(latent - self.arrays["test_latent"])) > 2.0e-5:
            raise ValueError("Encoder export self-check failed")
        if np.max(np.abs(action - self.arrays["test_action"])) > 2.0e-5:
            raise ValueError("Actor export self-check failed")


def decode_action(
    action: np.ndarray,
    base_height: float,
    equilibrium_length_nodes: np.ndarray | None = None,
    equilibrium_angle_nodes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Match VmcWheelAction exactly, with nominal (zero-delay/zero-DR) gains."""
    raw = np.clip(np.asarray(action, dtype=np.float64), -100.0, 100.0)
    if raw.shape != (6,):
        raise ValueError(f"Expected six actions, got {raw.shape}")
    target_length = base_height + 0.012 + np.tanh(raw[[1, 4]]) * 0.03
    target_length = np.clip(target_length, 0.16, 0.33)
    target_angle = raw[[0, 3]] * 0.5
    length_nodes = np.asarray(
        () if equilibrium_length_nodes is None else equilibrium_length_nodes, dtype=np.float64
    )
    angle_nodes = np.asarray(
        () if equilibrium_angle_nodes is None else equilibrium_angle_nodes, dtype=np.float64
    )
    if length_nodes.size:
        target_angle += np.interp(target_length, length_nodes, angle_nodes)
    return {
        "raw": raw,
        "angle": target_angle,
        "length": target_length,
        "wheel": np.clip(raw[[2, 5]] * 20.0, -55.0, 55.0),
    }
