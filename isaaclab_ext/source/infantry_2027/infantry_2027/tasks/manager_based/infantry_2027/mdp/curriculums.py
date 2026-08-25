"""Terrain curricula adapted from the final Fudan terrain experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def fudan_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_distance: float = 4.0,
    tracking_threshold: float = 0.4,
) -> torch.Tensor:
    """Move individual environments up/down the ten reference difficulty rows."""
    terrain: TerrainImporter = env.scene.terrain
    if terrain.terrain_levels is None:
        return torch.zeros((), device=env.device)
    if env.common_step_counter == 0:
        return terrain.terrain_levels.float().mean()

    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    robot: Articulation = env.scene[asset_cfg.name]
    distance = torch.norm(robot.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1)
    move_up = distance > move_up_distance
    tracking_sum = env.reward_manager._episode_sums["tracking_lin_vel"][ids]
    tracking_score = tracking_sum / env.max_episode_length_s
    move_down = (tracking_score < tracking_threshold) & ~move_up
    terrain.update_env_origins(ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()
