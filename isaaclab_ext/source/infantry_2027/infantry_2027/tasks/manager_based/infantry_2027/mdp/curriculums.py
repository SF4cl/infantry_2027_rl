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


def fudan_terrain_and_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "motion",
    move_up_distance: float = 4.0,
    move_down_tracking_threshold: float = 0.4,
    grow_tracking_threshold: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Reproduce Fudan's coupled terrain and per-environment command curriculum."""
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term(command_name)
    if terrain.terrain_levels is None:
        zero = torch.zeros((), device=env.device)
        return {"terrain_level": zero, "forward_range": zero, "yaw_range": zero}
    if env.common_step_counter == 0:
        return {
            "terrain_level": terrain.terrain_levels.float().mean(),
            "forward_range": command._forward_ranges[:, 1].mean(),
            "yaw_range": command._yaw_ranges[:, 1].mean(),
        }

    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    robot: Articulation = env.scene[asset_cfg.name]
    distance = torch.norm(
        robot.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1
    )
    linear_score = (
        env.reward_manager._episode_sums["tracking_lin_vel"][ids]
        / env.max_episode_length_s
    )
    yaw_score = (
        env.reward_manager._episode_sums["tracking_ang_vel"][ids]
        / env.max_episode_length_s
    )
    move_up = distance > move_up_distance
    move_down = (linear_score < move_down_tracking_threshold) & ~move_up

    # Capture out-of-range levels before TerrainImporter wraps/clamps them;
    # these are exactly Fudan's command-growth and command-shrink events.
    raw_levels = terrain.terrain_levels[ids] + move_up.long() - move_down.long()
    success_mask = raw_levels >= terrain.max_terrain_level
    fail_ids = ids[raw_levels < 0]
    tracking_success = success_mask & (linear_score > grow_tracking_threshold) & (
        yaw_score > grow_tracking_threshold
    )
    success_ids = ids[tracking_success]
    basic_success_ids = command.basic_success_ids(success_ids)
    command.update_ranges(fail_ids, success_ids, basic_success_ids)

    terrain.update_env_origins(ids, move_up, move_down)
    return {
        "terrain_level": terrain.terrain_levels.float().mean(),
        "forward_range": command._forward_ranges[:, 1].mean(),
        "yaw_range": command._yaw_ranges[:, 1].mean(),
    }
