"""Small domain-randomization helpers missing from the stock event library."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


def randomize_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
) -> None:
    asset: Articulation = env.scene[asset_cfg.name]
    inertia = asset.root_physx_view.get_inertias()
    tensor_device = inertia.device
    ids = torch.arange(env.scene.num_envs, device=tensor_device) if env_ids is None else torch.as_tensor(env_ids, device=tensor_device)
    body_ids = torch.as_tensor(asset_cfg.body_ids, device=tensor_device)
    scale = torch.empty(ids.numel(), body_ids.numel(), 1, device=inertia.device).uniform_(*scale_range)
    inertia[ids[:, None], body_ids] *= scale
    asset.root_physx_view.set_inertias(inertia, ids)


def randomize_default_joint_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
    asset_cfg: SceneEntityCfg,
    offset_range: tuple[float, float],
) -> None:
    asset: Articulation = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device=asset.device) if env_ids is None else torch.as_tensor(env_ids, device=asset.device)
    joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device)
    offset = torch.empty(ids.numel(), joint_ids.numel(), device=asset.device).uniform_(*offset_range)
    asset.data.default_joint_pos[ids[:, None], joint_ids] += offset


def randomize_rigid_body_com_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
) -> None:
    """Randomize CoM and retain the sampled *offset* for privileged observations.

    Isaac Lab exposes the resulting absolute body-frame CoM.  The Fudan critic,
    however, receives the random offset that was added to the nominal CoM.
    Keeping the sampled value avoids leaking an asset-dependent constant and
    preserves that convention exactly.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids_cpu = (
        torch.arange(env.scene.num_envs, device="cpu")
        if env_ids is None
        else torch.as_tensor(env_ids, device="cpu", dtype=torch.long)
    )
    body_ids = (
        torch.arange(asset.num_bodies, device="cpu", dtype=torch.long)
        if asset_cfg.body_ids == slice(None)
        else torch.as_tensor(asset_cfg.body_ids, device="cpu", dtype=torch.long)
    )
    ranges = torch.tensor(
        [com_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
        device="cpu",
        dtype=torch.float32,
    )
    offset = ranges[:, 0] + torch.rand((ids_cpu.numel(), 3), device="cpu") * (ranges[:, 1] - ranges[:, 0])
    coms = asset.root_physx_view.get_coms().clone()
    coms[ids_cpu[:, None], body_ids, :3] += offset[:, None, :]
    asset.root_physx_view.set_coms(coms, ids_cpu)

    if not hasattr(env, "_infantry_com_offset"):
        env._infantry_com_offset = torch.zeros(env.scene.num_envs, 3, device=env.device)
    env._infantry_com_offset[ids_cpu.to(env.device)] = offset.to(env.device)


def assign_random_disturbance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
) -> None:
    """Assign one Fudan disturbance type for the lifetime of each episode."""
    ids = (
        torch.arange(env.scene.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    )
    if not hasattr(env, "_infantry_disturbance_type"):
        env._infantry_disturbance_type = torch.full(
            (env.scene.num_envs,), -1, dtype=torch.long, device=env.device
        )
    # 0: body-frame push impulse; 1: world-frame downward impulse.
    env._infantry_disturbance_type[ids] = torch.randint(0, 2, (ids.numel(),), device=env.device)


def random_push_or_downward_impulse(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    push_speed: float = 2.0,
    downward_speed_range: tuple[float, float] = (2.4, 2.8),
) -> None:
    """Apply the disturbance assigned at the most recent environment reset."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not hasattr(env, "_infantry_disturbance_type"):
        assign_random_disturbance(env, ids)
    asset: Articulation = env.scene[asset_cfg.name]
    downward_mask = env._infantry_disturbance_type[ids] == 1
    push_ids = ids[~downward_mask]
    downward_ids = ids[downward_mask]
    if push_ids.numel() > 0:
        delta_v_b = torch.empty(push_ids.numel(), 3, device=env.device).uniform_(-push_speed, push_speed)
        delta_v_b[:, 2] *= 0.5
        delta_v_w = quat_apply(asset.data.root_quat_w[push_ids], delta_v_b)
        velocity = asset.data.root_vel_w[push_ids].clone()
        velocity[:, :3] += delta_v_w
        asset.write_root_velocity_to_sim(velocity, env_ids=push_ids)
    if downward_ids.numel() > 0:
        lo, hi = downward_speed_range
        velocity = asset.data.root_vel_w[downward_ids].clone()
        velocity[:, 2] -= torch.empty(downward_ids.numel(), device=env.device).uniform_(lo, hi)
        asset.write_root_velocity_to_sim(velocity, env_ids=downward_ids)
