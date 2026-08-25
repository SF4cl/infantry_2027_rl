"""Fudan-compatible 25-D proprioception and compact privileged state."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from .actions import VmcWheelAction


def _action(env: ManagerBasedRLEnv) -> VmcWheelAction:
    term = env.action_manager.get_term("vmc")
    if not isinstance(term, VmcWheelAction):
        raise TypeError("The 'vmc' action term has the wrong type.")
    return term


def proprioception(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    noise_scale: float = 0.0,
    max_abs: float = 100.0,
) -> torch.Tensor:
    """One reference frame: 3+3+3+4+6+6 = 25 values."""
    term = _action(env)
    robot = term._asset
    leg_q = robot.data.joint_pos[:, term._leg_ids] - robot.data.default_joint_pos[:, term._leg_ids]
    # Reference ordering: left leg, left wheel, right leg, right wheel.
    vel = torch.cat(
        (
            robot.data.joint_vel[:, term._leg_ids[:2]],
            robot.data.joint_vel[:, term._wheel_ids[:1]],
            robot.data.joint_vel[:, term._leg_ids[2:]],
            robot.data.joint_vel[:, term._wheel_ids[1:]],
        ),
        dim=-1,
    )
    command = env.command_manager.get_command(command_name) * robot.data.root_ang_vel_b.new_tensor((2.0, 0.25, 5.0))
    frame = torch.cat(
        (
            robot.data.root_ang_vel_b * 0.25,
            robot.data.projected_gravity_b,
            command,
            leg_q,
            vel * 0.05,
            term.raw_actions,
        ),
        dim=-1,
    )
    if noise_scale > 0.0:
        amplitude = frame.new_tensor(
            (*([0.05] * 3), *([0.05] * 3), *([0.0] * 3), *([0.02] * 4), *([0.075] * 6), *([0.0] * 6))
        )
        frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * amplitude * noise_scale
    return torch.nan_to_num(frame, nan=0.0, posinf=max_abs, neginf=-max_abs).clamp(-max_abs, max_abs)


def true_base_lin_vel(env: ManagerBasedRLEnv, max_abs: float = 50.0) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return torch.nan_to_num(robot.data.root_lin_vel_b).clamp(-max_abs, max_abs)


def scaled_base_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    return true_base_lin_vel(env) * 2.0


def local_base_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    nominal_height: float = 0.233,
) -> torch.Tensor:
    """Mean base clearance over the reference 11-by-7 yaw-aligned scan."""
    robot: Articulation = env.scene["robot"]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    clearance = robot.data.root_pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
    return torch.nan_to_num(
        clearance, nan=nominal_height, posinf=nominal_height, neginf=nominal_height
    ).mean(dim=1)


def privileged_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    nominal_height: float = 0.233,
) -> torch.Tensor:
    """Reference terrain scan for the critic only, centered for this asset."""
    robot: Articulation = env.scene["robot"]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    clearance = robot.data.root_pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
    clearance = torch.nan_to_num(
        clearance, nan=nominal_height, posinf=nominal_height, neginf=nominal_height
    )
    return ((clearance - nominal_height).clamp(-1.0, 1.0) * 5.0).clamp(-100.0, 100.0)


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _action(env).previous_action


def previous_previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _action(env).previous_previous_action


def controlled_joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(robot.data.joint_acc[:, asset_cfg.joint_ids] * 0.0025).clamp(-100.0, 100.0)


def applied_effort(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = _action(env)
    effort = torch.cat(
        (
            term.last_leg_effort[:, :2], term.last_wheel_effort[:, :1],
            term.last_leg_effort[:, 2:], term.last_wheel_effort[:, 1:],
        ),
        dim=-1,
    )
    return (effort * 0.05).clamp(-100.0, 100.0)


def physics_randomization(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = _action(env)
    robot = term._asset
    if not hasattr(env, "_infantry_base_id"):
        body_ids, _ = robot.find_bodies(["base_link"], preserve_order=True)
        env._infantry_base_id = body_ids[0]

    # ObservationManager calls every term once to infer its dimension before
    # startup events run.  Return only a correctly-shaped placeholder then;
    # caching at this point would freeze the pre-randomization properties.
    if not hasattr(env, "_infantry_com_offset"):
        return torch.zeros(env.scene.num_envs, 16, device=env.device)

    if not hasattr(env, "_infantry_privileged_cache"):
        base_id = env._infantry_base_id
        masses = robot.root_physx_view.get_masses()[:, base_id].to(env.device)
        materials = robot.root_physx_view.get_material_properties().to(env.device).mean(dim=1)
        env._infantry_mass_offset = (masses - robot.data.default_mass[:, base_id].to(env.device)).unsqueeze(-1)
        env._infantry_material = materials[:, (0, 2)]
        env._infantry_privileged_cache = True
    # Reference convention: sampled CoM offset, not absolute asset CoM.
    com = env._infantry_com_offset
    default_offset = robot.data.default_joint_pos[:, torch.tensor(
        [*term._leg_ids[:2], *term._wheel_ids[:1], *term._leg_ids[2:], *term._wheel_ids[1:]],
        device=env.device,
    )]
    return torch.nan_to_num(torch.cat(
        (
            env._infantry_mass_offset,
            com,
            default_offset,
            env._infantry_material,
            term.kp_scale,
            term.kd_scale,
            term.motor_scale,
            term.action_delay_steps.float().unsqueeze(-1) * env.physics_dt,
        ),
        dim=-1,
    ))
