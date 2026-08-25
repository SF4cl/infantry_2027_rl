"""The flat-ground reward set and weights from ref/fudan_rl_wheel_leg."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import rewards as base_rewards
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .actions import VmcWheelAction
from .observations import local_base_height


def _cmd(env, name):
    return env.command_manager.get_command(name)


def _finite(value, maximum=1.0):
    return torch.nan_to_num(value, nan=maximum, posinf=maximum, neginf=maximum).clamp_max(maximum)


def tracking_lin_vel(env: ManagerBasedRLEnv, command_name: str, multiplier: float = 1.0) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    error = (_cmd(env, command_name)[:, 0] - robot.data.root_lin_vel_b[:, 0]).square()
    return torch.nan_to_num(torch.exp(-error / 0.25) * multiplier, nan=0.0).clamp(-1.0, 1.0)


def tracking_lin_vel_enhance(env: ManagerBasedRLEnv, command_name: str, multiplier: float = 1.0) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    error = (_cmd(env, command_name)[:, 0] - robot.data.root_lin_vel_b[:, 0]).square()
    return torch.nan_to_num(
        (torch.exp(-error / 2.5) - 1.0) * multiplier, nan=-multiplier
    ).clamp(-1.0, 1.0)


def tracking_ang_vel(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    error = (_cmd(env, command_name)[:, 1] - robot.data.root_ang_vel_b[:, 2]).square()
    return torch.nan_to_num(torch.exp(-error / 0.25), nan=0.0)


def tracking_ang_vel_enhance(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    error = (_cmd(env, command_name)[:, 1] - robot.data.root_ang_vel_b[:, 2]).square()
    return torch.nan_to_num(torch.exp(-error / 2.5) - 1.0, nan=-1.0)


def base_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    multiplier: float = 1.0,
) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    measured_height = (
        robot.data.root_pos_w[:, 2]
        if sensor_cfg is None
        else local_base_height(env, sensor_cfg)
    )
    error = (_cmd(env, command_name)[:, 2] - measured_height).square()
    return torch.nan_to_num(torch.exp(-error / 0.001) * multiplier, nan=0.0).clamp(-1.0, 1.0)


def base_height_enhance(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    error = (_cmd(env, command_name)[:, 2] - local_base_height(env, sensor_cfg)).square()
    return torch.nan_to_num(torch.exp(-error / 0.01) - 1.0, nan=-1.0)


def wheel_air_leg_angle(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Encourage a neutral VMC swing angle only while a wheel is airborne."""
    term: VmcWheelAction = env.action_manager.get_term("vmc")
    term.update_leg_state()
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    wheel_contact = forces.norm(dim=-1).max(dim=1).values > threshold
    reward = torch.exp(-term.leg_angle.square() / 0.25) * (~wheel_contact).float()
    return torch.nan_to_num(reward.sum(dim=1), nan=0.0)


def nominal_state(env: ManagerBasedRLEnv, maximum: float = 1.0) -> torch.Tensor:
    term: VmcWheelAction = env.action_manager.get_term("vmc")
    term.update_leg_state()
    return _finite((term.leg_angle[:, 0] - term.leg_angle[:, 1]).square(), maximum)


def lin_vel_z(env: ManagerBasedRLEnv, maximum: float = 1.0) -> torch.Tensor:
    return _finite(base_rewards.lin_vel_z_l2(env), maximum)


def ang_vel_xy(env: ManagerBasedRLEnv, maximum: float = 5.0) -> torch.Tensor:
    return _finite(base_rewards.ang_vel_xy_l2(env), maximum)


def orientation(env: ManagerBasedRLEnv, maximum: float = 0.01) -> torch.Tensor:
    return _finite(base_rewards.flat_orientation_l2(env), maximum)


def dof_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, maximum: float = 20_000.0
) -> torch.Tensor:
    return _finite(base_rewards.joint_vel_l2(env, asset_cfg), maximum)


def dof_acc(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, maximum: float = 4_000_000.0
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return _finite(robot.data.joint_acc[:, asset_cfg.joint_ids].square().sum(dim=-1), maximum)


def torques(env: ManagerBasedRLEnv, maximum: float = 10_000.0) -> torch.Tensor:
    term: VmcWheelAction = env.action_manager.get_term("vmc")
    value = term.last_leg_effort.square().sum(dim=-1) + term.last_wheel_effort.square().sum(dim=-1)
    return _finite(value, maximum)


def action_rate(env: ManagerBasedRLEnv, maximum: float = 100.0) -> torch.Tensor:
    return _finite(base_rewards.action_rate_l2(env), maximum)


def action_smooth(env: ManagerBasedRLEnv, maximum: float = 100.0) -> torch.Tensor:
    term: VmcWheelAction = env.action_manager.get_term("vmc")
    return _finite(
        term.action_second_difference[:, (0, 1, 3, 4)].square().sum(dim=-1), maximum
    )


def collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    return _finite(base_rewards.undesired_contacts(env, threshold, sensor_cfg), 1.0)


def dof_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _finite(base_rewards.joint_pos_limits(env, asset_cfg), 1.0)


def non_finite_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return ~(
        torch.isfinite(robot.data.root_state_w).all(dim=-1)
        & torch.isfinite(robot.data.joint_pos).all(dim=-1)
        & torch.isfinite(robot.data.joint_vel).all(dim=-1)
    )
