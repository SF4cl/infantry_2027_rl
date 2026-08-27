"""Closed-chain VMC and direct joint-PD wheel-legged actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from infantry_2027.assets import ACTIVE_LEG_JOINTS, WHEEL_AXIS_SIGNS, WHEEL_JOINTS
from infantry_2027.vmc import PHI1_OFFSET, PHI4_OFFSET, TorchFiveBarVmc, wrap_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VmcWheelAction(ActionTerm):
    """Action order: left angle/length/wheel, right angle/length/wheel."""

    cfg: "VmcWheelActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "VmcWheelActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._leg_ids, names = self._asset.find_joints(list(ACTIVE_LEG_JOINTS), preserve_order=True)
        if names != list(ACTIVE_LEG_JOINTS):
            raise ValueError(f"Active VMC joint order mismatch: {names}")
        self._wheel_ids, names = self._asset.find_joints(list(WHEEL_JOINTS), preserve_order=True)
        if names != list(WHEEL_JOINTS):
            raise ValueError(f"Wheel joint order mismatch: {names}")
        self._side_signs = torch.tensor((1.0, -1.0), device=self.device)
        self._wheel_signs = torch.tensor(WHEEL_AXIS_SIGNS, device=self.device)
        self._solver = TorchFiveBarVmc(self.num_envs * 2, self.device)
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._previous_action = torch.zeros_like(self._raw_actions)
        self._previous_previous_action = torch.zeros_like(self._raw_actions)
        self._fifo = torch.zeros(self.num_envs, cfg.max_action_delay_steps + 1, 6, device=self.device)
        self.action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.kp_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.kd_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.motor_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.leg_length = torch.zeros(self.num_envs, 2, device=self.device)
        self.leg_length_rate = torch.zeros_like(self.leg_length)
        self.leg_angle = torch.zeros_like(self.leg_length)
        self.leg_angle_rate = torch.zeros_like(self.leg_length)
        self.singular = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.last_leg_effort = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_wheel_effort = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def previous_action(self) -> torch.Tensor:
        return self._previous_action

    @property
    def previous_previous_action(self) -> torch.Tensor:
        return self._previous_previous_action

    @property
    def action_second_difference(self) -> torch.Tensor:
        return self._raw_actions - 2.0 * self._previous_action + self._previous_previous_action

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        count = self.num_envs if env_ids is None else len(env_ids)
        for value in (
            self._raw_actions, self._processed_actions, self._previous_action,
            self._previous_previous_action, self._fifo, self.last_leg_effort, self.last_wheel_effort,
        ):
            value[ids] = 0.0
        lo, hi = self.cfg.action_delay_steps_range
        self.action_delay_steps[ids] = torch.randint(lo, hi + 1, (count,), device=self.device)
        self.kp_scale[ids] = torch.empty(count, 1, device=self.device).uniform_(*self.cfg.kp_scale_range)
        self.kd_scale[ids] = torch.empty(count, 1, device=self.device).uniform_(*self.cfg.kd_scale_range)
        self.motor_scale[ids] = torch.empty(count, 1, device=self.device).uniform_(*self.cfg.motor_scale_range)

    def process_actions(self, actions: torch.Tensor) -> None:
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Policy produced non-finite VMC actions.")
        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(self._raw_actions)
        self._raw_actions.copy_(actions.clamp(-self.cfg.policy_action_clip, self.cfg.policy_action_clip))

    def update_leg_state(self):
        q = self._asset.data.joint_pos[:, self._leg_ids].reshape(self.num_envs, 2, 2)
        qd = self._asset.data.joint_vel[:, self._leg_ids].reshape(self.num_envs, 2, 2)
        canonical_q = q * self._side_signs.view(1, 2, 1)
        canonical_qd = qd * self._side_signs.view(1, 2, 1)
        out = self._solver.update(
            (canonical_q[:, :, 0] + PHI1_OFFSET).reshape(-1),
            (canonical_q[:, :, 1] + PHI4_OFFSET).reshape(-1),
            canonical_qd[:, :, 0].reshape(-1),
            canonical_qd[:, :, 1].reshape(-1),
        )
        self.leg_length.copy_(torch.nan_to_num(out.length.reshape(self.num_envs, 2)))
        self.leg_length_rate.copy_(torch.nan_to_num(out.length_rate.reshape(self.num_envs, 2)))
        self.leg_angle.copy_(torch.nan_to_num(out.angle.reshape(self.num_envs, 2)))
        self.leg_angle_rate.copy_(torch.nan_to_num(out.angle_rate.reshape(self.num_envs, 2)))
        self.singular.copy_(out.singular.reshape(self.num_envs, 2))
        return out

    def _decode_delayed_action(self) -> None:
        self._fifo[:, 1:].copy_(self._fifo[:, :-1].clone())
        self._fifo[:, 0].copy_(self._raw_actions)
        ids = torch.arange(self.num_envs, device=self.device)
        action = self._fifo[ids, self.action_delay_steps]
        self._processed_actions[:, (0, 3)] = action[:, (0, 3)] * self.cfg.angle_scale
        # Base height is the third Fudan command.  The calibrated upright
        # geometry is height = leg_length + height_offset.
        height = self._env.command_manager.get_command(self.cfg.command_name)[:, 2:3]
        nominal_length = height - self.cfg.height_offset
        self._processed_actions[:, (1, 4)] = nominal_length + torch.tanh(action[:, (1, 4)]) * self.cfg.length_residual_scale
        self._processed_actions[:, (1, 4)].clamp_(*self.cfg.length_limits)
        if self.cfg.equilibrium_angle_nodes:
            lengths = self._processed_actions[:, (1, 4)]
            length_nodes = lengths.new_tensor(self.cfg.equilibrium_length_nodes)
            angle_nodes = lengths.new_tensor(self.cfg.equilibrium_angle_nodes)
            if length_nodes.numel() != angle_nodes.numel() or length_nodes.numel() < 2:
                raise ValueError("Equilibrium length/angle tables must contain the same number of nodes.")
            upper = torch.searchsorted(length_nodes, lengths).clamp(1, length_nodes.numel() - 1)
            lower = upper - 1
            alpha = (lengths - length_nodes[lower]) / (length_nodes[upper] - length_nodes[lower])
            bias = angle_nodes[lower] + alpha * (angle_nodes[upper] - angle_nodes[lower])
            self._processed_actions[:, (0, 3)] += bias
        self._processed_actions[:, (2, 5)] = action[:, (2, 5)] * self.cfg.wheel_velocity_scale
        self._processed_actions[:, (2, 5)].clamp_(-self.cfg.wheel_velocity_limit, self.cfg.wheel_velocity_limit)

    def apply_actions(self) -> None:
        self._decode_delayed_action()
        out = self.update_leg_state()
        target_angle = self._processed_actions[:, (0, 3)]
        target_length = self._processed_actions[:, (1, 4)]
        force = torch.stack(
            (
                self.cfg.kp_length * self.kp_scale * (target_length - self.leg_length)
                - self.cfg.kd_length * self.kd_scale * self.leg_length_rate
                + self.cfg.support_force,
                self.cfg.kp_angle * wrap_angle(target_angle - self.leg_angle)
                - self.cfg.kd_angle * self.leg_angle_rate,
            ),
            dim=-1,
        )
        torque = self._solver.force_to_torque(out.jacobian, torch.nan_to_num(force).reshape(-1, 2))
        torque = torque.reshape(self.num_envs, 2, 2) * self._side_signs.view(1, 2, 1)
        effort = torque.reshape(self.num_envs, 4) * self.motor_scale
        effort.clamp_(-self.cfg.leg_effort_limit, self.cfg.leg_effort_limit)
        effort[self.singular.repeat_interleave(2, dim=1)] = 0.0
        wheel_target = self._processed_actions[:, (2, 5)] * self._wheel_signs
        wheel_vel = self._asset.data.joint_vel[:, self._wheel_ids]
        wheel_effort = self.cfg.wheel_kd * (wheel_target - wheel_vel) * self.motor_scale
        wheel_effort.clamp_(-self.cfg.wheel_effort_limit, self.cfg.wheel_effort_limit)
        self.last_leg_effort.copy_(torch.nan_to_num(effort))
        self.last_wheel_effort.copy_(torch.nan_to_num(wheel_effort))
        self._asset.set_joint_effort_target(self.last_leg_effort, joint_ids=self._leg_ids)
        self._asset.set_joint_effort_target(self.last_wheel_effort, joint_ids=self._wheel_ids)


@configclass
class VmcWheelActionCfg(ActionTermCfg):
    class_type: type = VmcWheelAction
    asset_name: str = MISSING
    command_name: str = "motion"
    angle_scale: float = 0.5
    length_residual_scale: float = 0.03
    length_limits: tuple[float, float] = (0.16, 0.33)
    height_offset: float = -0.0120
    wheel_velocity_scale: float = 20.0
    wheel_velocity_limit: float = 55.0
    policy_action_clip: float = 100.0
    kp_length: float = 900.0
    kd_length: float = 20.0
    kp_angle: float = 50.0
    kd_angle: float = 3.0
    support_force: float = 118.88
    wheel_kd: float = 1.0
    leg_effort_limit: float = 45.0
    wheel_effort_limit: float = 5.0
    max_action_delay_steps: int = 5
    action_delay_steps_range: tuple[int, int] = (0, 5)
    kp_scale_range: tuple[float, float] = (0.95, 1.05)
    kd_scale_range: tuple[float, float] = (0.95, 1.05)
    motor_scale_range: tuple[float, float] = (0.95, 1.05)
    # Empty by default so existing v0 checkpoints retain their exact action
    # contract.  v1 enables the independently measured MuJoCo table.
    equilibrium_length_nodes: tuple[float, ...] = ()
    equilibrium_angle_nodes: tuple[float, ...] = ()


class JointPdWheelAction(VmcWheelAction):
    """Fudan-compatible motor action without VMC force control.

    The six policy values retain Fudan's physical order::

        [left_front, left_rear, left_wheel,
         right_front, right_rear, right_wheel]

    Leg values are position residuals about the randomized default joint
    positions.  Wheel values are joint-axis velocity targets.  The five-bar
    solver inherited from :class:`VmcWheelAction` is used only to measure the
    physical leg swing angles for the reference ``nominal_state`` reward; it
    never participates in the torque command.
    """

    cfg: "JointPdWheelActionCfg"

    def _decode_delayed_action(self) -> None:
        self._fifo[:, 1:].copy_(self._fifo[:, :-1].clone())
        self._fifo[:, 0].copy_(self._raw_actions)
        ids = torch.arange(self.num_envs, device=self.device)
        action = self._fifo[ids, self.action_delay_steps]

        leg_action = action[:, (0, 1, 3, 4)]
        default_leg_pos = self._asset.data.default_joint_pos[:, self._leg_ids]
        self._processed_actions[:, (0, 1, 3, 4)] = (
            default_leg_pos + leg_action * self.cfg.position_scale
        )
        self._processed_actions[:, (2, 5)] = (
            action[:, (2, 5)] * self.cfg.wheel_velocity_scale
        ).clamp(-self.cfg.wheel_velocity_limit, self.cfg.wheel_velocity_limit)

    def apply_actions(self) -> None:
        self._decode_delayed_action()

        leg_target = self._processed_actions[:, (0, 1, 3, 4)]
        leg_pos = self._asset.data.joint_pos[:, self._leg_ids]
        leg_vel = self._asset.data.joint_vel[:, self._leg_ids]
        leg_effort = (
            self.cfg.leg_kp * self.kp_scale * (leg_target - leg_pos)
            - self.cfg.leg_kd * self.kd_scale * leg_vel
        ) * self.motor_scale
        leg_effort.clamp_(-self.cfg.leg_effort_limit, self.cfg.leg_effort_limit)

        wheel_target = self._processed_actions[:, (2, 5)]
        wheel_vel = self._asset.data.joint_vel[:, self._wheel_ids]
        wheel_effort = (
            self.cfg.wheel_kd * self.kd_scale * (wheel_target - wheel_vel)
        ) * self.motor_scale
        wheel_effort.clamp_(-self.cfg.wheel_effort_limit, self.cfg.wheel_effort_limit)

        self.last_leg_effort.copy_(torch.nan_to_num(leg_effort))
        self.last_wheel_effort.copy_(torch.nan_to_num(wheel_effort))
        self._asset.set_joint_effort_target(self.last_leg_effort, joint_ids=self._leg_ids)
        self._asset.set_joint_effort_target(self.last_wheel_effort, joint_ids=self._wheel_ids)


@configclass
class JointPdWheelActionCfg(ActionTermCfg):
    """Direct motor-PD parameters from Fudan's final terrain snapshot."""

    class_type: type = JointPdWheelAction
    asset_name: str = MISSING
    position_scale: float = 0.5
    wheel_velocity_scale: float = 10.0
    wheel_velocity_limit: float = 60.0
    policy_action_clip: float = 100.0
    leg_kp: float = 60.0
    leg_kd: float = 1.0
    wheel_kd: float = 0.2
    leg_effort_limit: float = 45.0
    wheel_effort_limit: float = 5.0
    max_action_delay_steps: int = 5
    action_delay_steps_range: tuple[int, int] = (0, 5)
    kp_scale_range: tuple[float, float] = (0.9, 1.1)
    kd_scale_range: tuple[float, float] = (0.9, 1.1)
    motor_scale_range: tuple[float, float] = (0.9, 1.1)
