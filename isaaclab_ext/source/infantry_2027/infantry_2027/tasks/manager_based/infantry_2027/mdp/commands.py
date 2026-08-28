"""Direct yaw-rate, velocity, and asset-calibrated base-height command."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class DirectYawVelocityHeightCommand(CommandTerm):
    """Actor command is ``vx, direct yaw-rate, base height``.

    One from-scratch run continuously expands three command modes: stationary
    stance, pure point turns, and translating turns.  This keeps point-turn
    support explicit rather than hoping a continuous velocity sampler lands
    sufficiently close to zero.
    """

    cfg: "DirectYawVelocityHeightCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        if cfg.standing_probability < 0.0 or cfg.point_turn_probability < 0.0:
            raise ValueError("Command mode probabilities must be non-negative.")
        if cfg.standing_probability + cfg.point_turn_probability > 1.0:
            raise ValueError("Standing and point-turn probabilities must sum to at most one.")
        self.metrics["error_forward_velocity"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["positive_forward_error_sum"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["positive_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["negative_forward_error_sum"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["negative_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["standing_forward_error_sum"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["standing_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def set_manual_command(
        self,
        forward_velocity: torch.Tensor,
        yaw_rate: torch.Tensor,
        base_height: torch.Tensor,
    ) -> None:
        """Apply a bounded manual ``vx, yaw-rate, base-height`` command."""
        forward = forward_velocity.to(device=self.device).clamp(
            -self.cfg.ranges.forward_max, self.cfg.ranges.forward_max
        )
        self._command[:, 0].copy_(forward)
        yaw_limit = torch.where(
            forward.abs() < self.cfg.minimum_moving_speed,
            torch.full_like(forward, self.cfg.point_yaw_rate_limit),
            torch.full_like(forward, self.cfg.moving_yaw_rate_limit),
        )
        self._command[:, 1].copy_(yaw_rate.to(device=self.device).clamp(-yaw_limit, yaw_limit))
        self._command[:, 2].copy_(
            base_height.to(device=self.device).clamp(*self.cfg.ranges.base_height)
        )

    def _progress(self) -> float:
        if self.cfg.curriculum_steps <= 0:
            return 1.0
        return min(1.0, float(self._env.common_step_counter) / self.cfg.curriculum_steps)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = ids.numel()
        progress = self._progress()
        vx_max = self.cfg.initial_forward_max + progress * (
            self.cfg.ranges.forward_max - self.cfg.initial_forward_max
        )
        height_low = self.cfg.initial_height_range[0] + progress * (
            self.cfg.ranges.base_height[0] - self.cfg.initial_height_range[0]
        )
        height_high = self.cfg.initial_height_range[1] + progress * (
            self.cfg.ranges.base_height[1] - self.cfg.initial_height_range[1]
        )
        moving_yaw_limit = self.cfg.initial_moving_yaw_rate_limit + progress * (
            self.cfg.moving_yaw_rate_limit - self.cfg.initial_moving_yaw_rate_limit
        )
        point_yaw_limit = self.cfg.initial_point_yaw_rate_limit + progress * (
            self.cfg.point_yaw_rate_limit - self.cfg.initial_point_yaw_rate_limit
        )

        self._command[ids, 0] = torch.empty(count, device=self.device).uniform_(-vx_max, vx_max)
        self._command[ids, 1] = torch.empty(count, device=self.device).uniform_(
            -moving_yaw_limit, moving_yaw_limit
        )
        self._command[ids, 2] = torch.empty(count, device=self.device).uniform_(height_low, height_high)

        # Boundary samples make full-stroke height changes a regular part of
        # training instead of relying on a continuous sampler to hit an end.
        boundary = torch.rand(count, device=self.device) < self.cfg.height_boundary_probability
        if boundary.any():
            endpoints = self._command.new_tensor((height_low, height_high))
            sides = torch.randint(0, 2, (int(boundary.sum()),), device=self.device)
            self._command[ids[boundary], 2] = endpoints[sides]

        mode = torch.rand(count, device=self.device)
        standing = mode < self.cfg.standing_probability
        point_turn = (mode >= self.cfg.standing_probability) & (
            mode < self.cfg.standing_probability + self.cfg.point_turn_probability
        )
        moving = ~(standing | point_turn)

        too_slow = moving & (self._command[ids, 0].abs() < self.cfg.minimum_moving_speed)
        if too_slow.any():
            sign = torch.where(
                torch.rand(int(too_slow.sum()), device=self.device) < 0.5,
                -torch.ones(int(too_slow.sum()), device=self.device),
                torch.ones(int(too_slow.sum()), device=self.device),
            )
            self._command[ids[too_slow], 0] = sign * self.cfg.minimum_moving_speed

        if point_turn.any():
            point_ids = ids[point_turn]
            point_count = point_ids.numel()
            magnitude = torch.empty(point_count, device=self.device).uniform_(
                self.cfg.minimum_point_yaw_rate, point_yaw_limit
            )
            sign = torch.where(
                torch.rand(point_count, device=self.device) < 0.5,
                -torch.ones(point_count, device=self.device),
                torch.ones(point_count, device=self.device),
            )
            self._command[point_ids, 0] = 0.0
            self._command[point_ids, 1] = sign * magnitude

        if standing.any():
            stand_ids = ids[standing]
            self._command[stand_ids, :2] = 0.0

    def _update_command(self) -> None:
        # Direct yaw-rate commands remain constant between resampling events.
        pass

    def _update_metrics(self) -> None:
        steps = self._env.max_episode_length
        error = (self._command[:, 0] - self.robot.data.root_lin_vel_b[:, 0]).abs()
        self.metrics["error_forward_velocity"] += error / steps
        positive = self._command[:, 0] >= self.cfg.direction_metric_threshold
        negative = self._command[:, 0] <= -self.cfg.direction_metric_threshold
        standing = ~(positive | negative)
        for name, mask in (("positive", positive), ("negative", negative), ("standing", standing)):
            self.metrics[f"{name}_forward_error_sum"] += error * mask / steps
            self.metrics[f"{name}_fraction"] += mask.float() / steps
        self.metrics["error_yaw_rate"] += (
            self._command[:, 1] - self.robot.data.root_ang_vel_b[:, 2]
        ).abs() / steps
        self.metrics["error_base_height"] += (
            self._command[:, 2] - self._measured_base_height()
        ).abs() / steps

    def _measured_base_height(self) -> torch.Tensor:
        """Return the height quantity represented by the third command."""
        return self.robot.data.root_pos_w[:, 2]


@configclass
class DirectYawVelocityHeightCommandCfg(CommandTermCfg):
    class_type: type = DirectYawVelocityHeightCommand
    asset_name: str = MISSING
    moving_yaw_rate_limit: float = 4.0
    point_yaw_rate_limit: float = 10.0
    initial_moving_yaw_rate_limit: float = 1.0
    initial_point_yaw_rate_limit: float = 2.0
    minimum_point_yaw_rate: float = 0.5
    initial_forward_max: float = 0.5
    initial_height_range: tuple[float, float] = (0.188, 0.248)
    minimum_moving_speed: float = 0.15
    standing_probability: float = 0.10
    point_turn_probability: float = 0.20
    height_boundary_probability: float = 0.15
    curriculum_steps: int = 72_000
    direction_metric_threshold: float = 0.15

    @configclass
    class Ranges:
        forward_max: float = 2.3
        # Calibrated from height = L - 0.012 m, for L in [0.16, 0.33].
        base_height: tuple[float, float] = (0.148, 0.318)

    ranges: Ranges = Ranges()


class TerrainTraversalCommand(DirectYawVelocityHeightCommand):
    """Keep non-flat traversal aligned while retaining both travel directions.

    Flat columns keep the complete direct-yaw distribution.  On slopes and
    stairs, the robot receives equally likely positive and negative body-x
    commands, while yaw-rate becomes a bounded heading correction toward the
    terrain grid's x axis.  Thus reverse traversal is trained explicitly, but
    mechanically unrealistic sideways approaches are not sampled.
    """

    cfg: "TerrainTraversalCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._terrain_heading = torch.zeros(self.num_envs, device=self.device)
        self._flat_terrain_types = torch.as_tensor(
            cfg.flat_terrain_types, device=self.device, dtype=torch.long
        )

    def _nonflat_mask(self, ids: torch.Tensor) -> torch.Tensor:
        terrain = self._env.scene.terrain
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None:
            raise RuntimeError("Terrain traversal command requires TerrainImporter.terrain_types.")
        selected_types = terrain_types[ids].to(self.device)
        return ~torch.isin(selected_types, self._flat_terrain_types)

    def _measured_base_height(self) -> torch.Tensor:
        """Measure clearance over the local yaw-aligned terrain scan."""
        sensor = self._env.scene.sensors[self.cfg.height_sensor_name]
        clearance = self.robot.data.root_pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        nominal = 0.5 * sum(self.cfg.ranges.base_height)
        return torch.nan_to_num(
            clearance, nan=nominal, posinf=nominal, neginf=nominal
        ).mean(dim=1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        nonflat = self._nonflat_mask(ids)
        if not nonflat.any():
            return
        rough_ids = ids[nonflat]
        progress = self._progress()
        vx_max = self.cfg.initial_forward_max + progress * (
            self.cfg.ranges.forward_max - self.cfg.initial_forward_max
        )
        magnitude = torch.empty(rough_ids.numel(), device=self.device).uniform_(
            self.cfg.minimum_terrain_speed, max(self.cfg.minimum_terrain_speed, vx_max)
        )
        direction = torch.where(
            torch.rand(rough_ids.numel(), device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        self._command[rough_ids, 0] = direction * magnitude
        self._command[rough_ids, 1] = 0.0
        self._terrain_heading[rough_ids] = self.cfg.terrain_heading

    def _update_command(self) -> None:
        ids = torch.arange(self.num_envs, device=self.device)
        nonflat = self._nonflat_mask(ids)
        if not nonflat.any():
            return
        quat = self.robot.data.root_quat_w[nonflat]
        w, x, y, z = quat.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        error = torch.atan2(
            torch.sin(self._terrain_heading[nonflat] - yaw),
            torch.cos(self._terrain_heading[nonflat] - yaw),
        )
        self._command[nonflat, 1] = (self.cfg.heading_gain * error).clamp(
            -self.cfg.terrain_yaw_rate_limit, self.cfg.terrain_yaw_rate_limit
        )


@configclass
class TerrainTraversalCommandCfg(DirectYawVelocityHeightCommandCfg):
    class_type: type = TerrainTraversalCommand
    # FUDAN_TERRAINS_CFG allocates ten of its twenty columns to flat terrain.
    # TerrainImporter.terrain_types stores the column index, not the variant
    # index, so all ten columns must retain the flat command distribution.
    flat_terrain_types: tuple[int, ...] = tuple(range(10))
    height_sensor_name: str = "height_scanner"
    minimum_terrain_speed: float = 0.20
    terrain_heading: float = 0.0
    heading_gain: float = 1.5
    terrain_yaw_rate_limit: float = 1.0


class FudanTerrainCommand(DirectYawVelocityHeightCommand):
    """Per-environment command curriculum from Fudan's final terrain run.

    Unlike :class:`TerrainTraversalCommand`, this term does not impose a
    terrain heading or special command modes.  Every environment samples
    forward velocity, direct yaw-rate and base height uniformly from its own
    current range.  The terrain curriculum grows or shrinks those ranges when
    an environment wraps above the hardest level or fails below level zero.
    """

    cfg: "FudanTerrainCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._forward_ranges = torch.empty(self.num_envs, 2, device=self.device)
        self._yaw_ranges = torch.empty(self.num_envs, 2, device=self.device)
        self._forward_ranges[:] = self._forward_ranges.new_tensor(cfg.initial_forward_range)
        self._yaw_ranges[:] = self._yaw_ranges.new_tensor(cfg.initial_yaw_range)
        self._basic_terrain_types = torch.as_tensor(
            cfg.basic_terrain_types, device=self.device, dtype=torch.long
        )
        self.metrics["max_forward_command"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_yaw_command"] = torch.zeros(self.num_envs, device=self.device)

    def _measured_base_height(self) -> torch.Tensor:
        """Measure the commanded clearance over the local height scan."""
        sensor = self._env.scene.sensors[self.cfg.height_sensor_name]
        clearance = self.robot.data.root_pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        nominal = 0.5 * sum(self.cfg.ranges.base_height)
        return torch.nan_to_num(
            clearance, nan=nominal, posinf=nominal, neginf=nominal
        ).mean(dim=1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = ids.numel()
        unit = torch.rand(count, 3, device=self.device)
        self._command[ids, 0] = (
            self._forward_ranges[ids, 0]
            + unit[:, 0] * (self._forward_ranges[ids, 1] - self._forward_ranges[ids, 0])
        )
        self._command[ids, 1] = (
            self._yaw_ranges[ids, 0]
            + unit[:, 1] * (self._yaw_ranges[ids, 1] - self._yaw_ranges[ids, 0])
        )
        height_low, height_high = self.cfg.ranges.base_height
        self._command[ids, 2] = height_low + unit[:, 2] * (height_high - height_low)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        steps = self._env.max_episode_length
        self.metrics["max_forward_command"] += self._forward_ranges[:, 1] / steps
        self.metrics["max_yaw_command"] += self._yaw_ranges[:, 1] / steps

    def update_ranges(
        self,
        fail_ids: torch.Tensor,
        success_ids: torch.Tensor,
        basic_success_ids: torch.Tensor,
    ) -> None:
        """Apply Fudan's exact fail-shrink and success-grow increments."""
        if fail_ids.numel() > 0:
            self._forward_ranges[fail_ids, 0] = (
                self._forward_ranges[fail_ids, 0] + self.cfg.forward_shrink_step
            ).clamp(
                -self.cfg.forward_limit, -self.cfg.minimum_command_abs
            )
            self._forward_ranges[fail_ids, 1] = (
                self._forward_ranges[fail_ids, 1] - self.cfg.forward_shrink_step
            ).clamp(
                self.cfg.minimum_command_abs, self.cfg.forward_limit
            )
            self._yaw_ranges[fail_ids, 0] = (
                self._yaw_ranges[fail_ids, 0] + self.cfg.yaw_shrink_step
            ).clamp(
                -self.cfg.yaw_limit, -self.cfg.minimum_command_abs
            )
            self._yaw_ranges[fail_ids, 1] = (
                self._yaw_ranges[fail_ids, 1] - self.cfg.yaw_shrink_step
            ).clamp(
                self.cfg.minimum_command_abs, self.cfg.yaw_limit
            )

        if success_ids.numel() > 0:
            self._forward_ranges[success_ids, 0] -= self.cfg.forward_advanced_step
            self._forward_ranges[success_ids, 1] += self.cfg.forward_advanced_step
            self._yaw_ranges[success_ids, 0] -= self.cfg.yaw_advanced_step
            self._yaw_ranges[success_ids, 1] += self.cfg.yaw_advanced_step
        if basic_success_ids.numel() > 0:
            self._forward_ranges[basic_success_ids, 0] -= self.cfg.forward_basic_extra
            self._forward_ranges[basic_success_ids, 1] += self.cfg.forward_basic_extra
            self._yaw_ranges[basic_success_ids, 0] -= self.cfg.yaw_basic_extra
            self._yaw_ranges[basic_success_ids, 1] += self.cfg.yaw_basic_extra

        # Both basic and advanced maxima are equal in the final reference
        # snapshot, but keep the classification explicit to preserve its
        # curriculum contract if those limits are changed later.
        terrain_types = self._env.scene.terrain.terrain_types.to(self.device)
        basic = torch.isin(terrain_types, self._basic_terrain_types)
        self._forward_ranges[basic] = self._forward_ranges[basic].clamp(
            -self.cfg.basic_forward_limit, self.cfg.basic_forward_limit
        )
        self._forward_ranges[~basic] = self._forward_ranges[~basic].clamp(
            -self.cfg.advanced_forward_limit, self.cfg.advanced_forward_limit
        )
        self._yaw_ranges[basic] = self._yaw_ranges[basic].clamp(
            -self.cfg.basic_yaw_limit, self.cfg.basic_yaw_limit
        )
        self._yaw_ranges[~basic] = self._yaw_ranges[~basic].clamp(
            -self.cfg.advanced_yaw_limit, self.cfg.advanced_yaw_limit
        )

    def basic_success_ids(self, success_ids: torch.Tensor) -> torch.Tensor:
        """Return successful environments belonging to Fudan's basic set."""
        if success_ids.numel() == 0:
            return success_ids
        terrain_types = self._env.scene.terrain.terrain_types[success_ids].to(self.device)
        return success_ids[torch.isin(terrain_types, self._basic_terrain_types)]


@configclass
class FudanTerrainCommandCfg(DirectYawVelocityHeightCommandCfg):
    class_type: type = FudanTerrainCommand
    initial_forward_range: tuple[float, float] = (-2.0, 2.0)
    initial_yaw_range: tuple[float, float] = (-2.0, 2.0)
    basic_terrain_types: tuple[int, ...] = tuple(range(18))
    height_sensor_name: str = "height_scanner"
    minimum_command_abs: float = 1.0
    forward_shrink_step: float = 0.25
    yaw_shrink_step: float = 0.5
    forward_advanced_step: float = 0.05
    forward_basic_extra: float = 0.45
    yaw_advanced_step: float = 0.1
    yaw_basic_extra: float = 0.4
    forward_limit: float = 2.5
    yaw_limit: float = 4.0
    basic_forward_limit: float = 2.5
    advanced_forward_limit: float = 2.5
    basic_yaw_limit: float = 4.0
    advanced_yaw_limit: float = 4.0
    moving_yaw_rate_limit: float = 4.0
    point_yaw_rate_limit: float = 4.0
    standing_probability: float = 0.0
    point_turn_probability: float = 0.0
    height_boundary_probability: float = 0.0
    curriculum_steps: int = 0

    @configclass
    class Ranges:
        forward_max: float = 2.5
        # Fudan uses [0.14, 0.30].  The verified new asset cannot safely
        # represent values below 0.148 m (0.16 m physical leg length), so only
        # that unreachable lower endpoint is clipped.
        base_height: tuple[float, float] = (0.148, 0.300)

    ranges: Ranges = Ranges()
