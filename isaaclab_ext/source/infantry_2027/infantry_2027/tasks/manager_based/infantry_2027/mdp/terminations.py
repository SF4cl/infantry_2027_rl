"""Failure persistence matching the one-second Fudan terminal delay."""

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, TerminationTermCfg


class sustained_bad_orientation(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.counter[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_projected_gravity_z: float,
        failure_time_s: float,
    ) -> torch.Tensor:
        robot: Articulation = env.scene["robot"]
        failed = robot.data.projected_gravity_b[:, 2] > minimum_projected_gravity_z
        self.counter = torch.where(failed, self.counter + 1, torch.zeros_like(self.counter))
        return self.counter >= max(1, round(failure_time_s / env.step_dt))
