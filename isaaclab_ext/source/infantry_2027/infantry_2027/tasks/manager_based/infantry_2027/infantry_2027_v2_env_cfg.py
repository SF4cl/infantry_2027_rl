"""Stable VMC flat training with terrain-compatible privileged observations."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from . import mdp
from .infantry_2027_env_cfg import EventCfg, Infantry2027FlatEnvCfg, RewardsCfg
from .infantry_2027_terrain_env_cfg import TerrainObservationsCfg
from .infantry_2027_v1_env_cfg import (
    FlatCompatibleSceneCfg,
    _disable_randomization,
    _set_v1_action_contract,
)


@configclass
class StableFlatCommandsCfg:
    """Use the verified five-second command duration and existing final curriculum."""

    motion = mdp.DirectYawVelocityHeightCommandCfg(
        asset_name="robot", resampling_time_range=(5.0, 5.0)
    )


@configclass
class Infantry2027FlatStableEnvCfg(Infantry2027FlatEnvCfg):
    """Flat v2: verified VMC regularization with a 145-D terrain-compatible critic.

    The actor, estimator, action contract, command envelope, and generated flat
    terrain remain checkpoint-compatible with the v1 terrain route.  Only the
    behavior-shaping configuration returns to the smooth flat baseline:
    VMC-scaled rewards, moderate domain randomization, and no interval pushes.
    """

    scene: FlatCompatibleSceneCfg = FlatCompatibleSceneCfg(num_envs=4096, env_spacing=3.0)
    commands: StableFlatCommandsCfg = StableFlatCommandsCfg()
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_v1_action_contract(self)
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        # Match the simulator-wide material used by the verified flat run.
        # The generated ground keeps its explicit 0.5/multiply material, while
        # unspecified robot materials retain this average/zero-restitution base.
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="average",
        )


@configclass
class Infantry2027FlatStablePlayEnvCfg(Infantry2027FlatStableEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.curriculum_steps = 0
        _disable_randomization(self)
