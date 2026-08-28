"""Direct-joint terrain task aligned with Fudan's final terrain snapshot."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .infantry_2027_env_cfg import Infantry2027FlatEnvCfg
from .infantry_2027_terrain_env_cfg import (
    Infantry2027TerrainSceneCfg,
    TerrainEventCfg,
    TerrainObservationsCfg,
)
from .infantry_2027_v1_env_cfg import FudanFinalRewardsCfg
from .infantry_2027_v3_env_cfg import JointPdActionsCfg


@configclass
class FudanTerrainCommandsCfg:
    """Uniform per-environment commands and curriculum from Fudan."""

    motion = mdp.FudanTerrainCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
    )


@configclass
class FudanTerrainCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.fudan_terrain_and_command_curriculum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "move_up_distance": 4.0,
            "move_down_tracking_threshold": 0.4,
            "grow_tracking_threshold": 0.7,
        },
    )


@configclass
class Infantry2027JointFudanTerrainEnvCfg(Infantry2027FlatEnvCfg):
    """v4: Fudan final mixed-terrain training with the verified new asset.

    The terrain mix, difficulty initialization and curriculum, command
    curriculum, observations, direct motor actions, rewards, randomization,
    terminations and PPO contract match the final 50/20/10/10/10 reference
    snapshot.  Asset geometry, safe height floor and the equivalent 100 Hz
    control timing remain specific to ``infantry_2027_v0``.
    """

    scene: Infantry2027TerrainSceneCfg = Infantry2027TerrainSceneCfg(
        num_envs=4096, env_spacing=3.0
    )
    commands: FudanTerrainCommandsCfg = FudanTerrainCommandsCfg()
    actions: JointPdActionsCfg = JointPdActionsCfg()
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    rewards: FudanFinalRewardsCfg = FudanFinalRewardsCfg()
    events: TerrainEventCfg = TerrainEventCfg()
    curriculum: FudanTerrainCurriculumCfg = FudanTerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Fudan samples initial difficulty uniformly from rows 0 through 5.
        self.scene.terrain.max_init_terrain_level = 5
        # Its custom-terrain reset jitters the base by +/-1 m within the tile.
        self.events.reset_base.params["pose_range"].update(
            {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class Infantry2027JointFudanTerrainPlayEnvCfg(Infantry2027JointFudanTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.proprioception.params["noise_scale"] = 0.0
        self.actions.joint_pd.action_delay_steps_range = (0, 0)
        self.actions.joint_pd.kp_scale_range = (1.0, 1.0)
        self.actions.joint_pd.kd_scale_range = (1.0, 1.0)
        self.actions.joint_pd.motor_scale_range = (1.0, 1.0)
        self.events.disturbance_type = None
        self.events.interval_disturbance = None
        self.events.material.params.update(
            {
                "static_friction_range": (1.0, 1.0),
                "dynamic_friction_range": (1.0, 1.0),
                "restitution_range": (0.0, 0.0),
            }
        )
        self.events.base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.events.base_inertia.params["scale_range"] = (1.0, 1.0)
        self.events.base_com.params["com_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
        self.events.default_joint_pos.params["offset_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"].update(
            {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        )
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
