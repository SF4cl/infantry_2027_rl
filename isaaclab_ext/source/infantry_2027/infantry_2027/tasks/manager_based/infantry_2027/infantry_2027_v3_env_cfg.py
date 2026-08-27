"""From-scratch direct-joint training on Fudan's final mixed terrains."""

from __future__ import annotations

from isaaclab.utils import configclass

from . import mdp
from .infantry_2027_env_cfg import Infantry2027FlatEnvCfg
from .infantry_2027_terrain_env_cfg import (
    Infantry2027TerrainSceneCfg,
    TerrainCurriculumCfg,
    TerrainEventCfg,
    TerrainObservationsCfg,
)
from .infantry_2027_v1_env_cfg import FudanFinalRewardsCfg


@configclass
class JointPdActionsCfg:
    """Six direct motor commands in Fudan's left-leg/wheel/right-leg order."""

    joint_pd = mdp.JointPdWheelActionCfg(asset_name="robot")


@configclass
class JointTerrainCommandsCfg:
    """Mixed-terrain command curriculum with the requested final speed."""

    motion = mdp.TerrainTraversalCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        flat_terrain_types=tuple(range(10)),
    )
    motion.ranges.forward_max = 2.5


@configclass
class Infantry2027JointTerrainEnvCfg(Infantry2027FlatEnvCfg):
    """v3: direct joint PD, full DR and mixed terrain from random weights.

    This task deliberately has no recovery reset distribution.  Episodes start
    near the verified upright pose and terminate after a sustained fall.  Flat
    terrain occupies half of the environments, while slopes and stairs use the
    reference difficulty curriculum from level zero.
    """

    scene: Infantry2027TerrainSceneCfg = Infantry2027TerrainSceneCfg(
        num_envs=4096, env_spacing=3.0
    )
    commands: JointTerrainCommandsCfg = JointTerrainCommandsCfg()
    actions: JointPdActionsCfg = JointPdActionsCfg()
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    rewards: FudanFinalRewardsCfg = FudanFinalRewardsCfg()
    events: TerrainEventCfg = TerrainEventCfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # From-scratch policies begin on difficulty 0 and earn higher rows.
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class Infantry2027JointTerrainPlayEnvCfg(Infantry2027JointTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.curriculum_steps = 0
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
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
