"""Checkpoint-compatible flat-to-terrain training route for infantry_2027_v1.

The policy observation remains the verified 5x25 history.  Both stages expose
the same 145-D privileged observation to the critic, so the complete flat
checkpoint can be resumed on terrain without rebuilding either network.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from infantry_2027.assets import ACTIVE_LEG_JOINTS, CONTROLLED_JOINTS
from infantry_2027.terrains.fudan_plane import (
    HORIZONTAL_SCALE,
    SLOPE_THRESHOLD,
    TILE_SIZE,
    VERTICAL_SCALE,
    FudanSubTerrainCfg,
    FudanTerrainGenerator,
)

from . import mdp
from .infantry_2027_env_cfg import Infantry2027FlatEnvCfg, Infantry2027SceneCfg
from .infantry_2027_terrain_env_cfg import (
    TERRAIN_PENALIZED_CONTACT_BODIES,
    Infantry2027TerrainEnvCfg,
    Infantry2027TerrainSceneCfg,
    TerrainCurriculumCfg,
    TerrainEventCfg,
    TerrainObservationsCfg,
)


EQUILIBRIUM_LENGTH_NODES = (0.16, 0.22, 0.28, 0.33)
# Dynamic MuJoCo sweep with a horizontal/attitude measurement rig.  Keeping
# these small measured values avoids encoding the invalid -0.04 rad geometric
# estimate that was obtained without a stable free-body controller.
EQUILIBRIUM_ANGLE_NODES = (0.0, 0.0, -0.005, -0.005)

FLAT_COMPATIBLE_TERRAIN_CFG = TerrainGeneratorCfg(
    class_type=FudanTerrainGenerator,
    seed=1,
    curriculum=False,
    size=(TILE_SIZE, TILE_SIZE),
    border_width=25.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=HORIZONTAL_SCALE,
    vertical_scale=VERTICAL_SCALE,
    slope_threshold=SLOPE_THRESHOLD,
    difficulty_range=(0.0, 0.0),
    color_scheme="none",
    use_cache=False,
    sub_terrains={"flat": FudanSubTerrainCfg(proportion=1.0, variant="flat")},
)


@configclass
class FudanFinalRewardsCfg:
    """Reward set actually active in Fudan's final 50/20/10/10/10 run."""

    tracking_lin_vel = RewTerm(
        func=mdp.tracking_lin_vel,
        weight=1.0,
        params={"command_name": "motion", "multiplier": 1.3},
    )
    tracking_lin_vel_enhance = RewTerm(
        func=mdp.tracking_lin_vel_enhance,
        weight=1.0,
        params={"command_name": "motion", "multiplier": 1.45},
    )
    tracking_ang_vel = RewTerm(
        func=mdp.tracking_ang_vel, weight=1.0, params={"command_name": "motion"}
    )
    base_height = RewTerm(
        func=mdp.base_height,
        weight=1.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "multiplier": 1.5,
        },
    )
    nominal_state = RewTerm(func=mdp.nominal_state, weight=-0.1, params={"maximum": 10.0})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z, weight=-0.1, params={"maximum": 10.0})
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy, weight=-0.05, params={"maximum": 20.0})
    orientation = RewTerm(func=mdp.orientation, weight=-10.0, params={"maximum": 0.1})
    dof_vel = RewTerm(
        func=mdp.dof_vel,
        weight=-1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_LEG_JOINTS)),
            "maximum": 1.0e6,
        },
    )
    dof_acc = RewTerm(
        func=mdp.dof_acc,
        weight=-1.0e-8,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(CONTROLLED_JOINTS)),
            "maximum": 1.0e8,
        },
    )
    torques = RewTerm(func=mdp.torques, weight=-1.0e-5, params={"maximum": 1.0e5})
    action_rate = RewTerm(
        func=mdp.action_rate, weight=-0.003, params={"maximum": 333.333333}
    )
    action_smooth = RewTerm(
        func=mdp.action_smooth, weight=-0.003, params={"maximum": 333.333333}
    )
    collision = RewTerm(
        func=mdp.collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=TERRAIN_PENALIZED_CONTACT_BODIES
            ),
            "threshold": 0.1,
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.dof_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lf1_joint", "rf1_joint"])},
    )


@configclass
class FlatCompatibleSceneCfg(Infantry2027SceneCfg):
    ground = None
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_COMPATIBLE_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.5,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.14, 0.15)),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 0.6)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class FlatCompatibleCommandsCfg:
    motion = mdp.DirectYawVelocityHeightCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0)
    )


def _set_v1_action_contract(cfg) -> None:
    cfg.actions.vmc.equilibrium_length_nodes = EQUILIBRIUM_LENGTH_NODES
    cfg.actions.vmc.equilibrium_angle_nodes = EQUILIBRIUM_ANGLE_NODES


def _disable_randomization(cfg) -> None:
    cfg.observations.policy.proprioception.params["noise_scale"] = 0.0
    cfg.actions.vmc.action_delay_steps_range = (0, 0)
    cfg.actions.vmc.kp_scale_range = (1.0, 1.0)
    cfg.actions.vmc.kd_scale_range = (1.0, 1.0)
    cfg.actions.vmc.motor_scale_range = (1.0, 1.0)
    if hasattr(cfg.events, "disturbance_type"):
        cfg.events.disturbance_type = None
    if hasattr(cfg.events, "interval_disturbance"):
        cfg.events.interval_disturbance = None
    cfg.events.material.params.update(
        {
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
        }
    )
    cfg.events.base_mass.params["mass_distribution_params"] = (0.0, 0.0)
    cfg.events.base_inertia.params["scale_range"] = (1.0, 1.0)
    cfg.events.base_com.params["com_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }
    cfg.events.default_joint_pos.params["offset_range"] = (0.0, 0.0)
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }


@configclass
class Infantry2027FlatCompatibleEnvCfg(Infantry2027FlatEnvCfg):
    scene: FlatCompatibleSceneCfg = FlatCompatibleSceneCfg(num_envs=4096, env_spacing=3.0)
    commands: FlatCompatibleCommandsCfg = FlatCompatibleCommandsCfg()
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    rewards: FudanFinalRewardsCfg = FudanFinalRewardsCfg()
    # Full final DR is already active on flat, avoiding a second dynamics shift
    # at the terrain hand-off.
    events: TerrainEventCfg = TerrainEventCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_v1_action_contract(self)
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class Infantry2027FlatCompatiblePlayEnvCfg(Infantry2027FlatCompatibleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.curriculum_steps = 0
        _disable_randomization(self)


@configclass
class TerrainV1CommandsCfg:
    motion = mdp.TerrainTraversalCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0)
    )


@configclass
class Infantry2027TerrainV1EnvCfg(Infantry2027TerrainEnvCfg):
    scene: Infantry2027TerrainSceneCfg = Infantry2027TerrainSceneCfg(
        num_envs=1024, env_spacing=3.0
    )
    commands: TerrainV1CommandsCfg = TerrainV1CommandsCfg()
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    rewards: FudanFinalRewardsCfg = FudanFinalRewardsCfg()
    events: TerrainEventCfg = TerrainEventCfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_v1_action_contract(self)
        # A mature flat policy starts at the easiest terrain row and earns its
        # way upward; the reference's level 5 is appropriate only when its own
        # already-mature checkpoint is resumed.
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class Infantry2027TerrainV1PlayEnvCfg(Infantry2027TerrainV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.curriculum_steps = 0
        _disable_randomization(self)
