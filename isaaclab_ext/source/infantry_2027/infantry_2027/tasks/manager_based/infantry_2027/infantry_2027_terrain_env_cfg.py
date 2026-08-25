"""Fudan-style rough-terrain VMC locomotion task for infantry_2027_v0."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from infantry_2027.assets import ACTIVE_LEG_JOINTS, CONTROLLED_JOINTS
from infantry_2027.terrains import FUDAN_TERRAINS_CFG

from . import mdp
from .infantry_2027_env_cfg import (
    EventCfg,
    Infantry2027FlatEnvCfg,
    Infantry2027SceneCfg,
)


# Fudan's final terrain run penalizes only ``base``, ``lf`` and ``rf``.
# Auxiliary four-bar links are deliberately excluded: touching one on a step
# is not equivalent to dragging the chassis and should not dominate learning.
TERRAIN_PENALIZED_CONTACT_BODIES = ["base_link", "lf_link", "rf_link"]


@configclass
class Infantry2027TerrainSceneCfg(Infantry2027SceneCfg):
    ground = None
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FUDAN_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.5,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.23, 0.26)),
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
class TerrainObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprioception = ObsTerm(func=mdp.proprioception, params={"noise_scale": 1.0}, history_length=5)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.scaled_base_lin_vel)
        proprioception = ObsTerm(func=mdp.proprioception, params={"noise_scale": 0.0})
        previous_action = ObsTerm(func=mdp.previous_action)
        previous_previous_action = ObsTerm(func=mdp.previous_previous_action)
        dof_acc = ObsTerm(
            func=mdp.controlled_joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(CONTROLLED_JOINTS))},
        )
        terrain_heights = ObsTerm(
            func=mdp.privileged_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "nominal_height": 0.233},
        )
        torque = ObsTerm(func=mdp.applied_effort)
        physics_randomization = ObsTerm(func=mdp.physics_randomization)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EstimatorTargetCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.scaled_base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    estimator_target: EstimatorTargetCfg = EstimatorTargetCfg()


@configclass
class TerrainRewardsCfg:
    """Final Fudan terrain reward set, mapped onto the verified VMC action."""

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
    tracking_ang_vel = RewTerm(func=mdp.tracking_ang_vel, weight=1.0, params={"command_name": "motion"})
    base_height = RewTerm(
        func=mdp.base_height,
        weight=1.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "multiplier": 1.5,
        },
    )
    base_height_enhance = RewTerm(
        func=mdp.base_height_enhance,
        weight=1.0,
        params={"command_name": "motion", "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    wheel_air_theta0 = RewTerm(
        func=mdp.wheel_air_leg_angle,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["lw_link", "rw_link"]),
            "threshold": 1.0,
        },
    )
    nominal_state = RewTerm(func=mdp.nominal_state, weight=-2.0, params={"maximum": 0.5})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z, weight=-0.1, params={"maximum": 10.0})
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy, weight=-0.05, params={"maximum": 20.0})
    orientation = RewTerm(func=mdp.orientation, weight=-15.0, params={"maximum": 1.0 / 15.0})
    dof_vel = RewTerm(
        func=mdp.dof_vel,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTIVE_LEG_JOINTS))},
    )
    dof_acc = RewTerm(
        func=mdp.dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(CONTROLLED_JOINTS))},
    )
    torques = RewTerm(func=mdp.torques, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate, weight=-0.01)
    action_smooth = RewTerm(func=mdp.action_smooth, weight=-0.01)
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
class TerrainEventCfg(EventCfg):
    material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 2.0),
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    base_inertia = EventTerm(
        func=mdp.randomize_inertia,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "scale_range": (0.8, 1.2)},
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    default_joint_pos = EventTerm(
        func=mdp.randomize_default_joint_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(CONTROLLED_JOINTS)),
            "offset_range": (-0.05, 0.05),
        },
    )
    disturbance_type = EventTerm(func=mdp.assign_random_disturbance, mode="reset")
    interval_disturbance = EventTerm(
        func=mdp.random_push_or_downward_impulse,
        mode="interval",
        interval_range_s=(7.0, 7.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "push_speed": 2.0,
            "downward_speed_range": (2.4, 2.8),
        },
    )


@configclass
class TerrainCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.fudan_terrain_levels,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "move_up_distance": 4.0,
            "tracking_threshold": 0.4,
        },
    )


@configclass
class Infantry2027TerrainEnvCfg(Infantry2027FlatEnvCfg):
    scene: Infantry2027TerrainSceneCfg = Infantry2027TerrainSceneCfg(num_envs=1024, env_spacing=3.0)
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    rewards: TerrainRewardsCfg = TerrainRewardsCfg()
    events: TerrainEventCfg = TerrainEventCfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.vmc.kp_scale_range = (0.9, 1.1)
        self.actions.vmc.kd_scale_range = (0.9, 1.1)
        self.actions.vmc.motor_scale_range = (0.9, 1.1)
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class Infantry2027TerrainPlayEnvCfg(Infantry2027TerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.curriculum_steps = 0
        self.observations.policy.proprioception.params["noise_scale"] = 0.0
        self.actions.vmc.action_delay_steps_range = (0, 0)
        self.actions.vmc.kp_scale_range = (1.0, 1.0)
        self.actions.vmc.kd_scale_range = (1.0, 1.0)
        self.actions.vmc.motor_scale_range = (1.0, 1.0)
        self.events.disturbance_type = None
        self.events.interval_disturbance = None
        self.events.material.params.update({
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
        })
        self.events.base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.events.base_inertia.params["scale_range"] = (1.0, 1.0)
        self.events.base_com.params["com_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
        }
        self.events.default_joint_pos.params["offset_range"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
