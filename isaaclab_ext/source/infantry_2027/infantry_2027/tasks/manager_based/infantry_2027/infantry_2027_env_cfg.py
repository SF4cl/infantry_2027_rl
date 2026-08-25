"""Flat VMC locomotion task for infantry_2027_v0."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from infantry_2027.assets import ACTIVE_LEG_JOINTS, CONTROLLED_JOINTS, INFANTRY_2027_CFG

from . import mdp


NON_WHEEL_BODIES = [
    "base_link", "lf_link", "lf1_link", "lb_link", "lb1_link", "lb2_link", "lb3_link",
    "rf_link", "rf1_link", "rb_link", "rb1_link", "rb2_link", "rb3_link",
]


@configclass
class Infantry2027SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        # Generate the plane locally.  GroundPlaneCfg references an online
        # NVIDIA USD and can make a long run depend on network/cache state.
        spawn=sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.14, 0.15)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    robot: ArticulationCfg = INFANTRY_2027_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=700.0),
    )


@configclass
class CommandsCfg:
    motion = mdp.DirectYawVelocityHeightCommandCfg(asset_name="robot", resampling_time_range=(5.0, 5.0))


@configclass
class ActionsCfg:
    vmc = mdp.VmcWheelActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
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
        torque = ObsTerm(func=mdp.applied_effort)
        physics_randomization = ObsTerm(func=mdp.physics_randomization)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EstimatorTargetCfg(ObsGroup):
        # Fudan supervises against the first three privileged values, i.e.
        # true body linear velocity after its observation scale of 2.0.
        base_lin_vel = ObsTerm(func=mdp.scaled_base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    estimator_target: EstimatorTargetCfg = EstimatorTargetCfg()


@configclass
class RewardsCfg:
    """Exact Fudan flat reward names, scales, formulas, and per-term clipping."""

    tracking_lin_vel = RewTerm(func=mdp.tracking_lin_vel, weight=1.0, params={"command_name": "motion"})
    tracking_lin_vel_enhance = RewTerm(
        func=mdp.tracking_lin_vel_enhance, weight=1.0, params={"command_name": "motion"}
    )
    tracking_ang_vel = RewTerm(func=mdp.tracking_ang_vel, weight=1.0, params={"command_name": "motion"})
    tracking_ang_vel_enhance = RewTerm(
        func=mdp.tracking_ang_vel_enhance, weight=1.0, params={"command_name": "motion"}
    )
    base_height = RewTerm(func=mdp.base_height, weight=1.0, params={"command_name": "motion"})
    nominal_state = RewTerm(func=mdp.nominal_state, weight=-1.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy, weight=-0.20)
    orientation = RewTerm(func=mdp.orientation, weight=-100.0)
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
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NON_WHEEL_BODIES),
            "threshold": 0.1,
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.dof_pos_limits,
        weight=-1.0,
        # The new active motors rotate continuously; the CAD-derived physical
        # stops live on the two front-knee relative joints.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lf1_joint", "rf1_joint"])},
    )


@configclass
class EventCfg:
    material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.6, 1.4),
            "restitution_range": (0.6, 1.0),
            "num_buckets": 64,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    base_inertia = EventTerm(
        func=mdp.randomize_inertia,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "scale_range": (0.9, 1.1)},
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    default_joint_pos = EventTerm(
        func=mdp.randomize_default_joint_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(CONTROLLED_JOINTS)),
            "offset_range": (-0.03, 0.03),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.sustained_bad_orientation,
        params={"minimum_projected_gravity_z": -0.1, "failure_time_s": 1.0},
    )
    non_finite = DoneTerm(func=mdp.non_finite_state)


@configclass
class Infantry2027FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: Infantry2027SceneCfg = Infantry2027SceneCfg(num_envs=2048, env_spacing=3.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 20.0
        self.viewer.origin_type = "world"
        self.viewer.eye = (1.4, 1.4, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.20)
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.scene.robot.soft_joint_pos_limit_factor = 0.97


@configclass
class Infantry2027FlatPlayEnvCfg(Infantry2027FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # A trained checkpoint should be evaluated over its final command
        # envelope immediately instead of replaying the 1500-iteration
        # training curriculum from progress zero.
        self.commands.motion.curriculum_steps = 0
        self.observations.policy.proprioception.params["noise_scale"] = 0.0
        self.actions.vmc.action_delay_steps_range = (0, 0)
        self.actions.vmc.kp_scale_range = (1.0, 1.0)
        self.actions.vmc.kd_scale_range = (1.0, 1.0)
        self.actions.vmc.motor_scale_range = (1.0, 1.0)
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
