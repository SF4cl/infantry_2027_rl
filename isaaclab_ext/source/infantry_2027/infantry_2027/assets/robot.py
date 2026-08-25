"""Isaac Lab articulation configuration for the immutable infantry_2027_v0 snapshot."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[5]
ASSET_ROOT = PROJECT_ROOT / "assets" / "infantry_2027_v0"
USD_PATH = ASSET_ROOT / "isaac" / "infantry_2027_v0.usdc"
MJCF_PATH = ASSET_ROOT / "mujoco" / "infantry_2027_v0.xml"

# VMC order is front, back on each side.  This order is intentionally not the
# USD traversal order.
ACTIVE_LEG_JOINTS = ("lf_joint", "lb_joint", "rf_joint", "rb_joint")
WHEEL_JOINTS = ("lw_joint", "rw_joint")
CONTROLLED_JOINTS = (*ACTIVE_LEG_JOINTS, *WHEEL_JOINTS)
PASSIVE_JOINTS = (
    "lf1_joint", "lb1_joint", "lb2_joint", "lb3_joint",
    "rf1_joint", "rb1_joint", "rb2_joint", "rb3_joint",
)
# Positive normalized wheel target means +x motion.  The CAD wheel axes are
# opposite: left=-Y and right=+Y.
WHEEL_AXIS_SIGNS = (1.0, -1.0)

TOTAL_MASS_KG = 24.238245842
MIN_LEG_LENGTH = 0.16
MAX_LEG_LENGTH = 0.33
NOMINAL_LEG_LENGTH = 0.2276652114480664
WHEEL_RADIUS = 0.058
# At q=0 the wheel centre is 0.1574756 m below base_link.  Start with about
# 1.5 mm wheel clearance instead of dropping the vehicle from height.
INITIAL_ROOT_HEIGHT = 0.2170


INFANTRY_2027_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, INITIAL_ROOT_HEIGHT),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "active_legs": DCMotorCfg(
            joint_names_expr=list(ACTIVE_LEG_JOINTS),
            effort_limit=45.0,
            saturation_effort=45.0,
            velocity_limit=50.0,
            stiffness=0.0,
            damping=0.02,
            friction=0.0,
            armature=0.002,
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=list(WHEEL_JOINTS),
            effort_limit=5.0,
            saturation_effort=5.0,
            velocity_limit=60.0,
            stiffness=0.0,
            damping=0.001,
            friction=0.0,
            armature=0.001,
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=list(PASSIVE_JOINTS),
            stiffness=0.0,
            damping=0.02,
            effort_limit_sim=45.0,
            friction=0.0,
            armature=0.002,
        ),
    },
)
