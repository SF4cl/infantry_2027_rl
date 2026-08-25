"""Closed-loop MuJoCo runtime for infantry_2027_v0 sim2sim."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from policy import FRAME_DIM, HISTORY_LENGTH, NumpyPolicy, decode_action
from vmc import PHI1_OFFSET, PHI4_OFFSET, state, wrap


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "infantry_2027_v0" / "mujoco" / "infantry_2027_v0.xml"
DEFAULT_POLICY = ROOT / "exported" / "model_1600.npz"


@dataclass(frozen=True)
class VmcGains:
    """Explicit task-space gains used after the policy action is decoded."""

    kp_length: float = 900.0
    kd_length: float = 20.0
    kp_angle: float = 50.0
    kd_angle: float = 3.0

    def __post_init__(self) -> None:
        values = (self.kp_length, self.kd_length, self.kp_angle, self.kd_angle)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"VMC gains must be finite and non-negative, got {values}")


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise KeyError(name)
    return value


class Runtime:
    ACTIVE = (("lf_joint", "lb_joint"), ("rf_joint", "rb_joint"))
    WHEELS = ("lw_joint", "rw_joint")
    LEG_ACTUATORS = (("lf_joint_motor", "lb_joint_motor"), ("rf_joint_motor", "rb_joint_motor"))
    WHEEL_ACTUATORS = ("lw_joint_motor", "rw_joint_motor")
    SIDE_SIGNS = np.array((1.0, -1.0))
    WHEEL_SIGNS = np.array((1.0, -1.0))

    def __init__(
        self,
        policy_path: Path = DEFAULT_POLICY,
        model_path: Path = DEFAULT_MODEL,
        initial_height: float = 0.215,
        gains: VmcGains | None = None,
        load_visuals: bool = True,
    ):
        self.policy = NumpyPolicy(policy_path)
        self.gains = gains if gains is not None else VmcGains()
        self.model_path = Path(model_path).resolve()
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest().upper()
        self.load_visuals = load_visuals
        self.model = self._load_ground_scene()
        self.data = mujoco.MjData(self.model)
        if not math.isclose(float(self.model.opt.timestep), 0.002, abs_tol=1.0e-12):
            raise ValueError("The policy requires a 500 Hz physics timestep")
        self.base_id = object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.imu_id = object_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.joints = {
            name: (
                int(self.model.jnt_qposadr[object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
                int(self.model.jnt_dofadr[object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
            )
            for name in (*self.ACTIVE[0], *self.ACTIVE[1], *self.WHEELS)
        }
        self.leg_actuators = tuple(tuple(object_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in side) for side in self.LEG_ACTUATORS)
        self.wheel_actuators = tuple(object_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.WHEEL_ACTUATORS)
        self.command = np.array((0.0, 0.0, initial_height), dtype=np.float64)
        self.raw_action = np.zeros(6)
        self.latent = np.zeros(3)
        self.targets = decode_action(self.raw_action, initial_height)
        self.leg_length = np.zeros(2)
        self.leg_angle = np.zeros(2)
        self.leg_rate = np.zeros((2, 2))
        self.jacobian = np.zeros((2, 2, 2))
        self.singular = np.zeros(2, dtype=bool)
        self.last_leg_effort = np.zeros(4)
        self.last_wheel_effort = np.zeros(2)
        self.inner_leg_effort_peak = 0.0
        self.inner_leg_saturation_fraction = 0.0
        self.history = np.zeros((HISTORY_LENGTH, FRAME_DIM), dtype=np.float32)
        self.initialize(initial_height)

    def _load_ground_scene(self) -> mujoco.MjModel:
        """Compile a derived scene without changing the immutable asset XML."""
        tree = ET.parse(self.model_path)
        root = tree.getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str((self.model_path.parent / compiler.get("meshdir")).resolve()))
        if not self.load_visuals:
            # The CAD meshes have density=0 and collision disabled.  Removing
            # them from headless sweeps preserves dynamics while avoiding one
            # multi-megabyte mesh copy per worker process.
            for parent in root.iter():
                for child in list(parent):
                    if child.tag == "geom" and (
                        child.get("class") == "visual" or child.get("group") == "2"
                    ):
                        parent.remove(child)
            asset = root.find("asset")
            if asset is not None:
                for mesh in list(asset.findall("mesh")):
                    asset.remove(mesh)
        worldbody = root.find("worldbody")
        for geom in worldbody.iter("geom"):
            if geom.get("contype", "1") != "0":
                # Isaac training disables articulation self-collision.
                geom.set("contype", "1")
                geom.set("conaffinity", "2")
        ET.SubElement(
            worldbody, "geom", name="sim2sim_ground", type="plane", size="50 50 0.05",
            pos="0 0 0", friction="1.0 0.01 0.001", contype="2", conaffinity="1",
            rgba="0.15 0.17 0.19 1",
        )
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def initialize(self, height: float = 0.215) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # The immutable MJCF starts at 0.45 m. Match Isaac's q=0 near-ground reset.
        free_id = object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        qadr = int(self.model.jnt_qposadr[free_id])
        self.data.qpos[qadr:qadr + 7] = (0.0, 0.0, 0.217, 1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)
        # Preserve q=0 and translate only the free base so the wheel is 1.5 mm above the floor.
        wheel_bottoms = []
        for name in ("lw_link_collision", "rw_link_collision"):
            gid = object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            wheel_bottoms.append(float(self.data.geom_xpos[gid, 2] - self.model.geom_size[gid, 0]))
        self.data.qpos[qadr + 2] += 0.0015 - min(wheel_bottoms)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.command[:] = (0.0, 0.0, height)
        self.raw_action[:] = 0.0
        self.latent[:] = 0.0
        self.targets = decode_action(self.raw_action, height)
        self._update_leg_state()
        self.history[:] = self.observation_frame()

    def set_command(self, command) -> None:
        value = np.asarray(command, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Command must be finite [vx, yaw_rate, base_height]")
        if not (-2.3 <= value[0] <= 2.3 and -3.0 <= value[1] <= 3.0 and 0.148 <= value[2] <= 0.318):
            raise ValueError(f"Command outside the exported policy command envelope: {value}")
        self.command[:] = value

    def _update_leg_state(self) -> None:
        for side, names in enumerate(self.ACTIVE):
            sign = self.SIDE_SIGNS[side]
            (q1, v1), (q4, v4) = (self.joints[name] for name in names)
            result = state(
                sign * self.data.qpos[q1] + PHI1_OFFSET,
                sign * self.data.qpos[q4] + PHI4_OFFSET,
                sign * self.data.qvel[v1], sign * self.data.qvel[v4],
            )
            self.leg_length[side], self.leg_angle[side], self.leg_rate[side], self.jacobian[side], self.singular[side] = result

    def body_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.imu_id, velocity, 1)
        return velocity[:3].copy(), velocity[3:].copy()

    def projected_gravity(self) -> np.ndarray:
        return self.data.xmat[self.base_id].reshape(3, 3).T @ np.array((0.0, 0.0, -1.0))

    def observation_frame(self) -> np.ndarray:
        angular, _ = self.body_velocity()
        leg_q = np.array([self.data.qpos[self.joints[n][0]] for n in (*self.ACTIVE[0], *self.ACTIVE[1])])
        vel_names = (*self.ACTIVE[0], self.WHEELS[0], *self.ACTIVE[1], self.WHEELS[1])
        joint_vel = np.array([self.data.qvel[self.joints[n][1]] for n in vel_names])
        frame = np.concatenate((
            angular * 0.25, self.projected_gravity(), self.command * (2.0, 0.25, 5.0),
            leg_q, joint_vel * 0.05, self.raw_action,
        ))
        if frame.shape != (25,):
            raise AssertionError(frame.shape)
        return np.nan_to_num(frame, nan=0.0, posinf=100.0, neginf=-100.0).clip(-100.0, 100.0).astype(np.float32)

    def _control(self) -> None:
        self.data.ctrl[:] = 0.0
        efforts = []
        for side in range(2):
            force = (
                self.gains.kp_length * (self.targets["length"][side] - self.leg_length[side])
                - self.gains.kd_length * self.leg_rate[side, 0]
                + 118.88
            )
            moment = (
                self.gains.kp_angle * wrap(self.targets["angle"][side] - self.leg_angle[side])
                - self.gains.kd_angle * self.leg_rate[side, 1]
            )
            canonical = self.jacobian[side].T @ np.array((force, moment))
            physical = np.clip(canonical * self.SIDE_SIGNS[side], -45.0, 45.0)
            if self.singular[side]:
                physical[:] = 0.0
            self.data.ctrl[list(self.leg_actuators[side])] = physical
            efforts.extend(physical)
        wheel_velocity = np.array([self.data.qvel[self.joints[n][1]] for n in self.WHEELS])
        wheel_target = self.targets["wheel"] * self.WHEEL_SIGNS
        wheel_effort = np.clip(wheel_target - wheel_velocity, -5.0, 5.0)
        self.data.ctrl[list(self.wheel_actuators)] = wheel_effort
        self.last_leg_effort[:] = efforts
        self.last_wheel_effort[:] = wheel_effort

    def step(self) -> dict:
        self.latent[:], action = self.policy.infer(self.history.reshape(-1))
        self.raw_action[:] = np.clip(action, -100.0, 100.0)
        self.targets = decode_action(self.raw_action, float(self.command[2]))
        inner_efforts = []
        for _ in range(5):
            self._control()
            inner_efforts.append(self.last_leg_effort.copy())
            mujoco.mj_step(self.model, self.data)
            self._update_leg_state()
        inner_efforts = np.asarray(inner_efforts)
        self.inner_leg_effort_peak = float(np.max(np.abs(inner_efforts)))
        self.inner_leg_saturation_fraction = float(np.mean(np.abs(inner_efforts) >= 44.999))
        self.history[:-1] = self.history[1:]
        self.history[-1] = self.observation_frame()
        return self.metrics()

    def metrics(self) -> dict:
        angular, linear = self.body_velocity()
        gravity = self.projected_gravity()
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        closure = []
        for a, b in (("lf_link_closure", "lb2_link_closure"), ("lf1_link_closure", "lb3_link_closure"),
                     ("rf_link_closure", "rb2_link_closure"), ("rf1_link_closure", "rb3_link_closure")):
            aid, bid = object_id(self.model, mujoco.mjtObj.mjOBJ_SITE, a), object_id(self.model, mujoco.mjtObj.mjOBJ_SITE, b)
            closure.append(float(np.linalg.norm(self.data.site_xpos[aid] - self.data.site_xpos[bid])))
        position = self.data.xpos[self.base_id].copy()
        finite = all(np.isfinite(x).all() for x in (self.data.qpos, self.data.qvel, self.raw_action, self.leg_length))
        return {
            "time": float(self.data.time), "position": position, "linear": linear, "angular": angular,
            "height": float(position[2]), "tilt": tilt, "length": self.leg_length.copy(),
            "leg_angle": self.leg_angle.copy(), "target_length": self.targets["length"].copy(),
            "target_angle": self.targets["angle"].copy(), "target_wheel": self.targets["wheel"].copy(),
            "command": self.command.copy(),
            "action": self.raw_action.copy(), "latent": self.latent.copy(),
            "leg_effort": self.last_leg_effort.copy(), "wheel_effort": self.last_wheel_effort.copy(),
            "inner_leg_effort_peak": self.inner_leg_effort_peak,
            "inner_leg_saturation_fraction": self.inner_leg_saturation_fraction,
            "closure": max(closure), "finite": finite,
            "failed": (not finite) or tilt > math.radians(60.0) or position[2] < 0.08,
        }
