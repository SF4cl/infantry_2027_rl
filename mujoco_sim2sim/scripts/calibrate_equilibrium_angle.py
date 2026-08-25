"""Calibrate the free-standing VMC swing-angle bias in MuJoCo.

The five-bar geometric angle is intentionally left unchanged.  This sweep
finds the task-space angle that best balances the verified full robot for a
given leg length while both wheel velocity targets are zero.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from vmc import PHI1_OFFSET, PHI4_OFFSET, state, wrap


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "infantry_2027_v0" / "mujoco" / "infantry_2027_v0.xml"
DEFAULT_REPORT = ROOT / "results" / "equilibrium_angle_mujoco.json"

ACTIVE = (("lf_joint", "lb_joint"), ("rf_joint", "rb_joint"))
WHEELS = ("lw_joint", "rw_joint")
LEG_ACTUATORS = (("lf_joint_motor", "lb_joint_motor"), ("rf_joint_motor", "rb_joint_motor"))
WHEEL_ACTUATORS = ("lw_joint_motor", "rw_joint_motor")
SIDE_SIGNS = np.asarray((1.0, -1.0))
WHEEL_SIGNS = np.asarray((1.0, -1.0))


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise KeyError(name)
    return int(value)


def load_ground_scene(model_path: Path) -> mujoco.MjModel:
    tree = ET.parse(model_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((model_path.parent / compiler.get("meshdir")).resolve()))
    # Visual meshes do not contribute mass and only slow down a headless sweep.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and (child.get("class") == "visual" or child.get("group") == "2"):
                parent.remove(child)
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    worldbody = root.find("worldbody")
    for geom in worldbody.iter("geom"):
        if geom.get("contype", "1") != "0":
            geom.set("contype", "1")
            geom.set("conaffinity", "2")
    ET.SubElement(
        worldbody,
        "geom",
        name="equilibrium_ground",
        type="plane",
        size="50 50 0.05",
        pos="0 0 0",
        friction="1.0 0.01 0.001",
        contype="2",
        conaffinity="1",
    )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def load_fixed_pose_model(model_path: Path) -> mujoco.MjModel:
    tree = ET.parse(model_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((model_path.parent / compiler.get("meshdir")).resolve()))
    root.find("option").set("gravity", "0 0 0")
    base = root.find("worldbody/body[@name='base_link']")
    free_joint = base.find("freejoint")
    base.remove(free_joint)
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class FreeStandingVmc:
    def __init__(self, model_path: Path):
        self.model = load_ground_scene(model_path)
        self.data = mujoco.MjData(self.model)
        self.pose_model = load_fixed_pose_model(model_path)
        self.base_id = object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.imu_id = object_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.joints = {
            name: (
                int(self.model.jnt_qposadr[object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
                int(self.model.jnt_dofadr[object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
            )
            for name in (*ACTIVE[0], *ACTIVE[1], *WHEELS)
        }
        self.leg_actuators = tuple(
            tuple(object_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in side)
            for side in LEG_ACTUATORS
        )
        self.wheel_actuators = tuple(
            object_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in WHEEL_ACTUATORS
        )
        self.free_qadr = int(
            self.model.jnt_qposadr[object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")]
        )

    def pose_for(self, target_length: float, target_angle: float) -> dict[str, float]:
        model = self.pose_model
        data = mujoco.MjData(model)
        joints = {
            name: (
                int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
                int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
            )
            for name in (*ACTIVE[0], *ACTIVE[1])
        }
        actuators = tuple(
            tuple(object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in side)
            for side in LEG_ACTUATORS
        )
        for _ in range(round(2.0 / model.opt.timestep)):
            data.ctrl[:] = 0.0
            for side in range(2):
                sign = SIDE_SIGNS[side]
                (q1, v1), (q4, v4) = (joints[name] for name in ACTIVE[side])
                length, angle, rate, jacobian, singular = state(
                    sign * data.qpos[q1] + PHI1_OFFSET,
                    sign * data.qpos[q4] + PHI4_OFFSET,
                    sign * data.qvel[v1],
                    sign * data.qvel[v4],
                )
                wrench = np.asarray(
                    (
                        900.0 * (target_length - length) - 20.0 * rate[0],
                        50.0 * wrap(target_angle - angle) - 3.0 * rate[1],
                    )
                )
                torque = np.clip(jacobian.T @ wrench, -45.0, 45.0) * sign
                if singular:
                    torque[:] = 0.0
                data.ctrl[list(actuators[side])] = torque
            mujoco.mj_step(model, data)
        pose = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            pose[name] = float(data.qpos[model.jnt_qposadr[joint_id]])
        return pose

    def reset(self, target_length: float, target_angle: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.free_qadr : self.free_qadr + 7] = (0.0, 0.0, 0.217, 1.0, 0.0, 0.0, 0.0)
        for name, value in self.pose_for(target_length, target_angle).items():
            joint_id = object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(self.model, self.data)
        wheel_bottoms = []
        for name in ("lw_link_collision", "rw_link_collision"):
            geom_id = object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            wheel_bottoms.append(
                float(self.data.geom_xpos[geom_id, 2] - self.model.geom_size[geom_id, 0])
            )
        self.data.qpos[self.free_qadr + 2] += 0.0015 - min(wheel_bottoms)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def leg_state(self, side: int):
        sign = SIDE_SIGNS[side]
        (q1, v1), (q4, v4) = (self.joints[name] for name in ACTIVE[side])
        return state(
            sign * self.data.qpos[q1] + PHI1_OFFSET,
            sign * self.data.qpos[q4] + PHI4_OFFSET,
            sign * self.data.qvel[v1],
            sign * self.data.qvel[v4],
        )

    def base_pitch_state(self) -> tuple[float, float]:
        angular, _ = self.body_velocity(local=False)
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
        return pitch, float(angular[1])

    def control(self, target_length: float, target_angle: float) -> tuple[np.ndarray, np.ndarray]:
        self.data.ctrl[:] = 0.0
        leg_efforts = []
        for side in range(2):
            length, angle, rate, jacobian, singular = self.leg_state(side)
            force = 900.0 * (target_length - length) - 20.0 * rate[0] + 118.88
            moment = 50.0 * wrap(target_angle - angle) - 3.0 * rate[1]
            torque = np.clip(jacobian.T @ np.asarray((force, moment)), -45.0, 45.0)
            torque *= SIDE_SIGNS[side]
            if singular:
                torque[:] = 0.0
            self.data.ctrl[list(self.leg_actuators[side])] = torque
            leg_efforts.extend(torque)
        wheel_velocity = np.asarray([self.data.qvel[self.joints[name][1]] for name in WHEELS])
        canonical_velocity = wheel_velocity * WHEEL_SIGNS
        canonical_effort = np.clip(-canonical_velocity, -5.0, 5.0)
        wheel_effort = canonical_effort * WHEEL_SIGNS
        self.data.ctrl[list(self.wheel_actuators)] = wheel_effort
        return np.asarray(leg_efforts), wheel_effort

    def body_velocity(self, local: bool = True) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.imu_id,
            velocity,
            int(local),
        )
        return velocity[:3].copy(), velocity[3:].copy()

    def apply_measurement_rig(self) -> tuple[float, float]:
        """Hold only the unstable base coordinates and return the measured wrench.

        The rig is deliberately external to the wheel/leg controller.  It is
        analogous to a low-friction laboratory gantry: vertical motion remains
        free, while horizontal translation and base attitude are softly held.
        The equilibrium angle is the sweep point requiring the least fore-aft
        holding force and pitch holding torque.
        """
        angular, linear = self.body_velocity(local=False)
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        position = self.data.xpos[self.base_id]

        force = np.asarray(
            (
                -220.0 * position[0] - 55.0 * linear[0],
                -220.0 * position[1] - 55.0 * linear[1],
                0.0,
            )
        )
        torque = np.asarray(
            (
                -100.0 * roll - 10.0 * angular[0],
                -100.0 * pitch - 10.0 * angular[1],
                -60.0 * yaw - 6.0 * angular[2],
            )
        )
        self.data.xfrc_applied[self.base_id] = np.concatenate((force, torque))
        return float(force[0]), float(torque[1])

    def evaluate(
        self,
        target_length: float,
        target_angle: float,
        duration_s: float,
        evaluation_s: float,
    ) -> dict[str, float | bool]:
        self.reset(target_length, target_angle)
        steps = round(duration_s / self.model.opt.timestep)
        evaluation_steps = round(evaluation_s / self.model.opt.timestep)
        rows = []
        failed = False
        for step_index in range(steps):
            self.data.xfrc_applied[:] = 0.0
            leg_effort, wheel_effort = self.control(target_length, target_angle)
            holding_force, holding_torque = self.apply_measurement_rig()
            mujoco.mj_step(self.model, self.data)
            angular, linear = self.body_velocity()
            rotation = self.data.xmat[self.base_id].reshape(3, 3)
            pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
            position = self.data.xpos[self.base_id]
            finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
            failed = failed or not finite or abs(pitch) > math.radians(45.0) or position[2] < 0.08
            if step_index >= steps - evaluation_steps:
                rows.append(
                    (
                        linear[0],
                        pitch,
                        wheel_effort.copy(),
                        leg_effort.copy(),
                        position[0],
                        holding_force,
                        holding_torque,
                    )
                )
            if failed:
                break
        if not rows:
            return {
                "target_length_m": target_length,
                "target_angle_rad": target_angle,
                "survived": False,
                "failure_time_s": float(self.data.time),
                "vx_mean_mps": math.nan,
                "vx_abs_mean_mps": math.inf,
                "displacement_rate_mps": math.inf,
                "pitch_mean_rad": math.nan,
                "pitch_abs_mean_rad": math.inf,
                "wheel_effort_abs_mean_nm": math.inf,
                "holding_force_abs_mean_n": math.inf,
                "holding_torque_abs_mean_nm": math.inf,
                "leg_effort_abs_mean_nm": math.inf,
                "score": 1.0e6 - float(self.data.time),
            }
        values = np.asarray([row[:2] for row in rows])
        wheel = np.asarray([row[2] for row in rows])
        leg = np.asarray([row[3] for row in rows])
        x = np.asarray([row[4] for row in rows])
        holding_force = np.asarray([row[5] for row in rows])
        holding_torque = np.asarray([row[6] for row in rows])
        vx_abs_mean = float(np.nanmean(np.abs(values[:, 0])))
        pitch_abs_mean = float(np.nanmean(np.abs(values[:, 1])))
        wheel_effort_abs_mean = float(np.nanmean(np.abs(wheel)))
        holding_force_abs_mean = float(np.mean(np.abs(holding_force)))
        holding_torque_abs_mean = float(np.mean(np.abs(holding_torque)))
        displacement_rate = float(abs(x[-1] - x[0]) / max(evaluation_s, 1.0e-6))
        score = (
            vx_abs_mean / 0.05
            + displacement_rate / 0.05
            + pitch_abs_mean / math.radians(2.0)
            + holding_force_abs_mean / 5.0
            + holding_torque_abs_mean / 1.0
            + (100.0 if failed else 0.0)
        )
        return {
            "target_length_m": target_length,
            "target_angle_rad": target_angle,
            "survived": not failed,
            "failure_time_s": float(self.data.time),
            "vx_mean_mps": float(np.nanmean(values[:, 0])),
            "vx_abs_mean_mps": vx_abs_mean,
            "displacement_rate_mps": displacement_rate,
            "pitch_mean_rad": float(np.nanmean(values[:, 1])),
            "pitch_abs_mean_rad": pitch_abs_mean,
            "wheel_effort_abs_mean_nm": wheel_effort_abs_mean,
            "holding_force_mean_n": float(np.mean(holding_force)),
            "holding_force_abs_mean_n": holding_force_abs_mean,
            "holding_torque_mean_nm": float(np.mean(holding_torque)),
            "holding_torque_abs_mean_nm": holding_torque_abs_mean,
            "leg_effort_abs_mean_nm": float(np.nanmean(np.abs(leg))),
            "score": score,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--evaluation", type=float, default=1.5)
    parser.add_argument("--angle-min", type=float, default=-0.08)
    parser.add_argument("--angle-max", type=float, default=0.02)
    parser.add_argument("--angle-step", type=float, default=0.005)
    args = parser.parse_args()
    if not 0.0 < args.evaluation <= args.duration:
        parser.error("evaluation must be in (0, duration]")
    lengths = (0.16, 0.22, 0.28, 0.33)
    angles = np.arange(args.angle_min, args.angle_max + 0.5 * args.angle_step, args.angle_step)
    controller = FreeStandingVmc(args.model.resolve())
    rows = []
    selected = []
    for length in lengths:
        group = []
        for angle in angles:
            result = controller.evaluate(length, float(angle), args.duration, args.evaluation)
            group.append(result)
            rows.append(result)
        best = min(group, key=lambda row: float(row["score"]))
        selected.append({
            "leg_length_m": length,
            "equilibrium_angle_rad": best["target_angle_rad"],
            "score": best["score"],
            "vx_abs_mean_mps": best["vx_abs_mean_mps"],
            "pitch_abs_mean_rad": best["pitch_abs_mean_rad"],
            "wheel_effort_abs_mean_nm": best["wheel_effort_abs_mean_nm"],
            "holding_force_abs_mean_n": best["holding_force_abs_mean_n"],
            "holding_torque_abs_mean_nm": best["holding_torque_abs_mean_nm"],
        })
        print(
            f"[equilibrium] L={length:.3f} m theta={best['target_angle_rad']:+.4f} rad "
            f"vx_abs={best['vx_abs_mean_mps']:.4f} m/s pitch_abs={best['pitch_abs_mean_rad']:.4f} rad",
            flush=True,
        )
    coefficients = np.polyfit(
        np.asarray([row["leg_length_m"] for row in selected]),
        np.asarray([row["equilibrium_angle_rad"] for row in selected]),
        1,
    )
    report = {
        "model": str(args.model.resolve()),
        "controller": {
            "length_kp": 900.0,
            "length_kd": 20.0,
            "angle_kp": 50.0,
            "angle_kd": 3.0,
            "support_force_per_leg_n": 118.88,
            "leg_effort_limit_nm": 45.0,
            "wheel_effort_limit_nm": 5.0,
        },
        "measurement_rig": {
            "translation_xy_kp_n_per_m": 220.0,
            "translation_xy_kd_ns_per_m": 55.0,
            "attitude_roll_pitch_kp_nm_per_rad": 100.0,
            "attitude_roll_pitch_kd_nms_per_rad": 10.0,
            "attitude_yaw_kp_nm_per_rad": 60.0,
            "attitude_yaw_kd_nms_per_rad": 6.0,
            "vertical_axis_free": True,
        },
        "sweep": {
            "lengths_m": list(lengths),
            "angles_rad": angles.tolist(),
            "duration_s": args.duration,
            "evaluation_window_s": args.evaluation,
        },
        "selected": selected,
        "linear_fit": {
            "slope_rad_per_m": float(coefficients[0]),
            "intercept_rad": float(coefficients[1]),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[equilibrium] report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
