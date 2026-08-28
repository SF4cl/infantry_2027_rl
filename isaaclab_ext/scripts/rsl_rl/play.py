# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

# GUI extensions bundle another HDF5 runtime.  On Windows, importing h5py
# after Kit starts can therefore bind h5py._errors to the incompatible Kit
# DLL and terminate Python with 0xc0000139.  Preload h5py's own runtime before
# AppLauncher changes the DLL search order.
import h5py  # noqa: F401, E402
import torch  # noqa: E402

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--free-camera",
    action="store_true",
    help="Disable the default robot-following viewport camera.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--keyboard", action="store_true", help="Control final-range commands from the keyboard.")
parser.add_argument("--forward-speed", type=float, default=2.5, help="Held-key forward/backward target in m/s.")
parser.add_argument("--yaw-acceleration", type=float, default=10.0, help="Held-key yaw command ramp in rad/s^2.")
parser.add_argument("--moving-yaw-limit", type=float, default=4.0, help="Yaw limit while translating in rad/s.")
parser.add_argument("--point-yaw-limit", type=float, default=10.0, help="Yaw limit while point-turning in rad/s.")
parser.add_argument("--base-height", type=float, default=0.233, help="Initial keyboard base-height target in m.")
parser.add_argument("--height-step", type=float, default=0.02, help="Instant base-height change per key event in m.")
parser.add_argument(
    "--terrain-type",
    type=str,
    default=None,
    choices=(
        "flat",
        "smooth_up",
        "smooth_down",
        "rough_up",
        "rough_down",
        "stairs_down",
        "stairs_up",
    ),
    help="Fix Terrain-Play to one reference terrain family.",
)
parser.add_argument(
    "--terrain-level",
    type=int,
    default=0,
    help="Fixed Terrain-Play difficulty row in [0, 9], where difficulty=row/10.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.keyboard and args_cli.headless:
    parser.error("--keyboard requires the graphical viewport; remove --headless.")
if args_cli.forward_speed <= 0.0 or args_cli.forward_speed > 2.5:
    parser.error("--forward-speed must be in (0, 2.5].")
if args_cli.yaw_acceleration <= 0.0:
    parser.error("--yaw-acceleration must be positive.")
if not 0.0 < args_cli.moving_yaw_limit <= 4.0:
    parser.error("--moving-yaw-limit must be in (0, 4].")
if not args_cli.moving_yaw_limit <= args_cli.point_yaw_limit <= 10.0:
    parser.error("--point-yaw-limit must be in [moving-yaw-limit, 10].")
if not 0.148 <= args_cli.base_height <= 0.318:
    parser.error("--base-height must be in [0.148, 0.318] m.")
if args_cli.height_step <= 0.0:
    parser.error("--height-step must be positive.")
if not 0 <= args_cli.terrain_level <= 9:
    parser.error("--terrain-level must be in [0, 9].")
if args_cli.terrain_type is not None and (args_cli.task is None or "Terrain" not in args_cli.task):
    parser.error("--terrain-type is only valid with an Infantry-2027-Terrain play task.")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import carb
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import infantry_2027.tasks  # noqa: F401
from infantry_2027.learning import InfantryOnPolicyRunner


TERRAIN_REPRESENTATIVE_COLUMNS = {
    "flat": 0,
    "smooth_up": 10,
    "smooth_down": 12,
    "rough_up": 14,
    "rough_down": 15,
    "stairs_down": 16,
    "stairs_up": 18,
}


class KeyboardYawHeightCommand:
    """Produce final-range ``vx, direct-yaw-rate, base-height`` commands.

    I/J/K/L/U/O are deliberately used instead of the viewport's standard
    WASD, arrow and Q/W/E/R/F bindings.  Held turn keys ramp direct yaw-rate;
    releasing both turn keys resets the target to zero immediately.  Height
    changes are discrete events, not a simulation-time slew.
    """

    _FORWARD_KEYS = {"I"}
    _BACKWARD_KEYS = {"K"}
    _LEFT_KEYS = {"J"}
    _RIGHT_KEYS = {"L"}
    _HEIGHT_UP_KEYS = {"U"}
    _HEIGHT_DOWN_KEYS = {"O"}
    _MOTION_KEYS = _FORWARD_KEYS | _BACKWARD_KEYS | _LEFT_KEYS | _RIGHT_KEYS
    _CONTROL_KEYS = _MOTION_KEYS | _HEIGHT_UP_KEYS | _HEIGHT_DOWN_KEYS

    def __init__(
        self,
        device: str,
        num_envs: int,
        forward_speed: float,
        yaw_acceleration: float,
        moving_yaw_limit: float,
        point_yaw_limit: float,
        base_height: float,
        height_step: float,
        minimum_base_height: float,
        maximum_base_height: float,
    ):
        # The headless Kit experience does not provide omni.appwindow.  Import
        # it only when keyboard control is actually requested.
        import omni.appwindow

        self.device = device
        self.num_envs = num_envs
        self.forward_speed = forward_speed
        self.yaw_acceleration = yaw_acceleration
        self.moving_yaw_limit = moving_yaw_limit
        self.point_yaw_limit = point_yaw_limit
        self.yaw_rate = 0.0
        self.base_height = base_height
        self.initial_base_height = base_height
        self.height_step = height_step
        self.minimum_base_height = minimum_base_height
        self.maximum_base_height = maximum_base_height
        self._pressed: set[str] = set()
        self._report_pending = True
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *_) -> bool:
        # Isaac Sim versions expose these fields either as enums or strings.
        raw_key = event.input
        key = raw_key if isinstance(raw_key, str) else getattr(raw_key, "name", str(raw_key))
        key = key.rsplit(".", 1)[-1].upper()
        raw_type = event.type
        event_type = raw_type if isinstance(raw_type, str) else getattr(raw_type, "name", str(raw_type))
        event_type = event_type.rsplit(".", 1)[-1].upper()
        is_press = raw_type == carb.input.KeyboardEventType.KEY_PRESS or event_type in {"KEY_PRESS", "KEY_REPEAT"}
        is_release = raw_type == carb.input.KeyboardEventType.KEY_RELEASE or event_type == "KEY_RELEASE"

        handled = False
        if is_press:
            if key == "M":
                self._pressed.difference_update(self._MOTION_KEYS)
                self.yaw_rate = 0.0
                handled = True
            elif key == "N":
                self.base_height = self.initial_base_height
                handled = True
            elif key in self._HEIGHT_UP_KEYS:
                self.base_height = min(
                    self.maximum_base_height, self.base_height + self.height_step
                )
                handled = True
            elif key in self._HEIGHT_DOWN_KEYS:
                self.base_height = max(
                    self.minimum_base_height, self.base_height - self.height_step
                )
                handled = True
            elif key in self._MOTION_KEYS:
                self._pressed.add(key)
                handled = True
        elif is_release and key in self._CONTROL_KEYS:
            self._pressed.discard(key)
            handled = True
        if handled:
            self._report_pending = True
            action = "PRESS" if is_press else "RELEASE"
            print(f"[KEYBOARD] {action:<7} {key}", flush=True)
        return True

    @staticmethod
    def _axis(positive_keys: set[str], negative_keys: set[str], pressed: set[str]) -> float:
        return float(bool(positive_keys & pressed)) - float(bool(negative_keys & pressed))

    def advance(self, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forward_axis = self._axis(self._FORWARD_KEYS, self._BACKWARD_KEYS, self._pressed)
        forward = torch.full(
            (self.num_envs,), self.forward_speed * forward_axis, dtype=torch.float32, device=self.device
        )
        turn_axis = self._axis(self._LEFT_KEYS, self._RIGHT_KEYS, self._pressed)
        if turn_axis == 0.0:
            self.yaw_rate = 0.0
        else:
            yaw_limit = self.point_yaw_limit if forward_axis == 0.0 else self.moving_yaw_limit
            self.yaw_rate = min(
                yaw_limit,
                max(-yaw_limit, self.yaw_rate + self.yaw_acceleration * turn_axis * dt),
            )
        yaw = torch.full((self.num_envs,), self.yaw_rate, dtype=torch.float32, device=self.device)
        height = torch.full((self.num_envs,), self.base_height, dtype=torch.float32, device=self.device)

        if self._report_pending:
            print(
                f"[COMMAND] vx={forward[0].item():+.2f} m/s  "
                f"yaw_rate={self.yaw_rate:+.2f} rad/s  "
                f"base_height={self.base_height:.3f} m (nominal leg={self.base_height + 0.012:.3f} m)",
                flush=True,
            )
            self._report_pending = False
        return forward, yaw, height

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if not args_cli.free_camera:
        # Keep environment zero's robot centered while retaining a stable
        # world-aligned view.  ``asset_root`` follows translation only, so
        # chassis pitch/roll does not shake or rotate the viewport camera.
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (1.4, 1.4, 0.60)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    if args_cli.keyboard:
        # Manual commands must not be replaced by the periodic sampler or a
        # time-limit reset while the user is testing a checkpoint.
        env_cfg.commands.motion.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.episode_length_s = 1.0e9
    if args_cli.terrain_type is not None:
        # A fixed inspection tile must not be replaced by the terrain
        # curriculum after an episode reset.
        env_cfg.curriculum.terrain_levels = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.terrain_type is not None:
        terrain = env.unwrapped.scene.terrain
        terrain_col = TERRAIN_REPRESENTATIVE_COLUMNS[args_cli.terrain_type]
        terrain.terrain_levels.fill_(args_cli.terrain_level)
        terrain.terrain_types.fill_(terrain_col)
        selected_origin = terrain.terrain_origins[args_cli.terrain_level, terrain_col]
        terrain.env_origins[:] = selected_origin
        # InteractiveScene keeps the same origins tensor in current Isaac Lab,
        # but copy explicitly so this remains correct if that implementation
        # detail changes.
        env.unwrapped.scene.env_origins[:] = selected_origin
        env.unwrapped.reset()
        print(
            f"[INFO] Fixed terrain: type={args_cli.terrain_type}, level={args_cli.terrain_level}, "
            f"difficulty={args_cli.terrain_level / 10.0:.1f}, column={terrain_col}",
            flush=True,
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "InfantryOnPolicyRunner":
        runner = InfantryOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    if agent_cfg.class_name == "InfantryOnPolicyRunner":
        # The estimator policy consumes a TensorDict (five-frame history plus
        # estimator target during training).  The stock RSL-RL exporter expects
        # a single tensor, so defer export to the dedicated sim2sim exporter.
        print("[INFO] Skipping stock export for the estimator policy; starting visualization.")
    else:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    keyboard = None
    command_term = None
    if args_cli.keyboard:
        command_term = env.unwrapped.command_manager.get_term("motion")
        if not hasattr(command_term, "set_manual_command"):
            raise TypeError("The motion command term does not support direct-yaw keyboard control.")
        task_forward_limit = command_term.cfg.ranges.forward_max
        keyboard_forward_speed = min(args_cli.forward_speed, task_forward_limit)
        if keyboard_forward_speed < args_cli.forward_speed:
            print(
                f"[INFO] Clamping keyboard speed from {args_cli.forward_speed:.2f} to this task's "
                f"{task_forward_limit:.2f} m/s limit.",
                flush=True,
            )
        keyboard_moving_yaw_limit = min(
            args_cli.moving_yaw_limit, command_term.cfg.moving_yaw_rate_limit
        )
        keyboard_point_yaw_limit = min(
            args_cli.point_yaw_limit, command_term.cfg.point_yaw_rate_limit
        )
        minimum_base_height, maximum_base_height = command_term.cfg.ranges.base_height
        keyboard_base_height = min(
            maximum_base_height, max(minimum_base_height, args_cli.base_height)
        )
        keyboard = KeyboardYawHeightCommand(
            device=env.unwrapped.device,
            num_envs=env.num_envs,
            forward_speed=keyboard_forward_speed,
            yaw_acceleration=args_cli.yaw_acceleration,
            moving_yaw_limit=keyboard_moving_yaw_limit,
            point_yaw_limit=keyboard_point_yaw_limit,
            base_height=keyboard_base_height,
            height_step=args_cli.height_step,
            minimum_base_height=minimum_base_height,
            maximum_base_height=maximum_base_height,
        )
        print("[INFO] Keyboard control enabled. Click the viewport once, then hold:", flush=True)
        print("       I / K : forward / backward", flush=True)
        print("       J / L : ramp direct yaw left / right; release both to zero", flush=True)
        print("       U / O : instant base-height step up / down", flush=True)
        print("       M     : stop forward and yaw commands", flush=True)
        print("       N     : reset base height", flush=True)
        print("[INFO] These keys avoid Isaac Sim's WASD, arrows and Q/W/E/R/F viewport bindings.", flush=True)

        forward, yaw, height = keyboard.advance(dt)
        command_term.set_manual_command(forward, yaw, height)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            if keyboard is not None:
                forward, yaw, height = keyboard.advance(dt)
                command_term.set_manual_command(forward, yaw, height)
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, _ = env.step(actions)
                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if keyboard is not None:
            keyboard.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
