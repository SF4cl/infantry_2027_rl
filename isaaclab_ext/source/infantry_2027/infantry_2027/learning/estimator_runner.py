"""RSL-RL runner factory/checkpoint glue for the estimator policy."""

import torch
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from .estimator_policy import InfantryEstimatorActorCritic
from .estimator_ppo import InfantryEstimatorPPO


class InfantryOnPolicyRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs: TensorDict) -> InfantryEstimatorPPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        policy_name = self.policy_cfg.pop("class_name")
        algorithm_name = self.alg_cfg.pop("class_name")
        if policy_name != "InfantryEstimatorActorCritic" or algorithm_name != "InfantryEstimatorPPO":
            raise ValueError(f"Unsupported estimator classes: {policy_name}, {algorithm_name}")
        policy = InfantryEstimatorActorCritic(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        algorithm = InfantryEstimatorPPO(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        checkpoint["estimator_optimizer_state_dict"] = self.alg.extra_optimizer.state_dict()
        checkpoint["checkpoint_schema"] = "infantry-2027-v0-fudan-estimator"
        # Upstream stores the zero-based index of the update that just
        # finished.  Keep an explicit count so resume targets are unambiguous.
        terminal_save = isinstance(infos, dict) and infos.get("reason") in ("completed", "KeyboardInterrupt")
        checkpoint["completed_iterations"] = (
            self.current_learning_iteration if terminal_save else self.current_learning_iteration + 1
        )
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = True, map_location=None):
        checkpoint = torch.load(path, weights_only=False, map_location=map_location)
        schema = checkpoint.get("checkpoint_schema")
        if schema not in (None, "infantry-2027-v0-fudan-estimator"):
            raise ValueError(f"Incompatible checkpoint schema: {schema}")
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        self.current_learning_iteration = checkpoint.get(
            "completed_iterations", self.current_learning_iteration
        )
        if load_optimizer and "estimator_optimizer_state_dict" in checkpoint:
            self.alg.extra_optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"])
        return infos
