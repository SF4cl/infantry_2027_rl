"""Five-frame actor with a supervised three-dimensional velocity encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization
from tensordict import TensorDict
from torch.distributions import Normal


class InfantryEstimatorActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=(128, 64, 32),
        critic_hidden_dims=(256, 128, 64),
        encoder_hidden_dims=(128, 64),
        history_length: int = 5,
        single_frame_dim: int = 25,
        latent_dim: int = 3,
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"InfantryEstimatorActorCritic ignored arguments: {sorted(kwargs)}")
        history_dim = history_length * single_frame_dim
        if obs["policy"].shape[-1] != history_dim:
            raise ValueError(f"Expected {history_dim} policy values, got {obs['policy'].shape[-1]}.")
        if obs["estimator_target"].shape[-1] != 3:
            raise ValueError("Estimator target must be true 3-D base linear velocity.")
        critic_dim = sum(obs[name].shape[-1] for name in obs_groups["critic"])
        self.obs_groups = obs_groups
        self.single_frame_dim = single_frame_dim
        self.encoder = MLP(history_dim, latent_dim, encoder_hidden_dims, activation)
        self.actor = MLP(single_frame_dim + latent_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_dim, 1, critic_hidden_dims, activation)
        self.actor_obs_normalizer = EmpiricalNormalization(single_frame_dim + latent_dim) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(critic_dim) if critic_obs_normalization else nn.Identity()
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}")
        self.noise_std_type = noise_std_type
        self.distribution = None
        Normal.set_default_validate_args(False)
        print(f"Velocity encoder: {self.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def reset(self, dones=None):
        pass

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history)

    def _actor_obs(self, obs: TensorDict) -> torch.Tensor:
        history = obs["policy"]
        current = history[..., -self.single_frame_dim:]
        # The actor cannot alter the velocity estimate through PPO.  The
        # separate supervised optimizer owns the encoder, as in Fudan.
        return torch.cat((current, self.encode(history).detach()), dim=-1)

    def _critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["critic"]], dim=-1)

    def _update_distribution(self, actor_obs: torch.Tensor) -> None:
        mean = self.actor(actor_obs)
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" else torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        self._update_distribution(self.actor_obs_normalizer(self._actor_obs(obs)))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(self.actor_obs_normalizer(self._actor_obs(obs)))

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        return self.critic(self.critic_obs_normalizer(self._critic_obs(obs)))

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if isinstance(self.actor_obs_normalizer, EmpiricalNormalization):
            self.actor_obs_normalizer.update(self._actor_obs(obs))
        if isinstance(self.critic_obs_normalizer, EmpiricalNormalization):
            self.critic_obs_normalizer.update(self._critic_obs(obs))
