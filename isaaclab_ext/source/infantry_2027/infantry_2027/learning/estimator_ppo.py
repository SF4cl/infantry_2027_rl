"""PPO plus the Fudan supervised encoder update."""

import torch
from rsl_rl.algorithms import PPO


class InfantryEstimatorPPO(PPO):
    def __init__(self, policy, extra_learning_rate: float = 1.0e-3, **kwargs):
        super().__init__(policy, **kwargs)
        self.extra_optimizer = torch.optim.Adam(policy.encoder.parameters(), lr=extra_learning_rate)

    def update(self) -> dict[str, float]:
        # Keep the rollout encoder unchanged until PPO finishes, otherwise old
        # log-probabilities and current actor inputs describe different models.
        history = self.storage.observations["policy"].flatten(0, 1).detach().clone()
        target = self.storage.observations["estimator_target"].flatten(0, 1).detach().clone()
        if not torch.isfinite(history).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Non-finite estimator rollout data.")
        losses = super().update()
        batch_size = history.shape[0]
        mini_size = batch_size // self.num_mini_batches
        loss_sum = 0.0
        rmse_sum = torch.zeros(3, device=self.device)
        updates = self.num_learning_epochs * self.num_mini_batches
        for _ in range(self.num_learning_epochs):
            order = torch.randperm(batch_size, device=self.device)
            for index in range(self.num_mini_batches):
                ids = order[index * mini_size:(index + 1) * mini_size]
                estimate = self.policy.encode(history[ids])
                mse = (estimate - target[ids]).square().mean(dim=0)
                loss = mse.mean()
                self.extra_optimizer.zero_grad()
                loss.backward()
                # Match the reference encoder update guard.
                torch.nn.utils.clip_grad_norm_(self.policy.encoder.parameters(), 0.1)
                self.extra_optimizer.step()
                loss_sum += loss.item()
                rmse_sum += mse.detach().sqrt()
        losses["estimator"] = loss_sum / updates
        rmse = rmse_sum / updates
        losses["estimator_vx_rmse"] = rmse[0].item()
        losses["estimator_vy_rmse"] = rmse[1].item()
        losses["estimator_vz_rmse"] = rmse[2].item()
        return losses
