from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Categorical


class ActorNetwork(nn.Module):
    """
    Actor for IPPO agent.
    Outputs:
      - Dirichlet concentration params for budget allocation (simplex)
      - Categorical logits for each diplomatic target
    """

    def __init__(
        self,
        obs_dim: int,
        n_budget_channels: int,
        n_targets: int,
        n_diplomatic_options: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_budget_channels = n_budget_channels
        self.n_targets = n_targets
        self.n_diplomatic_options = n_diplomatic_options

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Dirichlet head: outputs concentration params > 0
        self.budget_head = nn.Linear(hidden_dim, n_budget_channels)

        # Categorical heads: one per diplomatic target
        self.diplomatic_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_diplomatic_options)
            for _ in range(n_targets)
        ])

    def forward(self, obs: torch.Tensor):
        """Returns (budget_dist, diplomatic_dists)."""
        h = self.shared(obs)

        # Dirichlet: concentration params must be > 0
        budget_conc = F.softplus(self.budget_head(h)) + 1e-3
        budget_dist = Dirichlet(budget_conc)

        diplomatic_dists = [
            Categorical(logits=head(h))
            for head in self.diplomatic_heads
        ]
        return budget_dist, diplomatic_dists

    def get_action_and_logprob(self, obs: torch.Tensor):
        budget_dist, diplomatic_dists = self.forward(obs)

        budget_sample = budget_dist.rsample()
        budget_logprob = budget_dist.log_prob(budget_sample)

        diplomatic_samples = []
        diplomatic_logprobs = []
        for dist in diplomatic_dists:
            a = dist.sample()
            diplomatic_samples.append(a)
            diplomatic_logprobs.append(dist.log_prob(a))

        total_logprob = budget_logprob + torch.stack(diplomatic_logprobs).sum()
        entropy = budget_dist.entropy() + sum(d.entropy() for d in diplomatic_dists)

        return budget_sample, diplomatic_samples, total_logprob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        budget_actions: torch.Tensor,
        diplomatic_actions: torch.Tensor,
    ):
        """Recompute log_prob and entropy for stored actions (for PPO update)."""
        budget_dist, diplomatic_dists = self.forward(obs)

        budget_logprob = budget_dist.log_prob(budget_actions)
        dipl_logprobs = []
        for i, dist in enumerate(diplomatic_dists):
            dipl_logprobs.append(dist.log_prob(diplomatic_actions[:, i]))

        log_prob = budget_logprob + torch.stack(dipl_logprobs, dim=1).sum(dim=1)
        entropy = budget_dist.entropy() + sum(d.entropy() for d in diplomatic_dists)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Value function estimator for IPPO (uses only own observation)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
