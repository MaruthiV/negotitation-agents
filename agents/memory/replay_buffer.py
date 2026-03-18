from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    budget_action: np.ndarray       # shape (n_budget_channels,)
    diplomatic_action: np.ndarray   # shape (n_targets,)
    log_prob: float
    reward: float
    value: float
    done: bool
    info: dict = field(default_factory=dict)


class RolloutBuffer:
    """On-policy rollout buffer for PPO. Stores one episode worth of data."""

    def __init__(self, obs_dim: int, n_budget_channels: int, n_targets: int):
        self.obs_dim = obs_dim
        self.n_budget_channels = n_budget_channels
        self.n_targets = n_targets
        self._transitions: list[Transition] = []

    def add(self, t: Transition) -> None:
        self._transitions.append(t)

    def clear(self) -> None:
        self._transitions.clear()

    def __len__(self) -> int:
        return len(self._transitions)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantage estimation."""
        n = len(self._transitions)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 0.0 if self._transitions[t].done else 1.0
            else:
                next_value = self._transitions[t + 1].value
                next_non_terminal = 0.0 if self._transitions[t].done else 1.0

            delta = (
                self._transitions[t].reward
                + gamma * next_value * next_non_terminal
                - self._transitions[t].value
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self._transitions[t].value

        return returns, advantages

    def to_tensors(
        self, device: torch.device, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = 0.0
    ) -> dict[str, torch.Tensor]:
        returns, advantages = self.compute_returns_and_advantages(last_value, gamma, gae_lambda)

        obs = np.stack([t.obs for t in self._transitions])
        budget_actions = np.stack([t.budget_action for t in self._transitions])
        diplomatic_actions = np.stack([t.diplomatic_action for t in self._transitions])
        log_probs = np.array([t.log_prob for t in self._transitions], dtype=np.float32)
        values = np.array([t.value for t in self._transitions], dtype=np.float32)

        return {
            "obs": torch.tensor(obs, dtype=torch.float32, device=device),
            "budget_actions": torch.tensor(budget_actions, dtype=torch.float32, device=device),
            "diplomatic_actions": torch.tensor(diplomatic_actions, dtype=torch.long, device=device),
            "log_probs": torch.tensor(log_probs, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
            "values": torch.tensor(values, dtype=torch.float32, device=device),
        }
