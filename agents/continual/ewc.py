from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC).
    Computes diagonal Fisher information at task boundaries.
    Adds penalty to PPO loss: (importance/2) * Σ F_i * (θ_i - θ*_i)²

    Use as lightweight regularizer on top of CLEAR replay, not standalone.
    """

    def __init__(self, importance: float = 5000.0):
        self.importance = importance
        self._fisher: dict[str, torch.Tensor] = {}
        self._params_star: dict[str, torch.Tensor] = {}

    def consolidate(self, model: nn.Module, dataloader: DataLoader) -> None:
        """
        Compute diagonal Fisher at context boundary.
        Called once when transitioning to a new context.
        """
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

        n_batches = 0
        for batch in dataloader:
            obs = batch[0]
            budget_actions = batch[1]
            diplomatic_actions = batch[2]

            log_probs, _ = model.evaluate_actions(obs, budget_actions, diplomatic_actions)
            loss = -log_probs.mean()
            model.zero_grad()
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1

        if n_batches > 0:
            for n in fisher:
                fisher[n] /= n_batches

        # Accumulate Fisher (sum over tasks)
        for n in fisher:
            if n in self._fisher:
                self._fisher[n] += fisher[n]
            else:
                self._fisher[n] = fisher[n]

        # Store current params as anchor
        self._params_star = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        model.train()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        EWC penalty: (importance/2) * Σ F_i * (θ_i - θ*_i)²
        Returns scalar tensor; add to PPO total loss.
        """
        if not self._fisher:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, requires_grad=False)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._fisher:
                loss = loss + (self._fisher[n] * (p - self._params_star[n]) ** 2).sum()

        return (self.importance / 2) * loss

    def has_consolidated(self) -> bool:
        return len(self._fisher) > 0
