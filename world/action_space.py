from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium


# Budget channels: military, trade_investment, tech_rd, internal_dev, reserves
BUDGET_CHANNELS = ["military", "trade_investment", "tech_rd", "internal_dev", "reserves"]
N_BUDGET_CHANNELS = 5

# Diplomatic action codes (per target nation)
DIPLOMATIC_ACTIONS = [
    "do_nothing",        # 0
    "propose_trade",     # 1
    "propose_alliance",  # 2
    "impose_sanctions",  # 3
    "threaten",          # 4
    "declare_war",       # 5
    "negotiate_peace",   # 6
]
N_DIPLOMATIC_OPTIONS = 7


def make_action_space(n_nations: int) -> gymnasium.spaces.Dict:
    """Build the per-agent action space for a world with n_nations total."""
    n_targets = n_nations - 1
    return gymnasium.spaces.Dict({
        "budget_allocation": gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(N_BUDGET_CHANNELS,), dtype=np.float32
        ),
        "diplomatic_actions": gymnasium.spaces.MultiDiscrete(
            [N_DIPLOMATIC_OPTIONS] * n_targets
        ),
    })


@dataclass
class DecodedAction:
    budget: np.ndarray          # shape (5,), sums to 1 after normalization
    diplomatic: list[int]       # one int per other nation


class ActionEncoder:
    def __init__(self, nation_ids: list[str], self_id: str):
        self.nation_ids = nation_ids
        self.self_id = self_id
        self.targets = [n for n in nation_ids if n != self_id]

    def decode(self, raw_action: dict) -> DecodedAction:
        budget = np.array(raw_action["budget_allocation"], dtype=np.float32)
        # Normalize budget to simplex
        budget = np.abs(budget)
        total = budget.sum()
        if total < 1e-8:
            budget = np.ones(N_BUDGET_CHANNELS, dtype=np.float32) / N_BUDGET_CHANNELS
        else:
            budget = budget / total

        diplomatic = list(raw_action["diplomatic_actions"])
        return DecodedAction(budget=budget, diplomatic=diplomatic)

    def get_target(self, idx: int) -> str:
        return self.targets[idx]

    def diplomatic_action_name(self, code: int) -> str:
        return DIPLOMATIC_ACTIONS[code]
