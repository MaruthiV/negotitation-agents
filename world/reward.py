from __future__ import annotations

import math
from dataclasses import dataclass

from world.nation_state import NationState


@dataclass
class RewardWeights:
    survival: float = 1.0
    economic: float = 1.0
    military: float = 0.5
    territory: float = 0.5
    stability: float = 0.8


ARCHETYPE_WEIGHTS: dict[str, RewardWeights] = {
    "expansionist": RewardWeights(survival=1.0, economic=0.5, military=1.5, territory=2.0, stability=0.3),
    "mercantile":   RewardWeights(survival=1.0, economic=2.0, military=0.3, territory=0.3, stability=1.0),
    "isolationist": RewardWeights(survival=1.0, economic=1.0, military=0.5, territory=0.2, stability=2.0),
    "hegemon":      RewardWeights(survival=1.0, economic=1.2, military=1.5, territory=1.0, stability=0.8),
}


class RewardCalculator:
    def __init__(self, weights: RewardWeights):
        self.weights = weights

    def compute(
        self,
        prev: NationState,
        curr: NationState,
    ) -> float:
        if not curr.alive:
            return -10.0

        w = self.weights
        gdp_safe = max(prev.gdp, 1e-6)

        d_gdp = curr.gdp - prev.gdp
        d_mil = curr.military_strength - prev.military_strength
        d_territory = curr.territory - prev.territory

        reward = (
            w.survival * 0.1
            + w.economic * math.tanh(d_gdp / gdp_safe)
            + w.military * (math.tanh(d_mil) * 0.5 + curr.military_strength * 0.1)
            + w.territory * math.tanh(d_territory * 10.0)
            + w.stability * (curr.internal_stability - 0.5)
        )
        # Clamp to design envelope
        return max(-15.0, min(5.0, reward))
