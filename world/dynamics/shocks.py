from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from world.nation_state import NationState


class ShockType(Enum):
    RESOURCE_DISCOVERY = "resource_discovery"
    PANDEMIC = "pandemic"
    TECH_BREAKTHROUGH = "tech_breakthrough"
    NATURAL_DISASTER = "natural_disaster"
    FINANCIAL_CRISIS = "financial_crisis"


@dataclass
class ActiveShock:
    shock_type: ShockType
    affected_nation: str
    duration_steps: int
    steps_remaining: int
    magnitude: float
    metadata: dict = field(default_factory=dict)


class ExogenousShockGenerator:
    """Fires exogenous shocks with configured per-step probabilities."""

    DEFAULT_PROBS: dict[ShockType, float] = {
        ShockType.RESOURCE_DISCOVERY: 0.002,
        ShockType.PANDEMIC: 0.003,
        ShockType.TECH_BREAKTHROUGH: 0.002,
        ShockType.NATURAL_DISASTER: 0.004,
        ShockType.FINANCIAL_CRISIS: 0.003,
    }

    def __init__(
        self,
        shock_probs: Optional[dict[ShockType, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.shock_probs = shock_probs or self.DEFAULT_PROBS
        self.rng = rng or np.random.default_rng()
        self.active_shocks: list[ActiveShock] = []

    def step(
        self,
        nations: dict[str, NationState],
        timestep: int,
        events: list[dict],
    ) -> None:
        """Fire new shocks and apply ongoing effects."""
        alive_ids = [nid for nid, n in nations.items() if n.alive]
        if not alive_ids:
            return

        # Potentially fire new shocks
        for shock_type, prob in self.shock_probs.items():
            if self.rng.random() < prob:
                target_id = self.rng.choice(alive_ids)
                shock = self._create_shock(shock_type, target_id)
                self.active_shocks.append(shock)
                events.append({
                    "type": "SHOCK_STARTED",
                    "shock_type": shock_type.value,
                    "nation": target_id,
                    "magnitude": shock.magnitude,
                    "timestep": timestep,
                })

        # Apply ongoing shocks
        still_active = []
        for shock in self.active_shocks:
            nation = nations.get(shock.affected_nation)
            if nation and nation.alive:
                self._apply_shock_effect(shock, nation)
            shock.steps_remaining -= 1
            if shock.steps_remaining > 0:
                still_active.append(shock)
            else:
                events.append({
                    "type": "SHOCK_EXPIRED",
                    "shock_type": shock.shock_type.value,
                    "nation": shock.affected_nation,
                    "timestep": timestep,
                })
        self.active_shocks = still_active

    def inject_shock(
        self,
        shock_type: ShockType,
        nation_id: str,
        magnitude: float = 0.5,
        duration_steps: int = 10,
    ) -> None:
        """Manually inject a shock (for scenario injection via API)."""
        shock = ActiveShock(
            shock_type=shock_type,
            affected_nation=nation_id,
            duration_steps=duration_steps,
            steps_remaining=duration_steps,
            magnitude=magnitude,
        )
        self.active_shocks.append(shock)

    def _create_shock(self, shock_type: ShockType, nation_id: str) -> ActiveShock:
        magnitude = float(self.rng.uniform(0.2, 0.8))
        duration = int(self.rng.integers(5, 20))
        return ActiveShock(
            shock_type=shock_type,
            affected_nation=nation_id,
            duration_steps=duration,
            steps_remaining=duration,
            magnitude=magnitude,
        )

    def _apply_shock_effect(self, shock: ActiveShock, nation: NationState) -> None:
        m = shock.magnitude * 0.1  # scale down per-step effect

        if shock.shock_type == ShockType.RESOURCE_DISCOVERY:
            resource = random.choice(["oil", "food", "minerals"])
            nation.resources[resource] = min(1.0, nation.resources.get(resource, 0.5) + m)
            nation.gdp = nation.gdp * (1.0 + m * 0.1)

        elif shock.shock_type == ShockType.PANDEMIC:
            nation.population = max(1.0, nation.population * (1.0 - m * 0.05))
            nation.internal_stability = max(0.0, nation.internal_stability - m * 0.3)
            nation.gdp = nation.gdp * (1.0 - m * 0.05)

        elif shock.shock_type == ShockType.TECH_BREAKTHROUGH:
            nation.tech_level = min(1.0, nation.tech_level + m * 0.2)
            nation.gdp = nation.gdp * (1.0 + m * 0.05)

        elif shock.shock_type == ShockType.NATURAL_DISASTER:
            nation.internal_stability = max(0.0, nation.internal_stability - m * 0.2)
            nation.gdp = nation.gdp * (1.0 - m * 0.08)
            nation.resources["food"] = max(0.0, nation.resources.get("food", 0.5) - m * 0.2)

        elif shock.shock_type == ShockType.FINANCIAL_CRISIS:
            nation.gdp = nation.gdp * (1.0 - m * 0.1)
            nation.internal_stability = max(0.0, nation.internal_stability - m * 0.15)
            for rel in nation.relationships.values():
                rel.trade_volume = max(0.0, rel.trade_volume - m * 0.1)
