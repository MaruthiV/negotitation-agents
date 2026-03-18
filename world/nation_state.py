from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RelationshipVector:
    trade_volume: float = 0.0       # [0, 1]
    alliance_strength: float = 0.0  # [-1, 1]
    hostility: float = 0.0          # [0, 1]
    grievance: float = 0.0          # [0, 1]

    def to_array(self) -> list[float]:
        return [self.trade_volume, self.alliance_strength, self.hostility, self.grievance]

    def clamp(self) -> None:
        self.trade_volume = max(0.0, min(1.0, self.trade_volume))
        self.alliance_strength = max(-1.0, min(1.0, self.alliance_strength))
        self.hostility = max(0.0, min(1.0, self.hostility))
        self.grievance = max(0.0, min(1.0, self.grievance))


ARCHETYPES = ["expansionist", "mercantile", "isolationist", "hegemon"]


@dataclass
class NationState:
    nation_id: str
    gdp: float                          # absolute value, e.g. 1.0 = baseline
    military_strength: float            # [0, 1]
    population: float                   # millions
    resources: dict[str, float] = field(default_factory=lambda: {"oil": 0.5, "food": 0.5, "minerals": 0.5})
    tech_level: float = 0.3             # [0, 1]
    internal_stability: float = 0.7     # [0, 1]; < 0.15 triggers regime change
    territory: float = 0.5             # [0, 1] normalized
    relationships: dict[str, RelationshipVector] = field(default_factory=dict)
    military_spending_pct: float = 0.03
    archetype: str = "mercantile"
    alive: bool = True
    age: int = 0                        # steps since last regime change

    def is_dead(self) -> bool:
        return not self.alive or self.gdp <= 0.0

    def in_regime_crisis(self) -> bool:
        return self.internal_stability < 0.15

    def get_relationship(self, target_id: str) -> RelationshipVector:
        if target_id not in self.relationships:
            self.relationships[target_id] = RelationshipVector()
        return self.relationships[target_id]

    def copy(self) -> "NationState":
        import copy
        return copy.deepcopy(self)
