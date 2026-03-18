from __future__ import annotations

import math
import numpy as np

from world.nation_state import NationState


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class MilitaryResolver:
    """Probabilistic war resolution — prevents pure arms-race equilibria."""

    GDP_ATTACKER_COST = 0.08
    GDP_DEFENDER_COST = 0.05
    MIL_ATTACKER_COST = 0.1
    MIL_DEFENDER_COST = 0.08
    TERRITORY_GAIN = 0.05
    RESOURCE_TRANSFER = 0.1

    def resolve_wars(
        self,
        nations: dict[str, NationState],
        pending_wars: set[tuple[str, str]],
        rng: np.random.Generator,
        events: list[dict],
    ) -> None:
        """Resolve all declared wars. Mutates nations, appends events."""
        resolved = set()
        for attacker_id, defender_id in list(pending_wars):
            pair = frozenset([attacker_id, defender_id])
            if pair in resolved:
                continue
            resolved.add(pair)

            attacker = nations.get(attacker_id)
            defender = nations.get(defender_id)
            if attacker is None or not attacker.alive:
                continue
            if defender is None or not defender.alive:
                continue

            self._fight(attacker, defender, rng, events)

        pending_wars.clear()

    def apply_military_buildup(
        self,
        nation: NationState,
        budget_military_frac: float,
    ) -> None:
        """Update military strength and spending pct from budget allocation."""
        nation.military_spending_pct = budget_military_frac * 0.2  # cap at 20%
        # Military strength grows proportional to spending, decays slightly each step
        growth = budget_military_frac * 0.05
        decay = 0.005
        nation.military_strength = min(1.0, max(0.0, nation.military_strength + growth - decay))

    def _fight(
        self,
        attacker: NationState,
        defender: NationState,
        rng: np.random.Generator,
        events: list[dict],
    ) -> None:
        # Compute alliance bonuses
        ally_bonus_a = sum(
            r.alliance_strength
            for r in attacker.relationships.values()
            if r.alliance_strength > 0
        ) * 0.1

        ally_bonus_d = sum(
            r.alliance_strength
            for r in defender.relationships.values()
            if r.alliance_strength > 0
        ) * 0.1

        p_attacker_wins = _sigmoid(
            (attacker.military_strength + ally_bonus_a)
            - (defender.military_strength + ally_bonus_d)
        )

        attacker_wins = rng.random() < p_attacker_wins

        # Both pay costs
        attacker.gdp = max(0.01, attacker.gdp - attacker.gdp * self.GDP_ATTACKER_COST)
        defender.gdp = max(0.01, defender.gdp - defender.gdp * self.GDP_DEFENDER_COST)
        attacker.military_strength = max(0.0, attacker.military_strength - self.MIL_ATTACKER_COST)
        defender.military_strength = max(0.0, defender.military_strength - self.MIL_DEFENDER_COST)

        # Internal stability suffers
        attacker.internal_stability = max(0.0, attacker.internal_stability - 0.05)
        defender.internal_stability = max(0.0, defender.internal_stability - 0.08)

        if attacker_wins:
            winner, loser = attacker, defender
        else:
            winner, loser = defender, attacker

        # Transfer territory and resources
        gain = min(self.TERRITORY_GAIN, loser.territory * 0.5)
        winner.territory = min(1.0, winner.territory + gain)
        loser.territory = max(0.0, loser.territory - gain)

        for resource in ["oil", "food", "minerals"]:
            transfer = loser.resources.get(resource, 0.0) * self.RESOURCE_TRANSFER
            winner.resources[resource] = min(1.0, winner.resources.get(resource, 0.0) + transfer)
            loser.resources[resource] = max(0.0, loser.resources.get(resource, 0.0) - transfer)

        # Loser gains grievance
        rel_loser = loser.get_relationship(winner.nation_id)
        rel_loser.grievance = min(1.0, rel_loser.grievance + 0.2)

        # Check if loser is eliminated
        if loser.territory <= 0.02 or loser.gdp <= 0.05:
            loser.alive = False

        events.append({
            "type": "WAR_RESOLVED",
            "attacker": attacker.nation_id,
            "defender": defender.nation_id,
            "winner": winner.nation_id,
            "p_attacker_wins": p_attacker_wins,
        })
