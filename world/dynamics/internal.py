from __future__ import annotations

from world.nation_state import NationState

# Stability changes from budget allocation
INTERNAL_DEV_GAIN = 0.04
TECH_STABILITY_BOOST = 0.01
MILITARY_STABILITY_COST = 0.005
NATURAL_DECAY = 0.003

# GDP growth from tech
TECH_GDP_RATE = 0.015


class InternalDynamicsResolver:
    """Handle per-step internal nation dynamics."""

    def resolve(
        self,
        nation: NationState,
        budget: dict[str, float],  # channel -> fraction
        regime_change_pending: list[str],
    ) -> None:
        """Mutate nation in-place. Append to regime_change_pending if triggered."""
        internal_dev = budget.get("internal_dev", 0.0)
        tech_rd = budget.get("tech_rd", 0.0)
        military = budget.get("military", 0.0)

        # Stability dynamics
        stability_delta = (
            internal_dev * INTERNAL_DEV_GAIN
            + tech_rd * TECH_STABILITY_BOOST
            - military * MILITARY_STABILITY_COST
            - NATURAL_DECAY
        )
        nation.internal_stability = max(0.0, min(1.0, nation.internal_stability + stability_delta))

        # Tech growth
        nation.tech_level = min(1.0, nation.tech_level + tech_rd * 0.01)

        # GDP natural growth from tech and internal investment
        gdp_growth_rate = tech_rd * TECH_GDP_RATE + internal_dev * 0.01
        nation.gdp = nation.gdp * (1.0 + gdp_growth_rate)

        # Age increment
        nation.age += 1

        # Regime change trigger
        if nation.in_regime_crisis() and nation.alive:
            regime_change_pending.append(nation.nation_id)
