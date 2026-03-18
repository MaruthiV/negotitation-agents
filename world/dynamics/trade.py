from __future__ import annotations

import numpy as np

from world.nation_state import NationState

# Comparative advantage multiplier per resource type
RESOURCE_ADVANTAGE = {"oil": 0.04, "food": 0.03, "minerals": 0.03}
BASE_GDP_BOOST_PER_TRADE = 0.02


class TradeResolver:
    """Resolve trade — bilateral volume adjusts from mutual proposals and sanctions."""

    def resolve(
        self,
        nations: dict[str, NationState],
        trade_proposals: dict[str, set[str]],  # nation_id -> set of targets it proposed to
        sanctions: dict[str, set[str]],         # nation_id -> set of targets it sanctioned
    ) -> None:
        alive_ids = [nid for nid, n in nations.items() if n.alive]

        for i, aid in enumerate(alive_ids):
            for bid in alive_ids[i + 1:]:
                a, b = nations[aid], nations[bid]
                rel_ab = a.get_relationship(bid)
                rel_ba = b.get_relationship(aid)

                # Mutual trade proposal → volume increases more
                mutual = aid in trade_proposals.get(bid, set()) and bid in trade_proposals.get(aid, set())
                unilateral = (
                    aid in trade_proposals.get(bid, set())
                    or bid in trade_proposals.get(aid, set())
                )
                sanctioned = (
                    aid in sanctions.get(bid, set())
                    or bid in sanctions.get(aid, set())
                )

                if sanctioned:
                    delta = -0.08
                elif mutual:
                    delta = +0.05
                elif unilateral:
                    delta = +0.01
                else:
                    # Natural drift toward zero
                    delta = -0.005

                new_vol = max(0.0, min(1.0, rel_ab.trade_volume + delta))
                rel_ab.trade_volume = new_vol
                rel_ba.trade_volume = new_vol

                # GDP boost from trade
                if new_vol > 0.01:
                    ca_a = sum(a.resources.get(r, 0) * mult for r, mult in RESOURCE_ADVANTAGE.items())
                    ca_b = sum(b.resources.get(r, 0) * mult for r, mult in RESOURCE_ADVANTAGE.items())
                    boost_a = new_vol * (BASE_GDP_BOOST_PER_TRADE + ca_b * 0.01)
                    boost_b = new_vol * (BASE_GDP_BOOST_PER_TRADE + ca_a * 0.01)
                    a.gdp += boost_a
                    b.gdp += boost_b
