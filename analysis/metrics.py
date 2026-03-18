from __future__ import annotations

import math
from typing import Optional


class EmergenceMetrics:
    """
    Compute emergence metrics from world snapshots.
    - arms_race_detection: rising mean military spending
    - liberal_peace_index: high trade + low wars among trading partners
    - power_concentration (Gini of GDP)
    """

    def __init__(self):
        self._war_history: list[int] = []  # war counts per call
        self._step_history: list[int] = []

    def compute(self, snapshot: dict) -> dict[str, float]:
        nations = snapshot.get("nations", {})
        events = snapshot.get("events", [])

        alive = {nid: n for nid, n in nations.items() if n.get("alive", True)}
        if not alive:
            return {}

        # Mean trade volume across all pairs
        trade_vols = []
        for n in alive.values():
            for rel in n.get("relationships", {}).values():
                trade_vols.append(rel.get("trade_volume", 0.0) if isinstance(rel, dict) else rel[0])
        mean_trade = float(sum(trade_vols) / len(trade_vols)) if trade_vols else 0.0

        # Mean military strength
        mean_mil = float(sum(n["military_strength"] for n in alive.values()) / len(alive))

        # Mean stability
        mean_stability = float(sum(n["internal_stability"] for n in alive.values()) / len(alive))

        # GDP Gini
        gdps = sorted(n["gdp"] for n in alive.values())
        gini = self._gini(gdps)

        # Wars in this snapshot's events
        wars = sum(1 for e in events if e.get("type") == "WAR_RESOLVED")
        self._war_history.append(wars)

        # Arms race: mean military trend over last 10 calls
        arms_race_score = mean_mil  # simplified

        # Liberal peace: high trade, low hostility
        hostilities = []
        for n in alive.values():
            for rel in n.get("relationships", {}).values():
                h = rel.get("hostility", 0.0) if isinstance(rel, dict) else rel[2]
                hostilities.append(h)
        mean_hostility = float(sum(hostilities) / len(hostilities)) if hostilities else 0.0
        liberal_peace_index = mean_trade * (1 - mean_hostility)

        # Dominance: share of total GDP held by richest nation
        total_gdp = sum(gdps)
        max_gdp_share = max(gdps) / max(total_gdp, 1e-6)

        return {
            "mean_trade_volume": mean_trade,
            "mean_military_strength": mean_mil,
            "mean_stability": mean_stability,
            "gdp_gini": gini,
            "liberal_peace_index": liberal_peace_index,
            "arms_race_score": arms_race_score,
            "max_gdp_share": max_gdp_share,
            "wars_this_eval": wars,
        }

    @staticmethod
    def _gini(sorted_values: list[float]) -> float:
        n = len(sorted_values)
        if n == 0:
            return 0.0
        total = sum(sorted_values)
        if total < 1e-9:
            return 0.0
        numer = sum((i + 1) * v for i, v in enumerate(sorted_values))
        return (2 * numer / (n * total)) - (n + 1) / n

    def arms_race_detection(self, military_history: list[float], window: int = 10) -> bool:
        """Returns True if mean military strength is trending up."""
        if len(military_history) < window:
            return False
        recent = military_history[-window:]
        return recent[-1] > recent[0] * 1.1  # 10% increase

    def liberal_peace_index(self, snapshot: dict) -> float:
        result = self.compute(snapshot)
        return result.get("liberal_peace_index", 0.0)

    def power_concentration(self, snapshot: dict) -> float:
        result = self.compute(snapshot)
        return result.get("gdp_gini", 0.0)
