from __future__ import annotations

import numpy as np
import gymnasium

from world.nation_state import NationState

# Own state dimensions (25):
# gdp(1), military_strength(1), population(1), resources(3), tech_level(1),
# internal_stability(1), territory(1), military_spending_pct(1),
# archetype_onehot(4), alive(1), age_norm(1),
# relationships_summary: (avg_trade, avg_alliance, avg_hostility, avg_grievance)(4)
# = 1+1+1+3+1+1+1+1+4+1+1+4 = 20 — padded to 25 with zeros
OWN_STATE_DIM = 25

# Per other nation partial obs (15 dims) + relationship row (4 dims) = 19
OTHER_NATION_DIM = 19

ARCHETYPE_ORDER = ["expansionist", "mercantile", "isolationist", "hegemon"]


def _archetype_onehot(archetype: str) -> list[float]:
    vec = [0.0] * 4
    if archetype in ARCHETYPE_ORDER:
        vec[ARCHETYPE_ORDER.index(archetype)] = 1.0
    return vec


class ObservationBuilder:
    def __init__(self, nation_ids: list[str], noise_std: float = 0.05):
        self.nation_ids = nation_ids
        self.noise_std = noise_std
        n_others = len(nation_ids) - 1
        self.obs_dim = OWN_STATE_DIM + OTHER_NATION_DIM * n_others

    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def build(
        self,
        observer_id: str,
        nations: dict[str, NationState],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        obs = self_id = observer_id
        own = nations[observer_id]
        own_vec = self._encode_own(own)

        others_vec = []
        for nid in self.nation_ids:
            if nid == observer_id:
                continue
            other = nations[nid]
            rel = own.get_relationship(nid)
            partial = self._encode_other_partial(other, rng)
            others_vec.extend(partial)
            others_vec.extend(rel.to_array())

        full = np.array(own_vec + others_vec, dtype=np.float32)
        assert len(full) == self.obs_dim, f"obs dim mismatch: {len(full)} != {self.obs_dim}"
        return full

    def _encode_own(self, n: NationState) -> list[float]:
        rel_vals = list(n.relationships.values())
        avg_trade = np.mean([r.trade_volume for r in rel_vals]) if rel_vals else 0.0
        avg_alliance = np.mean([r.alliance_strength for r in rel_vals]) if rel_vals else 0.0
        avg_hostility = np.mean([r.hostility for r in rel_vals]) if rel_vals else 0.0
        avg_grievance = np.mean([r.grievance for r in rel_vals]) if rel_vals else 0.0

        vec = [
            n.gdp,
            n.military_strength,
            n.population / 1000.0,  # normalize
            n.resources.get("oil", 0.5),
            n.resources.get("food", 0.5),
            n.resources.get("minerals", 0.5),
            n.tech_level,
            n.internal_stability,
            n.territory,
            n.military_spending_pct,
            *_archetype_onehot(n.archetype),
            float(n.alive),
            min(n.age / 1000.0, 1.0),
            float(avg_trade),
            float(avg_alliance),
            float(avg_hostility),
            float(avg_grievance),
        ]
        # Pad to OWN_STATE_DIM
        vec += [0.0] * (OWN_STATE_DIM - len(vec))
        return vec[:OWN_STATE_DIM]

    def _encode_other_partial(
        self, n: NationState, rng: np.random.Generator
    ) -> list[float]:
        """15 dims with Gaussian noise for imperfect information."""
        noise = rng.normal(0, self.noise_std, 8)
        vec = [
            n.gdp + noise[0],
            n.military_strength + noise[1],
            n.population / 1000.0 + noise[2],
            n.resources.get("oil", 0.5) + noise[3],
            n.resources.get("food", 0.5) + noise[4],
            n.resources.get("minerals", 0.5) + noise[5],
            n.tech_level + noise[6],
            n.internal_stability + noise[7],
            n.territory,
            n.military_spending_pct,
            *_archetype_onehot(n.archetype),
            float(n.alive),
        ]
        vec += [0.0] * (15 - len(vec))
        return vec[:15]


class NaturalLanguageObsBuilder:
    """
    Converts NationState + world context into a human-readable text briefing
    for consumption by the LLM strategist.
    """

    def build_text(
        self,
        observer_id: str,
        nations: dict[str, NationState],
        recent_events: list[dict] | None = None,
        step: int = 0,
    ) -> str:
        own = nations.get(observer_id)
        if own is None:
            return f"No data available for {observer_id}."

        lines: list[str] = [
            f"## Strategic Briefing — {observer_id.capitalize()} (Step {step})",
            f"**Archetype**: {own.archetype}",
            "",
            "### Own Status",
            f"- GDP: {own.gdp:.3f}",
            f"- Military Strength: {own.military_strength * 100:.0f}%",
            f"- Internal Stability: {own.internal_stability * 100:.0f}%"
            + (" ⚠ CRISIS" if own.internal_stability < 0.2 else ""),
            f"- Territory: {own.territory * 100:.0f}%",
            f"- Tech Level: {own.tech_level * 100:.0f}%",
            f"- Military Spending: {own.military_spending_pct * 100:.1f}% of GDP",
            "- Resources: "
            + ", ".join(
                f"{k.capitalize()} {v * 100:.0f}%"
                for k, v in sorted(own.resources.items())
            ),
            "",
            "### Relationships",
        ]

        for other_id, rel in own.relationships.items():
            other = nations.get(other_id)
            status_tags: list[str] = []
            if rel.hostility > 0.5:
                status_tags.append("HOSTILE")
            if rel.alliance_strength > 0.4:
                status_tags.append("ALLIED")
            if rel.trade_volume > 0.4:
                status_tags.append("KEY TRADE PARTNER")
            if other and not other.alive:
                status_tags.append("ELIMINATED")
            tag_str = f" ({', '.join(status_tags)})" if status_tags else ""
            lines.append(
                f"- **{other_id.capitalize()}**: Trade={rel.trade_volume:.2f}, "
                f"Alliance={rel.alliance_strength:+.2f}, Hostility={rel.hostility:.2f}, "
                f"Grievance={rel.grievance:.2f}{tag_str}"
            )

        if recent_events:
            lines += ["", "### Recent Events"]
            for ev in recent_events[-5:]:
                ev_type = ev.get("type", "UNKNOWN")
                detail_parts = [f"{k}={v}" for k, v in ev.items() if k != "type"]
                lines.append(f"- {ev_type}" + (f": {', '.join(detail_parts)}" if detail_parts else ""))

        return "\n".join(lines)
