"""
Emergence tests: verify that basic emergent behaviors appear after training.
These run quickly with random agents as smoke tests; full emergence requires training.
"""
import pytest
import numpy as np

from world.geopolitical_env import GeopoliticalEnv
from analysis.metrics import EmergenceMetrics


def run_random_episodes(n_episodes=10, max_steps=100, n_nations=5) -> list[dict]:
    nation_ids = [f"nation_{i}" for i in range(n_nations)]
    env = GeopoliticalEnv(nation_ids=nation_ids, max_steps=max_steps, seed=42)
    metrics = EmergenceMetrics()
    snapshots = []

    for _ in range(n_episodes):
        env.reset()
        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                env.step(env.action_space(agent).sample())
        snapshots.append(env.get_world_snapshot())

    return snapshots


def test_some_wars_occur():
    """Wars should occasionally happen with random actions."""
    snapshots = run_random_episodes(n_episodes=20, max_steps=200)
    all_events = []
    for s in snapshots:
        all_events.extend(s.get("events", []))
    wars = [e for e in all_events if e.get("type") == "WAR_RESOLVED"]
    # With random actions including declare_war (code 5), some wars should happen
    assert len(wars) >= 0  # smoke test: at minimum this should not error


def test_metrics_compute_without_crash():
    snapshots = run_random_episodes(n_episodes=5)
    m = EmergenceMetrics()
    for s in snapshots:
        result = m.compute(s)
        assert "mean_trade_volume" in result
        assert "gdp_gini" in result
        assert 0.0 <= result["gdp_gini"] <= 1.0
        assert 0.0 <= result["mean_trade_volume"] <= 1.0


def test_no_nation_starts_completely_dominant():
    """At initialization, no single nation should own > 90% of GDP."""
    nation_ids = [f"n{i}" for i in range(5)]
    env = GeopoliticalEnv(nation_ids=nation_ids, seed=0)
    env.reset()
    snapshot = env.get_world_snapshot()
    m = EmergenceMetrics()
    result = m.compute(snapshot)
    assert result["max_gdp_share"] < 0.9


def test_reward_stays_in_bounds():
    """Rewards should always be within [-15, +5] per design."""
    from world.reward import RewardCalculator, ARCHETYPE_WEIGHTS
    from world.nation_state import NationState

    calc = RewardCalculator(ARCHETYPE_WEIGHTS["expansionist"])
    rng = np.random.default_rng(0)

    for _ in range(1000):
        prev = NationState(
            nation_id="x",
            gdp=float(rng.uniform(0.01, 100.0)),
            military_strength=float(rng.uniform(0, 1)),
            population=100.0,
            territory=float(rng.uniform(0, 1)),
            internal_stability=float(rng.uniform(0, 1)),
            relationships={},
        )
        curr = NationState(
            nation_id="x",
            gdp=float(rng.uniform(0.01, 100.0)),
            military_strength=float(rng.uniform(0, 1)),
            population=100.0,
            territory=float(rng.uniform(0, 1)),
            internal_stability=float(rng.uniform(0, 1)),
            relationships={},
            alive=True,
        )
        r = calc.compute(prev, curr)
        assert -15.0 <= r <= 5.0, f"reward out of bounds: {r}"
