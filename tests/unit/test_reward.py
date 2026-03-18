import pytest
import math
from world.nation_state import NationState
from world.reward import RewardCalculator, ARCHETYPE_WEIGHTS, RewardWeights


def make_nation(gdp=1.0, mil=0.5, stability=0.7, territory=0.5, alive=True) -> NationState:
    return NationState(
        nation_id="alpha",
        gdp=gdp,
        military_strength=mil,
        population=100.0,
        tech_level=0.3,
        internal_stability=stability,
        territory=territory,
        relationships={},
        alive=alive,
    )


def test_reward_death():
    calc = RewardCalculator(ARCHETYPE_WEIGHTS["mercantile"])
    prev = make_nation()
    curr = make_nation(alive=False)
    assert calc.compute(prev, curr) == -10.0


def test_reward_bounds_normal():
    calc = RewardCalculator(ARCHETYPE_WEIGHTS["expansionist"])
    prev = make_nation(gdp=1.0, mil=0.5, stability=0.7)
    curr = make_nation(gdp=1.1, mil=0.55, stability=0.72)
    r = calc.compute(prev, curr)
    assert -15.0 <= r <= 5.0


def test_reward_bounds_extreme_gdp_crash():
    calc = RewardCalculator(ARCHETYPE_WEIGHTS["hegemon"])
    prev = make_nation(gdp=10.0)
    curr = make_nation(gdp=0.01)
    r = calc.compute(prev, curr)
    assert r >= -15.0


def test_reward_bounds_extreme_growth():
    calc = RewardCalculator(ARCHETYPE_WEIGHTS["mercantile"])
    prev = make_nation(gdp=0.01)
    curr = make_nation(gdp=100.0)
    r = calc.compute(prev, curr)
    assert r <= 5.0


def test_all_archetypes_have_weights():
    for archetype in ["expansionist", "mercantile", "isolationist", "hegemon"]:
        assert archetype in ARCHETYPE_WEIGHTS
        w = ARCHETYPE_WEIGHTS[archetype]
        assert isinstance(w, RewardWeights)
