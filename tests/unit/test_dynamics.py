import pytest
import numpy as np
from world.nation_state import NationState, RelationshipVector
from world.dynamics.trade import TradeResolver
from world.dynamics.military import MilitaryResolver
from world.dynamics.diplomacy import DiplomacyResolver
from world.dynamics.internal import InternalDynamicsResolver


def make_nation(nid, gdp=1.0, mil=0.5, stability=0.7, territory=0.5) -> NationState:
    n = NationState(
        nation_id=nid,
        gdp=gdp,
        military_strength=mil,
        population=100.0,
        resources={"oil": 0.5, "food": 0.5, "minerals": 0.5},
        tech_level=0.3,
        internal_stability=stability,
        territory=territory,
        relationships={},
    )
    return n


def init_relations(nations: dict) -> None:
    for a_id, a in nations.items():
        for b_id in nations:
            if a_id != b_id:
                a.relationships[b_id] = RelationshipVector()


# --- Trade ---

def test_trade_mutual_proposal_increases_volume():
    a = make_nation("a")
    b = make_nation("b")
    nations = {"a": a, "b": b}
    init_relations(nations)
    initial_vol = a.get_relationship("b").trade_volume

    resolver = TradeResolver()
    resolver.resolve(nations, {"a": {"b"}, "b": {"a"}}, {})

    assert a.get_relationship("b").trade_volume > initial_vol


def test_trade_sanctions_decrease_volume():
    a = make_nation("a")
    b = make_nation("b")
    nations = {"a": a, "b": b}
    init_relations(nations)
    a.get_relationship("b").trade_volume = 0.5
    b.get_relationship("a").trade_volume = 0.5

    resolver = TradeResolver()
    resolver.resolve(nations, {}, {"a": {"b"}})

    assert a.get_relationship("b").trade_volume < 0.5


def test_trade_boosts_gdp():
    a = make_nation("a", gdp=1.0)
    b = make_nation("b", gdp=1.0)
    nations = {"a": a, "b": b}
    init_relations(nations)
    a.get_relationship("b").trade_volume = 0.5
    b.get_relationship("a").trade_volume = 0.5
    prev_gdp_a = a.gdp

    resolver = TradeResolver()
    resolver.resolve(nations, {"a": {"b"}, "b": {"a"}}, {})

    assert a.gdp > prev_gdp_a


# --- Military ---

def test_war_resolution_both_pay_costs():
    rng = np.random.default_rng(42)
    a = make_nation("a", gdp=1.0, mil=0.8)
    b = make_nation("b", gdp=1.0, mil=0.2)
    nations = {"a": a, "b": b}
    init_relations(nations)

    resolver = MilitaryResolver()
    events = []
    resolver.resolve_wars(nations, {("a", "b")}, rng, events)

    assert a.gdp < 1.0
    assert b.gdp < 1.0
    assert any(e["type"] == "WAR_RESOLVED" for e in events)


def test_military_buildup():
    n = make_nation("a", mil=0.3)
    resolver = MilitaryResolver()
    resolver.apply_military_buildup(n, budget_military_frac=0.5)
    assert n.military_strength > 0.3


def test_war_winner_gains_territory():
    rng = np.random.default_rng(0)
    # Run many times to ensure winner gains territory on average
    a_territories = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        a = make_nation("a", mil=0.99, territory=0.5)
        b = make_nation("b", mil=0.01, territory=0.5)
        nations = {"a": a, "b": b}
        init_relations(nations)
        resolver = MilitaryResolver()
        resolver.resolve_wars(nations, {("a", "b")}, rng, [])
        a_territories.append(a.territory)
    assert sum(t > 0.5 for t in a_territories) > 12  # a wins most of the time


# --- Diplomacy ---

def test_declare_war_sets_hostility():
    a = make_nation("a")
    b = make_nation("b")
    nations = {"a": a, "b": b}
    init_relations(nations)

    resolver = DiplomacyResolver()
    pending_wars = set()
    resolver.resolve(nations, {"a": {"b": 5}}, pending_wars)  # 5 = declare_war

    assert a.get_relationship("b").hostility == 1.0
    assert ("a", "b") in pending_wars


def test_negotiate_peace_reduces_hostility():
    a = make_nation("a")
    b = make_nation("b")
    nations = {"a": a, "b": b}
    init_relations(nations)
    a.get_relationship("b").hostility = 0.8
    b.get_relationship("a").hostility = 0.8

    resolver = DiplomacyResolver()
    pending_wars = set()
    resolver.resolve(nations, {"a": {"b": 6}}, pending_wars)

    assert a.get_relationship("b").hostility < 0.8


# --- Internal ---

def test_internal_dev_raises_stability():
    n = make_nation("a", stability=0.5)
    resolver = InternalDynamicsResolver()
    pending = []
    resolver.resolve(n, {"internal_dev": 1.0, "military": 0.0, "tech_rd": 0.0}, pending)
    assert n.internal_stability > 0.5


def test_regime_change_triggered():
    n = make_nation("a", stability=0.1)
    resolver = InternalDynamicsResolver()
    pending = []
    # Low stability → should trigger regime change
    resolver.resolve(n, {"internal_dev": 0.0, "military": 0.0, "tech_rd": 0.0}, pending)
    assert "a" in pending
