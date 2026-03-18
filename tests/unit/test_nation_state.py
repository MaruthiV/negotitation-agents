import pytest
from world.nation_state import NationState, RelationshipVector, ARCHETYPES


def make_nation(nid="alpha") -> NationState:
    return NationState(
        nation_id=nid,
        gdp=1.0,
        military_strength=0.5,
        population=100.0,
        resources={"oil": 0.5, "food": 0.6, "minerals": 0.4},
        tech_level=0.3,
        internal_stability=0.7,
        territory=0.5,
        relationships={},
    )


def test_relationship_vector_clamp():
    rel = RelationshipVector(trade_volume=1.5, alliance_strength=2.0, hostility=-0.1, grievance=1.1)
    rel.clamp()
    assert rel.trade_volume == 1.0
    assert rel.alliance_strength == 1.0
    assert rel.hostility == 0.0
    assert rel.grievance == 1.0


def test_relationship_to_array():
    rel = RelationshipVector(0.3, 0.5, 0.1, 0.2)
    arr = rel.to_array()
    assert len(arr) == 4
    assert arr == [0.3, 0.5, 0.1, 0.2]


def test_nation_get_relationship_creates():
    n = make_nation()
    rel = n.get_relationship("beta")
    assert isinstance(rel, RelationshipVector)
    assert "beta" in n.relationships


def test_nation_is_dead():
    n = make_nation()
    assert not n.is_dead()
    n.alive = False
    assert n.is_dead()


def test_nation_regime_crisis():
    n = make_nation()
    n.internal_stability = 0.1
    assert n.in_regime_crisis()
    n.internal_stability = 0.5
    assert not n.in_regime_crisis()


def test_nation_copy_is_deep():
    n = make_nation()
    n.get_relationship("beta").trade_volume = 0.5
    copy = n.copy()
    copy.get_relationship("beta").trade_volume = 0.9
    assert n.get_relationship("beta").trade_volume == 0.5


def test_archetypes_list():
    assert len(ARCHETYPES) == 4
    assert "mercantile" in ARCHETYPES
