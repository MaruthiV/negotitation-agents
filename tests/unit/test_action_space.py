import pytest
import numpy as np
from world.action_space import make_action_space, ActionEncoder, DecodedAction, N_BUDGET_CHANNELS, N_DIPLOMATIC_OPTIONS


def test_action_space_shape():
    n_nations = 5
    space = make_action_space(n_nations)
    assert "budget_allocation" in space.spaces
    assert "diplomatic_actions" in space.spaces
    assert space["budget_allocation"].shape == (N_BUDGET_CHANNELS,)
    assert space["diplomatic_actions"].shape == (n_nations - 1,)


def test_action_space_sample():
    space = make_action_space(4)
    sample = space.sample()
    assert "budget_allocation" in sample
    assert "diplomatic_actions" in sample
    assert len(sample["budget_allocation"]) == N_BUDGET_CHANNELS
    assert len(sample["diplomatic_actions"]) == 3


def test_action_encoder_normalizes_budget():
    enc = ActionEncoder(["a", "b", "c", "d"], "a")
    raw = {
        "budget_allocation": np.array([2.0, 3.0, 1.0, 4.0, 0.0]),
        "diplomatic_actions": [0, 1, 2],
    }
    decoded = enc.decode(raw)
    assert abs(decoded.budget.sum() - 1.0) < 1e-5
    assert all(b >= 0 for b in decoded.budget)


def test_action_encoder_all_zeros_budget():
    enc = ActionEncoder(["a", "b"], "a")
    raw = {
        "budget_allocation": np.zeros(N_BUDGET_CHANNELS),
        "diplomatic_actions": [0],
    }
    decoded = enc.decode(raw)
    # Should default to uniform
    assert abs(decoded.budget.sum() - 1.0) < 1e-5


def test_action_encoder_targets():
    enc = ActionEncoder(["a", "b", "c"], "a")
    assert enc.targets == ["b", "c"]
    assert enc.get_target(0) == "b"
    assert enc.get_target(1) == "c"


def test_diplomatic_action_names():
    enc = ActionEncoder(["a", "b"], "a")
    assert enc.diplomatic_action_name(0) == "do_nothing"
    assert enc.diplomatic_action_name(5) == "declare_war"
    assert enc.diplomatic_action_name(6) == "negotiate_peace"
