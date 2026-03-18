import pytest
import numpy as np
import torch
from agents.continual.clear import CLEARBuffer, Experience
from agents.continual.ewc import EWCRegularizer
from agents.networks import ActorNetwork


def make_experience(context_id="ctx_a", priority=1.0) -> Experience:
    return Experience(
        obs=np.random.rand(25).astype(np.float32),
        budget_action=np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        diplomatic_action=np.zeros(4, dtype=np.int64),
        log_prob=-1.0,
        reward=0.5,
        value=0.4,
        done=False,
        context_id=context_id,
        priority=priority,
    )


# --- CLEARBuffer ---

def test_clear_buffer_add_and_size():
    buf = CLEARBuffer()
    for _ in range(10):
        buf.add(make_experience("ctx_a"))
    assert buf.size("ctx_a") == 10
    assert buf.size() == 10


def test_clear_buffer_partitioned_by_context():
    buf = CLEARBuffer()
    for _ in range(5):
        buf.add(make_experience("ctx_a"))
    for _ in range(5):
        buf.add(make_experience("ctx_b"))
    assert buf.size("ctx_a") == 5
    assert buf.size("ctx_b") == 5
    assert buf.size() == 10


def test_clear_buffer_mixed_batch_includes_replay():
    buf = CLEARBuffer(replay_ratio=0.3)
    # Add experiences from ctx_b
    for _ in range(50):
        buf.add(make_experience("ctx_b"))

    current_rollout = [make_experience("ctx_a") for _ in range(50)]
    mixed = buf.build_mixed_batch(current_rollout, "ctx_a")

    assert len(mixed) > len(current_rollout)  # has replay
    ctx_b_count = sum(1 for e in mixed if e.context_id == "ctx_b")
    assert ctx_b_count > 0


def test_clear_buffer_no_replay_when_only_current_context():
    buf = CLEARBuffer(replay_ratio=0.3)
    current_rollout = [make_experience("ctx_a") for _ in range(20)]
    mixed = buf.build_mixed_batch(current_rollout, "ctx_a")
    assert len(mixed) == 20  # no other contexts


def test_clear_buffer_respects_maxlen():
    buf = CLEARBuffer(max_per_context=10)
    for _ in range(20):
        buf.add(make_experience("ctx_a"))
    assert buf.size("ctx_a") == 10


# --- EWCRegularizer ---

def test_ewc_penalty_zero_before_consolidation():
    model = ActorNetwork(obs_dim=25, n_budget_channels=5, n_targets=4, n_diplomatic_options=7)
    ewc = EWCRegularizer(importance=100.0)
    penalty = ewc.penalty(model)
    assert penalty.item() == 0.0


def test_ewc_penalty_positive_after_param_change():
    """After consolidation, changing params should produce positive penalty."""
    obs_dim = 25
    model = ActorNetwork(obs_dim=obs_dim, n_budget_channels=5, n_targets=4, n_diplomatic_options=7)

    # Create fake dataloader
    obs = torch.randn(16, obs_dim)
    budget = torch.softmax(torch.randn(16, 5), dim=1)
    diplomatic = torch.randint(0, 7, (16, 4))

    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(obs, budget, diplomatic)
    loader = DataLoader(ds, batch_size=8)

    ewc = EWCRegularizer(importance=100.0)
    ewc.consolidate(model, loader)

    # Change params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 0.5)

    penalty = ewc.penalty(model)
    assert penalty.item() > 0.0
