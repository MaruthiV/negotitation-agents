"""
PettingZoo API compliance tests.
Verifies that GeopoliticalEnv passes the standard API contract.
"""
import pytest
import numpy as np

from world.geopolitical_env import GeopoliticalEnv


def make_env(**kwargs) -> GeopoliticalEnv:
    return GeopoliticalEnv(
        nation_ids=["alpha", "beta", "gamma"],
        max_steps=50,
        seed=42,
        **kwargs,
    )


def test_env_reset():
    env = make_env()
    env.reset()
    assert len(env.agents) > 0
    assert env.agent_selection in env.agents


def test_env_step_basic():
    env = make_env()
    env.reset()

    steps = 0
    while env.agents and steps < 200:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            action = env.action_space(agent).sample()
            env.step(action)
        steps += 1


def test_observation_shape():
    env = make_env()
    env.reset()
    agent = env.agent_selection
    obs = env.observe(agent)
    expected_dim = env.observation_space(agent).shape[0]
    assert obs.shape == (expected_dim,), f"obs.shape={obs.shape}, expected ({expected_dim},)"


def test_observation_space_contains_obs():
    env = make_env()
    env.reset()
    for agent in env.agents:
        obs = env.observe(agent)
        # Observation space is unbounded Box, so just check dtype
        assert obs.dtype == np.float32


def test_action_space_sample_accepted():
    env = make_env()
    env.reset()
    agent = env.agent_selection
    action = env.action_space(agent).sample()
    # Should not raise
    env.step(action)


def test_rewards_are_numeric():
    env = make_env()
    env.reset()
    agent = env.agent_selection
    action = env.action_space(agent).sample()
    env.step(action)
    # Rewards are set after world step (when last agent acts)
    # Just check types
    for r in env.rewards.values():
        assert isinstance(r, (int, float))


def test_terminations_and_truncations_are_bool():
    env = make_env()
    env.reset()
    for t in env.terminations.values():
        assert isinstance(t, bool)
    for t in env.truncations.values():
        assert isinstance(t, bool)


def test_full_episode_completes():
    env = make_env(max_steps=30)
    env.reset()
    steps = 0
    while env.agents:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            env.step(env.action_space(agent).sample())
        steps += 1
        if steps > 500:
            pytest.fail("Episode did not terminate within 500 steps")


def test_world_snapshot_structure():
    env = make_env()
    env.reset()
    # Run a few steps
    for _ in range(15):
        if not env.agents:
            break
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            env.step(env.action_space(agent).sample())

    snapshot = env.get_world_snapshot()
    assert "step" in snapshot
    assert "nations" in snapshot
    assert "events" in snapshot
    for nid in env.nation_ids:
        assert nid in snapshot["nations"]


def test_pettingzoo_api_test():
    """Run the official PettingZoo api_test."""
    try:
        from pettingzoo.test import api_test
        env = make_env(max_steps=30)
        api_test(env, num_cycles=100, verbose_progress=False)
    except ImportError:
        pytest.skip("pettingzoo.test not available")
