"""
Unit tests for Hybrid LLM + RL agent.

Verifies:
1. Fallback works when Ollama is offline (strategic mode = economic_focus)
2. Strategic embedding shape is correct (obs_dim + N_STRATEGIC_MODES)
3. Action dict is valid (matches gymnasium space)
4. store_transition stores augmented obs
5. update() runs without error (pure RL path)
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.hybrid_agent import HybridAgent
from agents.llm.strategist import STRATEGIC_MODES, N_STRATEGIC_MODES, FALLBACK_MODE
from world.observation_space import NaturalLanguageObsBuilder, ObservationBuilder
from world.nation_state import NationState, RelationshipVector


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

RAW_OBS_DIM = 101  # 5-nation setup
N_NATIONS = 5


def make_agent(enable_llm: bool = False) -> HybridAgent:
    return HybridAgent(
        nation_id="alpha",
        obs_dim=RAW_OBS_DIM,
        n_nations=N_NATIONS,
        archetype="mercantile",
        enable_llm=enable_llm,
    )


def make_obs() -> np.ndarray:
    return np.random.default_rng(0).random(RAW_OBS_DIM).astype(np.float32)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestFallback:
    def test_act_without_llm_returns_valid_action(self):
        agent = make_agent(enable_llm=False)
        obs = make_obs()
        action, log_prob, value = agent.act(obs)

        assert "budget_allocation" in action
        assert "diplomatic_actions" in action
        assert action["budget_allocation"].shape == (5,)
        # diplomatic_actions: (n_targets,) or (n_targets, 1) depending on torch.stack
        assert action["diplomatic_actions"].shape[0] == N_NATIONS - 1
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_fallback_mode_is_economic_focus(self):
        agent = make_agent(enable_llm=False)
        assert agent._current_intent.mode == FALLBACK_MODE

    def test_last_reasoning_updated_after_act(self):
        agent = make_agent(enable_llm=False)
        obs = make_obs()
        # Should not raise even with no obs_text
        agent.act(obs, obs_text=None)
        # reasoning stays as initialized since no LLM call happened
        assert isinstance(agent.last_reasoning, str)


class TestEmbeddingShape:
    def test_augmented_obs_shape(self):
        agent = make_agent()
        obs = make_obs()
        augmented = agent._augment_obs(obs)
        assert augmented.shape == (RAW_OBS_DIM + N_STRATEGIC_MODES,)

    def test_actor_input_dim(self):
        agent = make_agent()
        assert agent.actor.shared[0].in_features == RAW_OBS_DIM + N_STRATEGIC_MODES

    def test_critic_input_dim(self):
        agent = make_agent()
        assert agent.critic.net[0].in_features == RAW_OBS_DIM + N_STRATEGIC_MODES

    def test_onehot_sum_is_one(self):
        agent = make_agent()
        onehot = agent._current_intent.to_onehot()
        assert len(onehot) == N_STRATEGIC_MODES
        assert abs(sum(onehot) - 1.0) < 1e-6


class TestStoreTransition:
    def test_buffer_stores_augmented_obs(self):
        agent = make_agent()
        obs = make_obs()
        action, log_prob, value = agent.act(obs)
        agent.store_transition(obs, action, log_prob, 1.0, value, False)

        assert len(agent.buffer) == 1
        stored_obs = agent.buffer._transitions[0].obs
        assert stored_obs.shape == (RAW_OBS_DIM + N_STRATEGIC_MODES,)


class TestUpdate:
    def test_update_runs_without_error(self):
        agent = make_agent()
        obs = make_obs()
        # Fill buffer with a few transitions
        for _ in range(10):
            action, log_prob, value = agent.act(obs)
            agent.store_transition(obs, action, log_prob, 0.5, value, False)

        result = agent.update()
        # Should return a loss dict (or empty dict if buffer too small)
        assert isinstance(result, dict)


class TestNaturalLanguageObsBuilder:
    def _make_nations(self) -> dict[str, NationState]:
        nations = {}
        for nid in ["alpha", "beta", "gamma"]:
            n = NationState(
                nation_id=nid,
                gdp=1.0,
                military_strength=0.4,
                population=100.0,
                resources={"oil": 0.5, "food": 0.6, "minerals": 0.4},
                tech_level=0.3,
                internal_stability=0.7,
                territory=0.5,
                relationships={},
                archetype="mercantile",
            )
            nations[nid] = n
        # Add relationships
        for nid in nations:
            for other_id in nations:
                if nid != other_id:
                    nations[nid].relationships[other_id] = RelationshipVector(
                        trade_volume=0.3, alliance_strength=0.1, hostility=0.1, grievance=0.05
                    )
        return nations

    def test_build_text_returns_string(self):
        builder = NaturalLanguageObsBuilder()
        nations = self._make_nations()
        text = builder.build_text("alpha", nations, recent_events=[], step=10)
        assert isinstance(text, str)
        assert "alpha" in text.lower() or "Alpha" in text
        assert "mercantile" in text

    def test_build_text_includes_relationships(self):
        builder = NaturalLanguageObsBuilder()
        nations = self._make_nations()
        text = builder.build_text("alpha", nations, step=5)
        assert "beta" in text.lower() or "Beta" in text

    def test_build_text_with_events(self):
        builder = NaturalLanguageObsBuilder()
        nations = self._make_nations()
        events = [{"type": "WAR", "attacker": "alpha", "defender": "beta"}]
        text = builder.build_text("alpha", nations, recent_events=events, step=20)
        assert "WAR" in text

    def test_build_text_unknown_nation(self):
        builder = NaturalLanguageObsBuilder()
        text = builder.build_text("unknown", {})
        assert "No data" in text
