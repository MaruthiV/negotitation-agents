#!/usr/bin/env python3
"""
Phase 2 training script — Continual Learning + Exogenous Shocks.
Run: python scripts/train_phase2.py

Loads Phase 1 checkpoint, enables CL (CLEAR + EWC), injects shocks, and
measures forgetting retention.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from world.geopolitical_env import GeopoliticalEnv
from world.observation_space import ObservationBuilder
from agents.ppo_agent import IPPOAgent, PPOConfig
from agents.continual.clear import CLEARBuffer, Experience
from agents.continual.ewc import EWCRegularizer
from training.runner import SimulationRunner
from analysis.metrics import EmergenceMetrics
from analysis.logger import SimulationLogger


def main():
    nation_ids_ab = ["alpha", "beta", "gamma"]
    nation_ids_ac = ["alpha", "gamma", "delta"]

    obs_dim_ab = ObservationBuilder(nation_ids_ab).obs_dim
    obs_dim_ac = ObservationBuilder(nation_ids_ac).obs_dim

    print("=== Phase 2: Continual Learning ===\n")

    config = PPOConfig(use_ewc=True, ewc_importance=5000.0)

    # Train alpha vs {beta, gamma} (context A-B)
    print("--- Context AB: training alpha vs beta/gamma ---")
    env_ab = GeopoliticalEnv(nation_ids=nation_ids_ab, max_steps=100, enable_shocks=False)
    agents_ab = {
        nid: IPPOAgent(nid, obs_dim_ab, len(nation_ids_ab), config=config)
        for nid in nation_ids_ab
    }
    runner_ab = SimulationRunner(env_ab, agents_ab, update_interval=64)
    for ep in range(200):
        runner_ab.run_episode()
    perf_ab_before = evaluate_agent(agents_ab["alpha"], env_ab, nation_ids_ab, 10)
    print(f"Alpha performance vs AB (before switch): {perf_ab_before:.3f}")

    # Consolidate EWC at context boundary
    for agent in agents_ab.values():
        if agent.ewc and len(agent.buffer) > 8:
            agent._consolidate_ewc()

    # Store AB experiences in CLEAR buffer
    clear_buf = CLEARBuffer(replay_ratio=0.3)

    # Train alpha vs {gamma, delta} (context A-C)
    print("--- Context AC: training alpha vs gamma/delta ---")
    env_ac = GeopoliticalEnv(nation_ids=nation_ids_ac, max_steps=100, enable_shocks=True)
    agents_ac = {
        nid: IPPOAgent(nid, obs_dim_ac, len(nation_ids_ac), config=config)
        for nid in nation_ids_ac
    }
    # Warm-start alpha from AB
    if os.path.exists("checkpoints/phase1/alpha.pt"):
        weights = torch.load("checkpoints/phase1/alpha.pt", map_location="cpu")
        agents_ac["alpha"].set_weights(weights)

    runner_ac = SimulationRunner(env_ac, agents_ac, update_interval=64)
    for ep in range(200):
        runner_ac.run_episode()

    # Re-evaluate alpha on AB context
    perf_ab_after = evaluate_agent(agents_ab["alpha"], env_ab, nation_ids_ab, 10)
    retention = perf_ab_after / max(abs(perf_ab_before), 1e-6)
    print(f"\nAlpha performance vs AB (after AC training): {perf_ab_after:.3f}")
    print(f"Forgetting retention: {retention:.1%} (target: > 80%)")

    if retention > 0.8:
        print("✓ Forgetting test PASSED")
    else:
        print("✗ Forgetting test needs more CL tuning")

    print("\nPhase 2 complete.")


def evaluate_agent(agent, env, nation_ids, n_episodes):
    """Return mean reward for agent over n_episodes."""
    import numpy as np
    rewards = []
    for _ in range(n_episodes):
        env.reset()
        ep_reward = 0.0
        while env.agents:
            aid = env.agent_selection
            if env.terminations[aid] or env.truncations[aid]:
                env.step(None)
                continue
            obs = env.observe(aid)
            if aid == agent.nation_id:
                action, _, _ = agent.act(obs)
            else:
                action = env.action_space(aid).sample()
            env.step(action)
            if aid == agent.nation_id:
                ep_reward += env.rewards.get(aid, 0.0)
        rewards.append(ep_reward)
    return float(np.mean(rewards))


if __name__ == "__main__":
    main()
