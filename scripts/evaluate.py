#!/usr/bin/env python3
"""
Evaluation script — loads a checkpoint and reports emergence metrics.
Run: python scripts/evaluate.py [checkpoint_path]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from world.geopolitical_env import GeopoliticalEnv
from world.observation_space import ObservationBuilder
from agents.ppo_agent import IPPOAgent
from analysis.metrics import EmergenceMetrics


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/phase1"
    nation_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]
    n_eval = 50

    env = GeopoliticalEnv(nation_ids=nation_ids, max_steps=200, seed=1234)
    obs_dim = ObservationBuilder(nation_ids).obs_dim
    agents = {}

    for nid in nation_ids:
        agent = IPPOAgent(nid, obs_dim, len(nation_ids))
        path = f"{checkpoint_path}/{nid}.pt"
        if os.path.exists(path):
            weights = torch.load(path, map_location="cpu")
            agent.set_weights(weights)
            print(f"Loaded checkpoint for {nid}")
        else:
            print(f"No checkpoint for {nid}, using random policy")
        agents[nid] = agent

    metrics = EmergenceMetrics()
    all_results = []
    total_wars = 0

    print(f"\nRunning {n_eval} evaluation episodes...")

    for ep in range(n_eval):
        env.reset()
        ep_rewards = {a: 0.0 for a in nation_ids}

        while env.agents:
            aid = env.agent_selection
            if env.terminations[aid] or env.truncations[aid]:
                env.step(None)
                continue
            obs = env.observe(aid)
            action, _, _ = agents[aid].act(obs)
            env.step(action)
            ep_rewards[aid] += env.rewards.get(aid, 0.0)

        snapshot = env.get_world_snapshot()
        result = metrics.compute(snapshot)
        result["mean_episode_reward"] = np.mean(list(ep_rewards.values()))
        all_results.append(result)
        total_wars += result.get("wars_this_eval", 0)

    # Aggregate
    keys = all_results[0].keys()
    agg = {k: np.mean([r[k] for r in all_results]) for k in keys}

    print("\n=== Evaluation Results ===")
    print(f"Episodes:              {n_eval}")
    print(f"Mean reward:           {agg['mean_episode_reward']:+.4f}")
    print(f"Mean trade volume:     {agg['mean_trade_volume']:.4f}")
    print(f"GDP Gini:              {agg['gdp_gini']:.4f}")
    print(f"Liberal peace index:   {agg['liberal_peace_index']:.4f}")
    print(f"Arms race score:       {agg['arms_race_score']:.4f}")
    print(f"Max GDP share:         {agg['max_gdp_share']:.4f}")
    print(f"Total wars:            {int(total_wars)}")
    print(f"Mean stability:        {agg['mean_stability']:.4f}")


if __name__ == "__main__":
    main()
