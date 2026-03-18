#!/usr/bin/env python3
"""
Phase 1 training script.
Run: python scripts/train_phase1.py

Trains IPPO agents on 5 nations for 1000 episodes and checks emergence metrics.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from training.ippo_trainer import IPPOTrainer


def main():
    nation_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]

    trainer = IPPOTrainer(
        nation_ids=nation_ids,
        env_kwargs={"max_steps": 200, "noise_std": 0.05, "enable_shocks": False},
        agent_kwargs={"ppo_config": {"lr_actor": 3e-4, "n_epochs": 4}},
        update_interval=64,
        log_path="logs/phase1",
    )

    print("=== Phase 1 Training ===")
    print(f"Nations: {nation_ids}")
    print(f"Episodes: 1000")
    print()

    history = trainer.train(n_episodes=1000, eval_every=50)

    # Final check
    final_stats = history[-1]
    print("\n=== Final Stats ===")
    print(f"Mean reward: {final_stats.get('mean_reward', 0):+.3f}")

    trainer.save("checkpoints/phase1")
    print("\nCheckpoints saved to checkpoints/phase1/")

    # Verify emergence criteria
    from world.geopolitical_env import GeopoliticalEnv
    from world.observation_space import ObservationBuilder
    from analysis.metrics import EmergenceMetrics
    from training.runner import SimulationRunner

    env = GeopoliticalEnv(nation_ids=nation_ids, max_steps=200, seed=99)
    obs_dim = ObservationBuilder(nation_ids).obs_dim
    metrics = EmergenceMetrics()

    total_wars = 0
    trade_vols = []
    gdp_shares = []

    print("\n=== Emergence Check (20 eval episodes) ===")
    for ep in range(20):
        env.reset()
        while env.agents:
            agent_id = env.agent_selection
            if env.terminations[agent_id] or env.truncations[agent_id]:
                env.step(None)
                continue
            obs = env.observe(agent_id)
            agent = trainer.agents.get(agent_id)
            if agent:
                action, _, _ = agent.act(obs)
            else:
                action = env.action_space(agent_id).sample()
            env.step(action)

        snapshot = env.get_world_snapshot()
        result = metrics.compute(snapshot)
        trade_vols.append(result.get("mean_trade_volume", 0))
        gdp_shares.append(result.get("max_gdp_share", 1))
        total_wars += result.get("wars_this_eval", 0)

    import numpy as np
    print(f"Mean trade volume: {np.mean(trade_vols):.3f} (target: > 0.2)")
    print(f"Total wars (20 eps): {total_wars} (target: > 0)")
    print(f"Max GDP share: {np.mean(gdp_shares):.3f} (target: < 0.5)")


if __name__ == "__main__":
    main()
