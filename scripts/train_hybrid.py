#!/usr/bin/env python3
"""
Hybrid LLM + RL training script.

Usage:
  # With Ollama running (ollama serve + ollama pull llama3.2):
  python scripts/train_hybrid.py

  # Pure RL fallback (no Ollama required):
  python scripts/train_hybrid.py --no-llm

  # Use a different model:
  python scripts/train_hybrid.py --model mistral

  # Quick smoke test (50 episodes):
  python scripts/train_hybrid.py --no-llm --episodes 50
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid LLM + RL agents")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (pure RL fallback)")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--llm-interval", type=int, default=5, help="LLM call interval (steps)")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--log-path", default="logs/hybrid", help="Log directory")
    parser.add_argument("--checkpoint-path", default="checkpoints/hybrid", help="Checkpoint directory")
    args = parser.parse_args()

    nation_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]
    enable_llm = not args.no_llm

    if enable_llm:
        from agents.llm.ollama_client import OllamaClient
        client = OllamaClient(model=args.model)
        if client.is_available():
            print(f"Ollama is available — using model '{args.model}'")
        else:
            print(
                f"Warning: Ollama not available at localhost:11434. "
                f"Agents will use fallback strategy (economic_focus). "
                f"Start Ollama with: ollama serve && ollama pull {args.model}"
            )
        client.close()
    else:
        print("LLM disabled (--no-llm). Running as pure RL.")

    from training.ippo_trainer import IPPOTrainer

    trainer = IPPOTrainer(
        nation_ids=nation_ids,
        env_kwargs={"max_steps": 200, "noise_std": 0.05, "enable_shocks": False},
        agent_kwargs={"ppo_config": {"lr_actor": 3e-4, "n_epochs": 4}},
        update_interval=64,
        log_path=args.log_path,
        agent_type="hybrid",
        llm_kwargs={
            "ollama_model": args.model,
            "llm_interval": args.llm_interval,
            "enable_llm": enable_llm,
        },
    )

    print(f"\n=== Hybrid LLM + RL Training ===")
    print(f"Nations:    {nation_ids}")
    print(f"Episodes:   {args.episodes}")
    print(f"LLM:        {'enabled (' + args.model + ')' if enable_llm else 'disabled'}")
    print(f"LLM interval: every {args.llm_interval} steps")
    print()

    history = trainer.train(n_episodes=args.episodes, eval_every=args.eval_every)

    # Show last LLM reasoning for each agent
    print("\n=== Last LLM Reasoning per Nation ===")
    for nid, agent in trainer.agents.items():
        if hasattr(agent, "last_reasoning"):
            print(f"  {nid}: {agent.last_reasoning}")

    final_stats = history[-1]
    print(f"\nFinal mean reward: {final_stats.get('mean_reward', 0):+.3f}")

    trainer.save(args.checkpoint_path)
    print(f"\nCheckpoints saved to {args.checkpoint_path}/")

    # Emergence verification (same criteria as phase1)
    import numpy as np
    from world.geopolitical_env import GeopoliticalEnv
    from analysis.metrics import EmergenceMetrics

    env = GeopoliticalEnv(nation_ids=nation_ids, max_steps=200, seed=42)
    metrics = EmergenceMetrics()
    total_wars = 0
    trade_vols = []
    gdp_shares = []

    print("\n=== Emergence Check (20 eval episodes) ===")
    for _ in range(20):
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

    print(f"Mean trade volume: {np.mean(trade_vols):.3f} (target: > 0.2)")
    print(f"Total wars (20 eps): {total_wars} (target: > 0)")
    print(f"Max GDP share: {np.mean(gdp_shares):.3f} (target: < 0.5)")


if __name__ == "__main__":
    main()
