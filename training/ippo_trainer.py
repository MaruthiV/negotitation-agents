from __future__ import annotations

from typing import Optional

import numpy as np

from agents.ppo_agent import IPPOAgent, PPOConfig
from world.geopolitical_env import GeopoliticalEnv
from world.observation_space import ObservationBuilder
from training.runner import SimulationRunner
from analysis.logger import SimulationLogger
from analysis.metrics import EmergenceMetrics


class IPPOTrainer:
    """Orchestrates per-agent updates across a training run."""

    def __init__(
        self,
        nation_ids: list[str],
        env_kwargs: Optional[dict] = None,
        agent_kwargs: Optional[dict] = None,
        update_interval: int = 128,
        log_path: Optional[str] = None,
    ):
        self.nation_ids = nation_ids
        env_kwargs = env_kwargs or {}
        agent_kwargs = agent_kwargs or {}

        self.env = GeopoliticalEnv(nation_ids=nation_ids, **env_kwargs)
        obs_builder = ObservationBuilder(nation_ids)
        obs_dim = obs_builder.obs_dim

        config = PPOConfig(**agent_kwargs.get("ppo_config", {}))
        self.agents: dict[str, IPPOAgent] = {
            nid: IPPOAgent(
                nation_id=nid,
                obs_dim=obs_dim,
                n_nations=len(nation_ids),
                config=config,
            )
            for nid in nation_ids
        }

        self.logger = SimulationLogger(log_path) if log_path else None
        self.runner = SimulationRunner(
            self.env, self.agents, update_interval=update_interval, logger=self.logger
        )
        self.metrics = EmergenceMetrics()

    def train(self, n_episodes: int, eval_every: int = 50) -> list[dict]:
        history = []
        for ep in range(n_episodes):
            stats = self.runner.run_episode()
            history.append(stats)

            if ep % eval_every == 0:
                snapshot = self.env.get_world_snapshot()
                emergence = self.metrics.compute(snapshot)
                stats.update(emergence)
                print(
                    f"Episode {ep:4d} | mean_r={stats['mean_reward']:+.3f} "
                    f"| trade={emergence.get('mean_trade_volume', 0):.3f} "
                    f"| wars={emergence.get('wars_this_eval', 0)}"
                )
                if self.logger:
                    self.logger.log_metrics(ep, emergence)

        return history

    def save(self, path: str) -> None:
        import json, os
        os.makedirs(path, exist_ok=True)
        import torch
        for nid, agent in self.agents.items():
            torch.save(agent.get_weights(), f"{path}/{nid}.pt")
        print(f"Saved agents to {path}")

    def load(self, path: str) -> None:
        import torch
        for nid, agent in self.agents.items():
            weights = torch.load(f"{path}/{nid}.pt", map_location="cpu")
            agent.set_weights(weights)
