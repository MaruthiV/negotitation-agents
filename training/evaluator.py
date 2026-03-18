from __future__ import annotations

from typing import Optional

import numpy as np

from world.geopolitical_env import GeopoliticalEnv
from agents.ppo_agent import IPPOAgent
from analysis.metrics import EmergenceMetrics


class Evaluator:
    """Run deterministic evaluation episodes and collect metrics."""

    def __init__(
        self,
        env: GeopoliticalEnv,
        agents: dict[str, IPPOAgent],
        n_eval_episodes: int = 20,
    ):
        self.env = env
        self.agents = agents
        self.n_eval_episodes = n_eval_episodes
        self.metrics = EmergenceMetrics()

    def evaluate(self) -> dict[str, float]:
        """Run n_eval_episodes and return aggregated metrics."""
        episode_stats = []

        for _ in range(self.n_eval_episodes):
            self.env.reset()
            ep_rewards = {a: 0.0 for a in self.env.possible_agents}
            war_count = 0

            while self.env.agents:
                agent_id = self.env.agent_selection
                if self.env.terminations[agent_id] or self.env.truncations[agent_id]:
                    self.env.step(None)
                    continue

                obs = self.env.observe(agent_id)
                agent = self.agents.get(agent_id)
                if agent:
                    action, _, _ = agent.act(obs)
                else:
                    action = self.env.action_space(agent_id).sample()

                self.env.step(action)
                ep_rewards[agent_id] += self.env.rewards.get(agent_id, 0.0)

                for evt in self.env._events:
                    if evt.get("type") == "WAR_RESOLVED":
                        war_count += 1

            snapshot = self.env.get_world_snapshot()
            emergence = self.metrics.compute(snapshot)

            episode_stats.append({
                "mean_reward": float(np.mean(list(ep_rewards.values()))),
                "wars": war_count,
                **emergence,
            })

        # Aggregate
        keys = episode_stats[0].keys()
        return {k: float(np.mean([s[k] for s in episode_stats])) for k in keys}
