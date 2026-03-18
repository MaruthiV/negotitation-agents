from __future__ import annotations

"""
Phase 3 distributed training via Ray.
DO NOT use until single-process training loop is working and profiled.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RolloutResult:
    agent_id: str
    transitions: list[dict]
    episode_stats: dict[str, float]


try:
    import ray

    @ray.remote
    class SimulationWorker:
        """Remote worker that runs rollouts with given policy weights."""

        def __init__(self, nation_ids: list[str], env_kwargs: dict, seed: Optional[int] = None):
            from world.geopolitical_env import GeopoliticalEnv
            from agents.ppo_agent import IPPOAgent, PPOConfig
            from world.observation_space import ObservationBuilder

            self.env = GeopoliticalEnv(nation_ids=nation_ids, seed=seed, **env_kwargs)
            obs_dim = ObservationBuilder(nation_ids).obs_dim
            self.agents = {
                nid: IPPOAgent(nid, obs_dim, len(nation_ids))
                for nid in nation_ids
            }

        def run_rollout(self, policy_weights: dict) -> dict:
            for nid, weights in policy_weights.items():
                if nid in self.agents:
                    self.agents[nid].set_weights(weights)

            from training.runner import SimulationRunner
            runner = SimulationRunner(self.env, self.agents)
            stats = runner.run_episode()
            return {
                "stats": stats,
                "buffers": {
                    nid: {
                        "transitions": [
                            {
                                "obs": t.obs.tolist(),
                                "budget_action": t.budget_action.tolist(),
                                "diplomatic_action": t.diplomatic_action.tolist(),
                                "log_prob": t.log_prob,
                                "reward": t.reward,
                                "value": t.value,
                                "done": t.done,
                            }
                            for t in agent.buffer._transitions
                        ]
                    }
                    for nid, agent in self.agents.items()
                },
            }

    class DistributedIPPOTrainer:
        """Central trainer that dispatches rollouts to N Ray workers."""

        def __init__(
            self,
            nation_ids: list[str],
            n_workers: int = 4,
            env_kwargs: Optional[dict] = None,
        ):
            ray.init(ignore_reinit_error=True)
            self.nation_ids = nation_ids
            self.n_workers = n_workers

            from world.observation_space import ObservationBuilder
            from agents.ppo_agent import IPPOAgent
            obs_dim = ObservationBuilder(nation_ids).obs_dim

            self.central_agents = {
                nid: IPPOAgent(nid, obs_dim, len(nation_ids))
                for nid in nation_ids
            }

            env_kwargs = env_kwargs or {}
            self.workers = [
                SimulationWorker.remote(nation_ids, env_kwargs, seed=i)
                for i in range(n_workers)
            ]

        def train(self, n_iterations: int) -> list[dict]:
            history = []
            for iteration in range(n_iterations):
                weights = {
                    nid: agent.get_weights()
                    for nid, agent in self.central_agents.items()
                }
                futures = [w.run_rollout.remote(weights) for w in self.workers]
                results = ray.get(futures)

                all_stats = [r["stats"] for r in results]
                mean_reward = sum(s["mean_reward"] for s in all_stats) / len(all_stats)

                # Update central agents with aggregated experience
                # (simplified: use first worker's buffer for now)
                for nid, agent in self.central_agents.items():
                    agent.update()

                step_info = {"iteration": iteration, "mean_reward": mean_reward}
                history.append(step_info)
                print(f"Iter {iteration:4d} | mean_r={mean_reward:+.3f}")

            return history

except ImportError:
    pass  # Ray not installed; Phase 3 feature
