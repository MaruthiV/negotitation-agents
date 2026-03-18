from __future__ import annotations

from typing import Optional

import numpy as np

from world.geopolitical_env import GeopoliticalEnv
from agents.ppo_agent import IPPOAgent, PPOConfig
from world.observation_space import ObservationBuilder
from analysis.logger import SimulationLogger


class SimulationRunner:
    """
    Main episode loop. Manages env + agents, collects rollouts, triggers updates.
    """

    def __init__(
        self,
        env: GeopoliticalEnv,
        agents: dict[str, IPPOAgent],
        update_interval: int = 128,
        logger: Optional[SimulationLogger] = None,
    ):
        self.env = env
        self.agents = agents
        self.update_interval = update_interval
        self.logger = logger

        self._episode = 0
        self._global_step = 0
        self._last_actions: dict[str, dict] = {}
        self._last_logprobs: dict[str, float] = {}
        self._last_values: dict[str, float] = {}
        self._last_obs: dict[str, np.ndarray] = {}

    def run_episode(self) -> dict[str, float]:
        """Run one full episode, return episode stats."""
        self.env.reset()
        episode_rewards = {a: 0.0 for a in self.env.possible_agents}
        step_count = 0

        while self.env.agents:
            agent_id = self.env.agent_selection
            if self.env.terminations[agent_id] or self.env.truncations[agent_id]:
                self.env.step(None)
                continue

            obs = self.env.observe(agent_id)
            self._last_obs[agent_id] = obs

            agent = self.agents.get(agent_id)
            if agent is None:
                # Random action fallback
                action = self.env.action_space(agent_id).sample()
                log_prob, value = 0.0, 0.0
            else:
                action, log_prob, value = agent.act(obs)

            self._last_actions[agent_id] = action
            self._last_logprobs[agent_id] = log_prob
            self._last_values[agent_id] = value

            self.env.step(action)

            # After step, collect reward for this agent
            reward = self.env.rewards.get(agent_id, 0.0)
            done = self.env.terminations.get(agent_id, False) or self.env.truncations.get(agent_id, False)

            if agent is not None:
                agent.store_transition(obs, action, log_prob, reward, value, done)

            episode_rewards[agent_id] += reward
            self._global_step += 1
            step_count += 1

            # Periodic update
            if agent is not None and len(agent.buffer) >= self.update_interval:
                loss_info = agent.update(last_obs=obs)
                if self.logger and loss_info:
                    self.logger.log_training(agent_id, self._global_step, loss_info)

        # Final update for all agents
        for agent_id, agent in self.agents.items():
            if len(agent.buffer) > 0:
                last_obs = self._last_obs.get(agent_id)
                loss_info = agent.update(last_obs=last_obs)
                if self.logger and loss_info:
                    self.logger.log_training(agent_id, self._global_step, loss_info)

        self._episode += 1

        stats = {
            "episode": self._episode,
            "steps": step_count,
            "global_step": self._global_step,
            **{f"reward_{a}": r for a, r in episode_rewards.items()},
            "mean_reward": float(np.mean(list(episode_rewards.values()))),
        }

        if self.logger:
            snapshot = self.env.get_world_snapshot()
            self.logger.log_episode(self._episode, stats, snapshot)

        return stats

    def run_n_episodes(self, n: int) -> list[dict[str, float]]:
        return [self.run_episode() for _ in range(n)]

    def step_and_snapshot(self) -> Optional[dict]:
        """
        Advance one agent turn and return a world snapshot if a full world step occurred.
        Used by the API server for real-time streaming.
        """
        if not self.env.agents:
            self.env.reset()
            self._last_obs.clear()

        agent_id = self.env.agent_selection
        if self.env.terminations.get(agent_id, False) or self.env.truncations.get(agent_id, False):
            self.env.step(None)
            return None

        obs = self.env.observe(agent_id)
        agent = self.agents.get(agent_id)
        if agent:
            action, log_prob, value = agent.act(obs)
        else:
            action = self.env.action_space(agent_id).sample()
            log_prob, value = 0.0, 0.0

        prev_step = self.env._step_count
        self.env.step(action)
        new_step = self.env._step_count

        if new_step > prev_step:
            return self.env.get_world_snapshot()
        return None
