from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agents.networks import ActorNetwork, CriticNetwork
from agents.memory.replay_buffer import RolloutBuffer, Transition
from agents.continual.ewc import EWCRegularizer
from world.action_space import N_BUDGET_CHANNELS, N_DIPLOMATIC_OPTIONS


@dataclass
class PPOConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    minibatch_size: int = 64
    hidden_dim: int = 256
    use_ewc: bool = False
    ewc_importance: float = 5000.0


class IPPOAgent:
    """
    Independent PPO agent for a single nation.
    CleanRL-style: actor + critic trained independently with shared obs.
    """

    def __init__(
        self,
        nation_id: str,
        obs_dim: int,
        n_nations: int,
        archetype: str = "mercantile",
        config: Optional[PPOConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.nation_id = nation_id
        self.obs_dim = obs_dim
        self.n_targets = n_nations - 1
        self.archetype = archetype
        self.config = config or PPOConfig()
        self.device = device or torch.device("cpu")

        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            n_budget_channels=N_BUDGET_CHANNELS,
            n_targets=self.n_targets,
            n_diplomatic_options=N_DIPLOMATIC_OPTIONS,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        self.critic = CriticNetwork(obs_dim=obs_dim, hidden_dim=self.config.hidden_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

        self.buffer = RolloutBuffer(obs_dim, N_BUDGET_CHANNELS, self.n_targets)

        self.ewc: Optional[EWCRegularizer] = (
            EWCRegularizer(self.config.ewc_importance) if self.config.use_ewc else None
        )

        self._total_steps = 0
        self._total_updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[dict, float, float]:
        """
        Returns (action_dict, log_prob, value).
        action_dict matches gymnasium Dict space keys.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        budget_sample, diplomatic_samples, log_prob, _ = self.actor.get_action_and_logprob(obs_t)
        value = self.critic(obs_t).item()

        budget_np = budget_sample.squeeze(0).cpu().numpy()
        diplomatic_np = torch.stack(diplomatic_samples).cpu().numpy()

        action = {
            "budget_allocation": budget_np,
            "diplomatic_actions": diplomatic_np,
        }
        return action, float(log_prob.item()), float(value)

    def store_transition(
        self,
        obs: np.ndarray,
        action: dict,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.buffer.add(Transition(
            obs=obs,
            budget_action=action["budget_allocation"],
            diplomatic_action=action["diplomatic_actions"],
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
        ))
        self._total_steps += 1

    def update(self, last_obs: Optional[np.ndarray] = None) -> dict[str, float]:
        """Run PPO update on the current buffer. Returns loss info dict."""
        if len(self.buffer) < 2:
            return {}

        last_value = 0.0
        if last_obs is not None and not self.buffer._transitions[-1].done:
            with torch.no_grad():
                obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_value = self.critic(obs_t).item()

        batch = self.buffer.to_tensors(
            self.device,
            self.config.gamma,
            self.config.gae_lambda,
            last_value,
        )

        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(self.buffer)
        indices = np.arange(n)

        actor_losses, critic_losses, entropy_losses = [], [], []

        for _ in range(self.config.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.config.minibatch_size):
                mb_idx = indices[start: start + self.config.minibatch_size]

                mb_obs = batch["obs"][mb_idx]
                mb_budget = batch["budget_actions"][mb_idx]
                mb_diplomatic = batch["diplomatic_actions"][mb_idx]
                mb_old_logprob = batch["log_probs"][mb_idx]
                mb_returns = batch["returns"][mb_idx]
                mb_advantages = advantages[mb_idx]

                new_logprob, entropy = self.actor.evaluate_actions(mb_obs, mb_budget, mb_diplomatic)
                ratio = torch.exp(new_logprob - mb_old_logprob)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                if self.ewc is not None:
                    actor_loss = actor_loss + self.ewc.penalty(self.actor)

                entropy_loss = -entropy.mean() * self.config.entropy_coef

                self.actor_opt.zero_grad()
                (actor_loss + entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_opt.step()

                values_pred = self.critic(mb_obs)
                critic_loss = self.config.vf_coef * ((values_pred - mb_returns) ** 2).mean()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_opt.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy.mean().item())

        self.buffer.clear()
        self._total_updates += 1

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_losses)),
        }

    def _consolidate_ewc(self) -> None:
        """Build dataloader from buffer and consolidate EWC Fisher."""
        if self.ewc is None or len(self.buffer) < 8:
            return
        batch = self.buffer.to_tensors(self.device)
        ds = TensorDataset(
            batch["obs"], batch["budget_actions"], batch["diplomatic_actions"]
        )
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        self.ewc.consolidate(self.actor, loader)

    def get_weights(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def set_weights(self, weights: dict) -> None:
        self.actor.load_state_dict(weights["actor"])
        self.critic.load_state_dict(weights["critic"])
