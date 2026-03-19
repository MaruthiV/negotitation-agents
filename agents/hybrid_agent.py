from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from agents.ppo_agent import IPPOAgent, PPOConfig
from agents.networks import HybridActorNetwork, CriticNetwork
from agents.llm.ollama_client import OllamaClient
from agents.llm.strategist import LLMStrategist, StrategicIntent, STRATEGIC_MODES, N_STRATEGIC_MODES, FALLBACK_MODE
from agents.llm.prompts import get_system_prompt
from world.observation_space import NaturalLanguageObsBuilder
from world.action_space import N_BUDGET_CHANNELS, N_DIPLOMATIC_OPTIONS


class HybridAgent(IPPOAgent):
    """
    Hybrid LLM + RL agent.

    Architecture:
      - OllamaClient queries a local LLM every `llm_interval` steps.
      - The LLM returns a StrategicIntent (one of 8 discrete modes + reasoning text).
      - The strategic mode is encoded as an 8-dim one-hot vector.
      - This embedding is concatenated to the raw obs vector before feeding the actor/critic.
      - PPO update logic is inherited from IPPOAgent (buffer stores augmented obs).

    When Ollama is offline, the agent falls back to pure RL (strategy = economic_focus).
    """

    def __init__(
        self,
        nation_id: str,
        obs_dim: int,
        n_nations: int,
        archetype: str = "mercantile",
        config: Optional[PPOConfig] = None,
        device: Optional[torch.device] = None,
        # LLM-specific
        ollama_model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        llm_interval: int = 5,
        enable_llm: bool = True,
    ):
        # Networks use augmented obs dimension
        augmented_obs_dim = obs_dim + N_STRATEGIC_MODES

        # Initialize IPPOAgent with augmented_obs_dim so actor/critic/buffer are sized correctly
        super().__init__(
            nation_id=nation_id,
            obs_dim=augmented_obs_dim,
            n_nations=n_nations,
            archetype=archetype,
            config=config,
            device=device,
        )

        # Replace actor with typed HybridActorNetwork (same dims, just named differently)
        cfg = self.config
        self.actor = HybridActorNetwork(
            obs_dim=augmented_obs_dim,
            n_budget_channels=N_BUDGET_CHANNELS,
            n_targets=self.n_targets,
            n_diplomatic_options=N_DIPLOMATIC_OPTIONS,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)

        # LLM components
        self._raw_obs_dim = obs_dim
        self._llm_interval = llm_interval
        self._enable_llm = enable_llm
        self._steps_since_llm: int = llm_interval  # force call on first step

        ollama_client = OllamaClient(model=ollama_model, base_url=ollama_base_url) if enable_llm else None
        self._strategist = LLMStrategist(ollama_client) if ollama_client is not None else None  # type: ignore[arg-type]
        self._system_prompt: str = get_system_prompt(archetype)

        self._current_intent: StrategicIntent = StrategicIntent(
            mode=FALLBACK_MODE,
            reasoning="Initializing strategy.",
            confidence=0.5,
        )
        self._last_augmented_obs: Optional[np.ndarray] = None
        self.last_reasoning: str = "No strategy yet."

        self._nl_obs_builder = NaturalLanguageObsBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, obs_text: Optional[str] = None, **kwargs) -> tuple[dict, float, float]:  # type: ignore[override]
        """
        Returns (action_dict, log_prob, value).

        Calls LLM every `llm_interval` steps if obs_text is supplied.
        Uses cached StrategicIntent otherwise.
        """
        self._maybe_refresh_intent(obs_text)

        augmented = self._augment_obs(obs)
        self._last_augmented_obs = augmented

        obs_t = torch.tensor(augmented, dtype=torch.float32, device=self.device).unsqueeze(0)
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
        """Store the augmented obs (not raw obs) so that replay matches network input."""
        augmented = (
            self._last_augmented_obs
            if self._last_augmented_obs is not None
            else self._augment_obs(obs)
        )
        super().store_transition(augmented, action, log_prob, reward, value, done)

    def update(self, last_obs: Optional[np.ndarray] = None) -> dict[str, float]:
        """Use the cached augmented obs for the bootstrap value estimate."""
        augmented_last = self._last_augmented_obs
        return super().update(last_obs=augmented_last)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Concatenate strategic one-hot embedding to raw obs vector."""
        onehot = np.array(self._current_intent.to_onehot(), dtype=np.float32)
        return np.concatenate([obs, onehot], axis=0)

    def _maybe_refresh_intent(self, obs_text: Optional[str]) -> None:
        """Query LLM if interval elapsed and obs_text is available."""
        self._steps_since_llm += 1
        if (
            self._strategist is None
            or obs_text is None
            or self._steps_since_llm < self._llm_interval
        ):
            return

        self._steps_since_llm = 0
        intent = self._strategist.decide(obs_text, self._system_prompt)
        self._current_intent = intent
        self.last_reasoning = (
            f"[{intent.mode}] (conf={intent.confidence:.2f}) {intent.reasoning}"
        )
