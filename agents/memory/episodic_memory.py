from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class EventType(Enum):
    WAR = "WAR"
    BETRAYAL = "BETRAYAL"
    ALLIANCE_FORMED = "ALLIANCE_FORMED"
    TRADE_BOOM = "TRADE_BOOM"
    REGIME_CHANGE = "REGIME_CHANGE"
    SHOCK = "SHOCK"
    PEACE = "PEACE"


@dataclass
class EpisodicEvent:
    event_type: EventType
    timestep: int
    actor_id: str
    target_id: Optional[str]
    outcome: dict[str, float]
    salience: float

    def to_feature_vector(self, n_event_types: int = 7) -> np.ndarray:
        """Convert event to a fixed-size feature vector."""
        type_onehot = np.zeros(n_event_types, dtype=np.float32)
        try:
            idx = list(EventType).index(self.event_type)
            type_onehot[idx] = 1.0
        except ValueError:
            pass
        outcome_vec = np.array([
            self.outcome.get("gdp_delta", 0.0),
            self.outcome.get("military_delta", 0.0),
            self.outcome.get("stability_delta", 0.0),
            self.outcome.get("territory_delta", 0.0),
        ], dtype=np.float32)
        return np.concatenate([type_onehot, outcome_vec, [self.salience]])


class EventEmbedding(nn.Module):
    """Learned embedding for episodic events."""

    def __init__(self, event_feature_dim: int = 12, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(event_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EpisodicMemory:
    """
    Stores episodic events and retrieves top-k by cosine similarity to query observation.
    Returns a fixed-size context vector appended to the agent observation.
    """

    def __init__(
        self,
        max_events: int = 200,
        k: int = 10,
        embed_dim: int = 32,
        context_dim: int = 64,
    ):
        self.max_events = max_events
        self.k = k
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self._events: list[EpisodicEvent] = []
        self.embedding = EventEmbedding(embed_dim=embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(k * embed_dim, context_dim)

    def add_event(self, event: EpisodicEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events.pop(0)

    def to_context_vector(
        self, query_obs: np.ndarray, device: torch.device = torch.device("cpu")
    ) -> np.ndarray:
        """Retrieve top-k events by cosine similarity and project to context_dim."""
        if len(self._events) == 0:
            return np.zeros(self.context_dim, dtype=np.float32)

        event_features = np.stack([e.to_feature_vector() for e in self._events])
        event_tensors = torch.tensor(event_features, dtype=torch.float32, device=device)
        with torch.no_grad():
            event_embeds = self.embedding(event_tensors)  # (n_events, embed_dim)

        # Use a simple hash of the obs as query vector (or project it)
        query_vec = torch.tensor(
            query_obs[:self.embed_dim] if len(query_obs) >= self.embed_dim else
            np.pad(query_obs, (0, max(0, self.embed_dim - len(query_obs)))),
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            query_embed = self.query_proj(query_vec.unsqueeze(0))  # (1, embed_dim)

        # Cosine similarity
        sim = torch.nn.functional.cosine_similarity(
            query_embed, event_embeds, dim=1
        )  # (n_events,)

        k = min(self.k, len(self._events))
        top_k_indices = torch.topk(sim, k).indices
        top_k_embeds = event_embeds[top_k_indices]  # (k, embed_dim)

        # Pad to k if fewer events
        if k < self.k:
            pad = torch.zeros(self.k - k, self.embed_dim, device=device)
            top_k_embeds = torch.cat([top_k_embeds, pad], dim=0)

        flat = top_k_embeds.reshape(1, -1)
        with torch.no_grad():
            context = self.output_proj(flat).squeeze(0)
        return context.cpu().numpy()
