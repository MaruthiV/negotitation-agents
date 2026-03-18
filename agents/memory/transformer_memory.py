from __future__ import annotations

import torch
import torch.nn as nn


class TransformerMemoryAugmentation(nn.Module):
    """
    2-layer transformer encoder with cross-attention from current obs to event sequence.
    Richer alternative to cosine retrieval in EpisodicMemory.
    """

    def __init__(
        self,
        obs_dim: int,
        event_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        context_dim: int = 64,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.event_dim = event_dim
        self.context_dim = context_dim
        self.max_seq_len = max_seq_len

        # Project obs and events to same hidden dim
        hidden_dim = max(obs_dim, event_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.event_proj = nn.Linear(event_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cross-attention: query from obs, key/value from event sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=0.0
        )
        self.output_proj = nn.Linear(hidden_dim, context_dim)

    def forward(
        self,
        obs: torch.Tensor,       # (batch, obs_dim)
        event_seq: torch.Tensor, # (batch, seq_len, event_dim)
    ) -> torch.Tensor:           # (batch, context_dim)
        obs_h = self.obs_proj(obs).unsqueeze(1)           # (batch, 1, hidden)
        event_h = self.event_proj(event_seq)              # (batch, seq_len, hidden)

        # Encode events
        event_encoded = self.encoder(event_h)             # (batch, seq_len, hidden)

        # Cross-attention: obs attends over events
        attn_out, _ = self.cross_attn(obs_h, event_encoded, event_encoded)  # (batch, 1, hidden)
        context = self.output_proj(attn_out.squeeze(1))   # (batch, context_dim)
        return context
