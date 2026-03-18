from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Experience:
    obs: np.ndarray
    budget_action: np.ndarray
    diplomatic_action: np.ndarray
    log_prob: float
    reward: float
    value: float
    done: bool
    context_id: str
    priority: float = 1.0


class CLEARBuffer:
    """
    CLEAR (Continual LEArning with Replay) buffer.
    Partitions experiences by context_id (opponent set).
    During training, 30% of each batch is sampled from *other* contexts to prevent forgetting.
    """

    def __init__(
        self,
        max_per_context: int = 5000,
        replay_ratio: float = 0.3,
    ):
        self.max_per_context = max_per_context
        self.replay_ratio = replay_ratio
        self.buffers: dict[str, deque] = {}

    def add(self, experience: Experience) -> None:
        ctx = experience.context_id
        if ctx not in self.buffers:
            self.buffers[ctx] = deque(maxlen=self.max_per_context)
        self.buffers[ctx].append(experience)

    def add_rollout(
        self, transitions: list[Experience], context_id: str
    ) -> None:
        for t in transitions:
            t.context_id = context_id
            self.add(t)

    def build_mixed_batch(
        self,
        current_rollout: list[Experience],
        current_context: str,
        batch_size: Optional[int] = None,
    ) -> list[Experience]:
        """
        Returns current_rollout + 30% sampled from all other contexts (priority-weighted).
        """
        other_contexts = [k for k in self.buffers.keys() if k != current_context]

        if not other_contexts:
            return list(current_rollout)

        n_replay = int(len(current_rollout) * self.replay_ratio)
        if n_replay == 0:
            return list(current_rollout)

        # Collect all experiences from other contexts
        other_experiences = []
        for ctx in other_contexts:
            other_experiences.extend(self.buffers[ctx])

        if not other_experiences:
            return list(current_rollout)

        # Priority-weighted sampling
        priorities = np.array([e.priority for e in other_experiences], dtype=np.float32)
        priorities = priorities / priorities.sum()
        n_sample = min(n_replay, len(other_experiences))
        indices = np.random.choice(len(other_experiences), size=n_sample, replace=False, p=priorities)
        replay_batch = [other_experiences[i] for i in indices]

        return list(current_rollout) + replay_batch

    def context_ids(self) -> list[str]:
        return list(self.buffers.keys())

    def size(self, context_id: Optional[str] = None) -> int:
        if context_id is not None:
            return len(self.buffers.get(context_id, []))
        return sum(len(b) for b in self.buffers.values())
