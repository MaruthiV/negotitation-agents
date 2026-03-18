from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from world.nation_state import NationState, ARCHETYPES

if TYPE_CHECKING:
    from agents.ppo_agent import IPPOAgent


class RegimeChangeHandler:
    """
    Phase 2 warm-start regime change handler.
    1. Consolidate EWC on old agent's replay buffer
    2. Deepcopy agent + Gaussian noise (std=0.05) on actor params
    3. Assign random new archetype
    4. Preserve 60% of relationship values, decay grievances 60%
    5. Reset stability to 0.4
    """

    ACTOR_NOISE_STD = 0.05
    RELATIONSHIP_PRESERVE = 0.60
    GRIEVANCE_DECAY = 0.60
    STABILITY_RESET = 0.40

    def handle(
        self,
        old_agent: "IPPOAgent",
        nation: NationState,
        rng: np.random.Generator,
    ) -> "IPPOAgent":
        """Returns a new agent (warm-started from old), mutates nation in-place."""
        # 1. EWC consolidation
        if old_agent.ewc is not None and len(old_agent.buffer) > 16:
            old_agent._consolidate_ewc()

        # 2. Deepcopy + noise
        new_agent = copy.deepcopy(old_agent)
        with torch.no_grad():
            for param in new_agent.actor.parameters():
                param.add_(torch.randn_like(param) * self.ACTOR_NOISE_STD)

        # 3. New archetype → new reward weights
        new_archetype = rng.choice([a for a in ARCHETYPES if a != nation.archetype])
        nation.archetype = str(new_archetype)
        new_agent.archetype = str(new_archetype)

        # 4. Institutional memory: preserve relationships, decay grievances
        for rel in nation.relationships.values():
            rel.trade_volume *= self.RELATIONSHIP_PRESERVE
            rel.alliance_strength *= self.RELATIONSHIP_PRESERVE
            rel.hostility *= self.RELATIONSHIP_PRESERVE
            rel.grievance *= (1.0 - self.GRIEVANCE_DECAY)

        # 5. Reset stability
        nation.internal_stability = self.STABILITY_RESET
        nation.age = 0

        return new_agent
