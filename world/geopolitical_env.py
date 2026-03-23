from __future__ import annotations

import copy
import random
from typing import Any, Optional

import numpy as np
import gymnasium
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector as AgentSelector

from world.nation_state import NationState, RelationshipVector, ARCHETYPES
from world.action_space import make_action_space, ActionEncoder, DIPLOMATIC_ACTIONS
from world.observation_space import ObservationBuilder
from world.reward import RewardCalculator, ARCHETYPE_WEIGHTS
from world.dynamics.diplomacy import DiplomacyResolver
from world.dynamics.military import MilitaryResolver
from world.dynamics.trade import TradeResolver
from world.dynamics.internal import InternalDynamicsResolver
from world.dynamics.shocks import ExogenousShockGenerator


def _default_nations(nation_ids: list[str], rng: np.random.Generator) -> dict[str, NationState]:
    nations = {}
    archetypes = ARCHETYPES * (len(nation_ids) // len(ARCHETYPES) + 1)
    for i, nid in enumerate(nation_ids):
        nations[nid] = NationState(
            nation_id=nid,
            gdp=float(rng.uniform(0.8, 1.5)),
            military_strength=float(rng.uniform(0.2, 0.6)),
            population=float(rng.uniform(50.0, 500.0)),
            resources={
                "oil": float(rng.uniform(0.2, 0.8)),
                "food": float(rng.uniform(0.3, 0.9)),
                "minerals": float(rng.uniform(0.2, 0.7)),
            },
            tech_level=float(rng.uniform(0.2, 0.6)),
            internal_stability=float(rng.uniform(0.5, 0.9)),
            territory=float(rng.uniform(0.3, 0.8)),
            relationships={},
            military_spending_pct=float(rng.uniform(0.02, 0.06)),
            archetype=archetypes[i],
        )
    # Initialize relationship vectors
    for nid in nation_ids:
        for other_id in nation_ids:
            if nid != other_id:
                nations[nid].relationships[other_id] = RelationshipVector(
                    trade_volume=float(rng.uniform(0.05, 0.25)),
                    alliance_strength=float(rng.uniform(-0.1, 0.2)),
                    hostility=float(rng.uniform(0.0, 0.2)),
                    grievance=float(rng.uniform(0.0, 0.1)),
                )
    return nations


class GeopoliticalEnv(AECEnv):
    """
    PettingZoo AECEnv subclass — Geopolitical Multi-Agent RL environment.

    AEC pattern: collect all actions, then resolve world step when last agent acts.
    Dynamics resolution order: Diplomacy → Military → Trade → Internal.
    """

    metadata = {
        "render_modes": ["human", "json"],
        "name": "geopolitical_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        nation_ids: Optional[list[str]] = None,
        max_steps: int = 500,
        noise_std: float = 0.05,
        enable_shocks: bool = False,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.nation_ids = nation_ids or ["alpha", "beta", "gamma", "delta", "epsilon"]
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.enable_shocks = enable_shocks
        self.render_mode = render_mode
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.possible_agents = list(self.nation_ids)
        self.agents = list(self.possible_agents)

        # Spaces
        self._action_spaces = {
            nid: make_action_space(len(self.nation_ids))
            for nid in self.possible_agents
        }
        self._obs_builder = ObservationBuilder(self.nation_ids, noise_std)
        self._observation_spaces = {
            nid: self._obs_builder.observation_space()
            for nid in self.possible_agents
        }

        # Resolvers
        self._diplomacy = DiplomacyResolver()
        self._military = MilitaryResolver()
        self._trade = TradeResolver()
        self._internal = InternalDynamicsResolver()
        self._shock_gen = ExogenousShockGenerator(rng=self._rng) if enable_shocks else None

        # AEC state
        self._agent_selector = AgentSelector(self.possible_agents)
        self._nations: dict[str, NationState] = {}
        self._prev_nations: dict[str, NationState] = {}
        self._pending_actions: dict[str, dict] = {}
        self._pending_wars: set[tuple[str, str]] = set()
        self._step_count: int = 0
        self._events: list[dict] = []

        # AEC required attributes
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict] = {}
        self.observations: dict[str, np.ndarray] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.agent_selection: str = ""

    # ------------------------------------------------------------------ #
    # AECEnv required interface                                            #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self._nations = _default_nations(self.nation_ids, self._rng)
        self._prev_nations = {k: v.copy() for k, v in self._nations.items()}
        self._pending_actions = {}
        self._pending_wars = set()
        self._step_count = 0
        self._events = []

        if self._shock_gen:
            self._shock_gen.active_shocks.clear()

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.observations = {
            a: self._obs_builder.build(a, self._nations, self._rng)
            for a in self.agents
        }

    def observe(self, agent: str) -> np.ndarray:
        return self._obs_builder.build(agent, self._nations, self._rng)

    def step(self, action: dict) -> None:
        # Handle dead/done agents by removing them cleanly
        if self.terminations.get(self.agent_selection) or self.truncations.get(self.agent_selection):
            agent = self.agent_selection
            if action is not None:
                raise ValueError("when an agent is dead, the only valid action is None")
            # Remove dead agent from all dicts
            self.agents.remove(agent)
            del self.terminations[agent]
            del self.truncations[agent]
            del self.rewards[agent]
            del self._cumulative_rewards[agent]
            del self.infos[agent]
            # Clear rewards dict (keys must match agents)
            self._clear_rewards()
            # Rebuild selector from remaining agents and advance
            if self.agents:
                self._agent_selector = AgentSelector(self.agents)
                self.agent_selection = self._agent_selector.next()
            return

        current_agent = self.agent_selection

        # Reset cumulative reward for current agent (consumed by last())
        self._cumulative_rewards[current_agent] = 0
        self._clear_rewards()

        self._pending_actions[current_agent] = action

        # Determine alive agents that still need to act this round
        # World step fires when all alive+active agents have submitted an action
        active_agents = [
            a for a in self.agents
            if not self.terminations.get(a, False) and not self.truncations.get(a, False)
        ]
        all_acted = all(a in self._pending_actions for a in active_agents)

        if all_acted:
            self._apply_world_step()
            self._step_count += 1
            # Update observations and check termination/truncation
            for a in list(self.agents):
                self.observations[a] = self._obs_builder.build(a, self._nations, self._rng)
                if not self._nations[a].alive:
                    self.terminations[a] = True
                if self._step_count >= self.max_steps:
                    self.truncations[a] = True

        # Accumulate rewards for all current agents
        self._accumulate_rewards()

        # Advance to next agent
        self.agent_selection = self._agent_selector.next()

    def action_space(self, agent: str) -> gymnasium.spaces.Dict:
        return self._action_spaces[agent]

    def observation_space(self, agent: str) -> gymnasium.spaces.Box:
        return self._observation_spaces[agent]

    def render(self) -> Optional[str]:
        if self.render_mode == "json":
            return self._snapshot_json()
        if self.render_mode == "human":
            print(self._snapshot_json())
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Internal world step                                                  #
    # ------------------------------------------------------------------ #

    def _apply_world_step(self) -> None:
        self._prev_nations = {k: v.copy() for k, v in self._nations.items()}
        self._events.clear()

        # Parse all actions
        encoders = {
            nid: ActionEncoder(self.nation_ids, nid)
            for nid in self.nation_ids
        }

        diplomatic_by_actor: dict[str, dict[str, int]] = {}
        budget_by_actor: dict[str, dict[str, float]] = {}
        trade_proposals: dict[str, set[str]] = {nid: set() for nid in self.nation_ids}
        sanctions: dict[str, set[str]] = {nid: set() for nid in self.nation_ids}

        for agent_id, raw_action in self._pending_actions.items():
            if not self._nations[agent_id].alive:
                continue
            enc = encoders[agent_id]
            decoded = enc.decode(raw_action)

            budget_by_actor[agent_id] = {
                ch: float(decoded.budget[i])
                for i, ch in enumerate(["military", "trade_investment", "tech_rd", "internal_dev", "reserves"])
            }

            dipl_map = {}
            for idx, code in enumerate(decoded.diplomatic):
                if idx < len(enc.targets):
                    target_id = enc.targets[idx]
                    dipl_map[target_id] = code
                    if code == 1:  # propose_trade
                        trade_proposals[agent_id].add(target_id)
                    elif code == 3:  # impose_sanctions
                        sanctions[agent_id].add(target_id)
            diplomatic_by_actor[agent_id] = dipl_map

        # 1. Diplomacy
        self._diplomacy.resolve(self._nations, diplomatic_by_actor, self._pending_wars)

        # 2. Military
        for agent_id, budget in budget_by_actor.items():
            if self._nations[agent_id].alive:
                self._military.apply_military_buildup(
                    self._nations[agent_id], budget.get("military", 0.0)
                )
        self._military.resolve_wars(self._nations, self._pending_wars, self._rng, self._events)

        # 3. Trade
        self._trade.resolve(self._nations, trade_proposals, sanctions)

        # 4. Internal
        regime_change_pending: list[str] = []
        for agent_id, budget in budget_by_actor.items():
            if self._nations[agent_id].alive:
                self._internal.resolve(self._nations[agent_id], budget, regime_change_pending)

        # Handle regime changes (Phase 1: simple reset)
        for nid in regime_change_pending:
            self._handle_regime_change_phase1(nid)

        # 5. Exogenous shocks (Phase 2+)
        if self._shock_gen:
            self._shock_gen.step(self._nations, self._step_count, self._events)

        # Compute rewards
        for a in self.agents:
            n_curr = self._nations[a]
            n_prev = self._prev_nations[a]
            weights = ARCHETYPE_WEIGHTS.get(n_curr.archetype, ARCHETYPE_WEIGHTS["mercantile"])
            calc = RewardCalculator(weights)
            self.rewards[a] = calc.compute(n_prev, n_curr)

        self._pending_actions.clear()

    def _handle_regime_change_phase1(self, nation_id: str) -> None:
        """Phase 1 regime change: simple stability reset + archetype change."""
        nation = self._nations[nation_id]
        nation.internal_stability = 0.4
        nation.archetype = self._rng.choice(ARCHETYPES)
        nation.age = 0
        # Decay grievances slightly
        for rel in nation.relationships.values():
            rel.grievance *= 0.7
        self._events.append({"type": "REGIME_CHANGE", "nation": nation_id})

    def _accumulate_rewards(self) -> None:
        for a in self.agents:
            self._cumulative_rewards[a] = self._cumulative_rewards.get(a, 0.0) + self.rewards.get(a, 0.0)

    def _clear_rewards(self) -> None:
        self.rewards = {a: 0.0 for a in self.agents}

    def _snapshot_json(self) -> str:
        import json
        data = {
            "step": self._step_count,
            "nations": {
                nid: {
                    "gdp": n.gdp,
                    "military_strength": n.military_strength,
                    "internal_stability": n.internal_stability,
                    "territory": n.territory,
                    "archetype": n.archetype,
                    "alive": n.alive,
                    "relationships": {
                        tid: r.to_array()
                        for tid, r in n.relationships.items()
                    },
                }
                for nid, n in self._nations.items()
            },
            "events": self._events[-20:],
        }
        return json.dumps(data, indent=2)

    def get_world_snapshot(self) -> dict:
        """Return a serializable snapshot of current world state."""
        return {
            "step": self._step_count,
            "nations": {
                nid: {
                    "gdp": n.gdp,
                    "military_strength": n.military_strength,
                    "population": n.population,
                    "resources": dict(n.resources),
                    "tech_level": n.tech_level,
                    "internal_stability": n.internal_stability,
                    "territory": n.territory,
                    "archetype": n.archetype,
                    "alive": n.alive,
                    "age": n.age,
                    "relationships": {
                        tid: {
                            "trade_volume": r.trade_volume,
                            "alliance_strength": r.alliance_strength,
                            "hostility": r.hostility,
                            "grievance": r.grievance,
                        }
                        for tid, r in n.relationships.items()
                    },
                }
                for nid, n in self._nations.items()
            },
            "events": list(self._events),
            "active_shocks": [
                {
                    "type": s.shock_type.value,
                    "nation": s.affected_nation,
                    "steps_remaining": s.steps_remaining,
                }
                for s in (self._shock_gen.active_shocks if self._shock_gen else [])
            ],
        }
