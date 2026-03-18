from __future__ import annotations

from world.nation_state import NationState, RelationshipVector


class DiplomacyResolver:
    """Resolve diplomatic actions from all agents simultaneously."""

    # Action codes matching action_space.py
    DO_NOTHING = 0
    PROPOSE_TRADE = 1
    PROPOSE_ALLIANCE = 2
    IMPOSE_SANCTIONS = 3
    THREATEN = 4
    DECLARE_WAR = 5
    NEGOTIATE_PEACE = 6

    def resolve(
        self,
        nations: dict[str, NationState],
        actions: dict[str, dict],  # nation_id -> {target_id -> action_code}
        pending_wars: set[tuple[str, str]],
    ) -> None:
        """Mutate nations in-place. Populate pending_wars with (attacker, defender) pairs."""
        for actor_id, target_actions in actions.items():
            actor = nations.get(actor_id)
            if actor is None or not actor.alive:
                continue
            for target_id, action_code in target_actions.items():
                target = nations.get(target_id)
                if target is None or not target.alive:
                    continue
                self._apply_action(actor, target, action_code, pending_wars)

    def _apply_action(
        self,
        actor: NationState,
        target: NationState,
        code: int,
        pending_wars: set[tuple[str, str]],
    ) -> None:
        rel_at = actor.get_relationship(target.nation_id)
        rel_ta = target.get_relationship(actor.nation_id)

        if code == self.DO_NOTHING:
            pass

        elif code == self.PROPOSE_TRADE:
            # Unilateral proposal improves trade a little
            rel_at.trade_volume = min(1.0, rel_at.trade_volume + 0.03)
            rel_ta.trade_volume = min(1.0, rel_ta.trade_volume + 0.03)
            rel_at.hostility = max(0.0, rel_at.hostility - 0.01)

        elif code == self.PROPOSE_ALLIANCE:
            rel_at.alliance_strength = min(1.0, rel_at.alliance_strength + 0.05)
            rel_ta.alliance_strength = min(1.0, rel_ta.alliance_strength + 0.05)
            rel_at.hostility = max(0.0, rel_at.hostility - 0.02)

        elif code == self.IMPOSE_SANCTIONS:
            rel_at.trade_volume = max(0.0, rel_at.trade_volume - 0.1)
            rel_ta.trade_volume = max(0.0, rel_ta.trade_volume - 0.1)
            rel_at.hostility = min(1.0, rel_at.hostility + 0.05)
            rel_ta.hostility = min(1.0, rel_ta.hostility + 0.03)
            rel_ta.grievance = min(1.0, rel_ta.grievance + 0.02)

        elif code == self.THREATEN:
            rel_at.hostility = min(1.0, rel_at.hostility + 0.08)
            rel_ta.hostility = min(1.0, rel_ta.hostility + 0.05)
            rel_ta.grievance = min(1.0, rel_ta.grievance + 0.03)

        elif code == self.DECLARE_WAR:
            rel_at.hostility = 1.0
            rel_ta.hostility = 1.0
            rel_at.alliance_strength = min(rel_at.alliance_strength, 0.0)
            rel_ta.alliance_strength = min(rel_ta.alliance_strength, 0.0)
            pending_wars.add((actor.nation_id, target.nation_id))

        elif code == self.NEGOTIATE_PEACE:
            rel_at.hostility = max(0.0, rel_at.hostility - 0.15)
            rel_ta.hostility = max(0.0, rel_ta.hostility - 0.15)
            rel_at.grievance = max(0.0, rel_at.grievance - 0.05)
            rel_ta.grievance = max(0.0, rel_ta.grievance - 0.05)
            # Remove from pending wars if present
            pending_wars.discard((actor.nation_id, target.nation_id))
            pending_wars.discard((target.nation_id, actor.nation_id))
