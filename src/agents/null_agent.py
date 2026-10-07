"""Null (goldfish) agent — always passes priority.

Used as the *do-nothing* opponent in solo deck simulations ("goldfishing").
It never casts spells, plays lands, attacks, or blocks.  Its only response
to any decision is ``PASS_PRIORITY`` (or ``CONCEDE`` if that is the only
legal action).
"""

from __future__ import annotations

from src.agents.base_agent import MTGAgent
from src.engine_legacy.game_state import Action, ActionType, GameState


class NullAgent(MTGAgent):
    """Passive opponent for goldfishing — always passes priority."""

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        # Prefer PASS_PRIORITY; fall back to the first available action
        # (which may be CONCEDE if NullAgent is dead).
        for action in legal_actions:
            if action.action_type == ActionType.PASS_PRIORITY:
                return action
        return legal_actions[0] if legal_actions else Action(
            action_type=ActionType.PASS_PRIORITY,
            player_id=self.player_id,
        )
