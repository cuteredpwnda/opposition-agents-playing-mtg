"""
Random agent — picks legal actions with bias toward playing threats. Baseline.

Reference: open-mtg (MIT) — https://github.com/hlynurd/open-mtg
"""

from __future__ import annotations

import random

from src.agents.base_agent import MTGAgent
from src.engine.game_state import Action, GameState, ActionType


class RandomAgent(MTGAgent):
    """Baseline agent that picks legal actions with strategic bias."""

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Choose an action with bias: prefer CAST_SPELL > PLAY_LAND > ACTIVATE_ABILITY > PASS"""
        if not legal_actions:
            return Action(action_type=ActionType.PASS_PRIORITY, player_id=self.player_id)
        
        # Weight actions by type
        weighted = []
        for action in legal_actions:
            if action.action_type == ActionType.CAST_SPELL:
                weight = 10  # Strongly prefer casting creatures/spells
            elif action.action_type == ActionType.PLAY_LAND:
                weight = 5   # Prefer playing lands
            elif action.action_type == ActionType.ACTIVATE_ABILITY:
                weight = 2   # Mana abilities less preferred
            elif action.action_type == ActionType.PASS_PRIORITY:
                weight = 1   # Pass is last resort
            else:
                weight = 1
            
            weighted.extend([action] * weight)
        
        return random.choice(weighted)
