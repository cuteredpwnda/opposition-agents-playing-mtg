"""
Random agent — picks a legal action uniformly at random. Baseline.

Reference: open-mtg (MIT) — https://github.com/hlynurd/open-mtg
"""

from __future__ import annotations

import random

from src.agents.base_agent import MTGAgent
from src.engine.game_state import Action, GameState


class RandomAgent(MTGAgent):
    """Baseline agent that picks a uniformly random legal action."""

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        return random.choice(legal_actions)
