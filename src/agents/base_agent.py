"""
Abstract agent interface for MTG gameplay.

Every agent must implement `decide_action` to choose from legal actions.
Reference: mtg-player (MIT) — structured tool-calling LLM agent pattern.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from src.engine.game_state import Action, GameState


@dataclass
class AgentMemory:
    """Per-game memory for an agent: observations, key moments, plans."""

    game_log: list[str] = field(default_factory=list)
    cards_seen_from_opponent: dict[str, list[str]] = field(default_factory=dict)
    key_moments: list[str] = field(default_factory=list)
    current_plan: str = ""


class MTGAgent(abc.ABC):
    """Base class for all MTG-playing agents.

    Subclasses must implement `decide_action`.
    """

    def __init__(self, player_id: str):
        self.player_id = player_id
        self.memory = AgentMemory()

    @abc.abstractmethod
    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Choose an action from the legal actions given the game state."""
        ...

    def observe(self, game_state: GameState, action: Action) -> None:
        """Observe an action taken (by any player). Override for belief updates."""
        self.memory.game_log.append(
            f"T{game_state.turn_number} {action.action_type.value}: "
            f"{action.card.name if action.card else 'pass'}"
        )

    def reset(self) -> None:
        """Reset agent state for a new game."""
        self.memory = AgentMemory()
