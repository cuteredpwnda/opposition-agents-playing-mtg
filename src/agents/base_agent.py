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

    def __init__(self, player_id: str, name: str = ""):
        self.player_id = player_id
        self.name = name or player_id
        self.memory = AgentMemory()
        # Most-recent reasoning trace — populated by ``set_reasoning``
        # at the end of ``decide_action``.  The priority loop reads this
        # after the agent returns and forwards it to the JSONL trace.
        self.last_reasoning: dict | None = None

    def set_reasoning(self, trace) -> None:
        """Record the rationale for the most recent action.

        ``trace`` may be a :class:`~src.agents.reasoning.ReasoningTrace`
        instance, a plain dict, or ``None`` (clears).
        """
        from src.agents.reasoning import attach_reasoning
        attach_reasoning(self, trace)

    @abc.abstractmethod
    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Choose an action from the legal actions given the game state."""
        ...

    async def observe(self, game_state: GameState, action: Action) -> None:
        """Observe an action taken (by any player). Override for belief updates."""
        self.memory.game_log.append(
            f"T{game_state.turn_number} {action.action_type.value}"
        )

    # ------------------------------------------------------------------
    # Mulligan hooks (London mulligan).  Default implementations delegate
    # to the strategy-aware policy in ``src/agents/mulligan.py`` so every
    # agent has sensible behaviour out of the box; LLM / world-model /
    # active-inference agents override these for learned decisions.
    # ------------------------------------------------------------------

    @property
    def strategy(self):  # pragma: no cover - thin accessor
        """Strategy enum value used by the default mulligan heuristics.

        Subclasses that have a real strategy should override this; the
        default of ``None`` triggers the generic fallback heuristic.
        """
        return None

    def decide_mulligan(
        self,
        hand,
        mulligans_taken: int,
        max_mulligans: int,
    ) -> bool:
        """Return True to KEEP the current opening hand, False to mulligan.

        Default: strategy-aware land-and-curve heuristic.
        """
        from src.agents.mulligan import should_keep

        return should_keep(
            hand,
            strategy=self.strategy,
            mulligans_taken=mulligans_taken,
            max_mulligans=max_mulligans,
        )

    def select_bottom_cards(self, hand, n: int) -> list:
        """Return the ``n`` cards from ``hand`` to put on the bottom."""
        from src.agents.mulligan import select_bottom_cards as _bottom

        return _bottom(hand, n, strategy=self.strategy)

    def reset(self) -> None:
        """Reset agent state for a new game."""
        self.memory = AgentMemory()
