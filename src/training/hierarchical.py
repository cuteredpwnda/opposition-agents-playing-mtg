"""
Hierarchical learning — multi-level decision hierarchy.

Level 3: Meta-Strategy (between games) — deck/sideboard selection
Level 2: Game Plan (per turn) — overall strategy for this turn
Level 1: Turn Tactics (per priority pass) — specific sequencing
Level 0: Card Actions (individual plays) — which card/action to take

Reference: Section 12.4 of PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.game_state import Action, GameState


@dataclass
class TurnPlan:
    """High-level plan for a turn."""

    strategy: str  # e.g., "aggro", "hold_up_interaction", "deploy_threats"
    priority_targets: list[str]  # cards to play/prioritize
    mana_reservation: dict[str, int]  # mana to hold open


class HierarchicalAgent:
    """Multi-level decision hierarchy for MTG.

    Each level operates at a different temporal scale and abstraction level.
    Higher levels set goals/constraints for lower levels.
    """

    def __init__(self):
        self.current_strategy: str = "balanced"
        self.current_turn_plan: TurnPlan | None = None

    async def set_game_strategy(
        self, my_deck_info: dict[str, Any], opponent_archetype: str | None
    ) -> str:
        """Level 3/2: Determine overall game plan at start of game.

        Based on deck composition and opponent matchup analysis.
        """
        # Stub — would use meta_policy network
        if opponent_archetype and "control" in opponent_archetype.lower():
            self.current_strategy = "aggressive"
        elif opponent_archetype and "aggro" in opponent_archetype.lower():
            self.current_strategy = "controlling"
        else:
            self.current_strategy = "balanced"
        return self.current_strategy

    async def plan_turn(self, game_state: GameState) -> TurnPlan:
        """Level 1: Plan the broad strokes of this turn."""
        # Stub — would use game_policy network
        self.current_turn_plan = TurnPlan(
            strategy=self.current_strategy,
            priority_targets=[],
            mana_reservation={},
        )
        return self.current_turn_plan

    async def select_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Level 0: Choose specific card/action consistent with turn plan."""
        # Stub — would use action_policy network, constrained by turn plan
        # Default: pick first non-pass action, or pass
        for action in legal_actions:
            if action.action_type.value != "pass_priority":
                return action
        return legal_actions[0]
