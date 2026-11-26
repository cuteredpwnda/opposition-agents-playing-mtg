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

from src.engine_legacy.game_state import Action, GameState


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
        # Determine strategy based on archetype matchup
        if opponent_archetype:
            opp_arch = opponent_archetype.lower()
            if "control" in opp_arch:
                self.current_strategy = "aggressive"  # Race before they set up
            elif "aggro" in opp_arch:
                self.current_strategy = "controlling"  # Stabilize and trade
            elif "combo" in opp_arch:
                self.current_strategy = "disruption"  # Interrupt their win
            elif "ramp" in opp_arch:
                self.current_strategy = "tempo"  # Race early advantage
            else:
                self.current_strategy = "balanced"
        else:
            self.current_strategy = "balanced"
        
        return self.current_strategy

    async def plan_turn(self, game_state: GameState) -> TurnPlan:
        """Level 1: Plan the broad strokes of this turn."""
        from src.engine_legacy.game_state import Zone
        
        # Analyze current board state
        our_player = game_state.players[0]
        opponent = game_state.players[1] if len(game_state.players) > 1 else None
        
        # Identify priority targets (threats to remove or cards to play)
        priority_targets = []
        mana_reservation = {}
        
        # If opponent has threats, prioritize removal
        if opponent:
            opponent_threats = [
                c.card_name for c in opponent.battlefield
                if c.type_line and "creature" in c.type_line.lower()
            ]
            priority_targets = opponent_threats[:3]  # Top 3 threats
        
        # Analyze our hand for playable prioritizations
        mana_available = our_player.mana_pool.total
        
        # Reserve mana based on strategy
        if self.current_strategy == "disruption":
            mana_reservation = {"blue": 2, "black": 1}  # Hold for counterspells
        elif self.current_strategy == "controlling":
            mana_reservation = {"blue": 2}  # Hold up removal
        elif self.current_strategy == "aggressive":
            mana_reservation = {}  # Tap out for threats
        
        self.current_turn_plan = TurnPlan(
            strategy=self.current_strategy,
            priority_targets=priority_targets,
            mana_reservation=mana_reservation,
        )
        return self.current_turn_plan

    async def select_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Level 0: Choose specific card/action consistent with turn plan."""
        from src.engine_legacy.game_state import ActionType
        
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Filter actions based on turn plan constraints
        best_action = None
        best_score = -float('inf')
        
        for action in legal_actions:
            # Skip pass actions unless no other option
            if action.action_type == ActionType.PASS_PRIORITY:
                continue
            
            # Score action based on priority targets
            score = 0.0
            
            if action.card_instance_id:
                # Check if this action targets a priority threat
                if self.current_turn_plan:
                    for target in self.current_turn_plan.priority_targets:
                        if target in str(action).lower():
                            score += 10.0
                
                # Bonus for removal spells
                if action.action_type == ActionType.CAST_SPELL:
                    score += 5.0  # Basic spell value
            
            if score > best_score:
                best_score = score
                best_action = action
        
        # Fallback to first non-pass action, then pass
        if best_action is None:
            for action in legal_actions:
                if action.action_type != ActionType.PASS_PRIORITY:
                    return action
        
        return best_action or legal_actions[0]
