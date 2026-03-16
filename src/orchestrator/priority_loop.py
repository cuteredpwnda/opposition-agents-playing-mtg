"""
Priority passing logic — APNAP order for multiplayer.

In MTG, after any game event players receive priority in Active Player,
Non-Active Player (APNAP) order. An item on the stack resolves only when
all players pass priority in succession without adding anything.

Reference: Section 5.3 of PLAN.md, CR 117.
"""

from __future__ import annotations

from src.engine.game_state import GameState
from src.engine.stack import is_empty as stack_is_empty


def get_priority_order(game_state: GameState) -> list[str]:
    """Return player IDs in APNAP order starting from active player."""
    player_ids = [p.player_id for p in game_state.players]
    active_player_id = game_state.active_player.player_id
    
    if active_player_id not in player_ids:
        return player_ids
    
    idx = player_ids.index(active_player_id)
    return player_ids[idx:] + player_ids[:idx]


def advance_priority(game_state: GameState) -> GameState:
    """Move priority to the next player in APNAP order."""
    order = get_priority_order(game_state)
    current_priority_id = game_state.priority_player.player_id
    
    if current_priority_id not in order:
        game_state.priority_player_index = 0
        return game_state
    
    idx = order.index(current_priority_id)
    next_idx = (idx + 1) % len(order)
    next_player_id = order[next_idx]
    
    # Find index of next player in players list
    game_state.priority_player_index = next(
        i for i, p in enumerate(game_state.players) if p.player_id == next_player_id
    )
    return game_state


def all_players_passed(
    game_state: GameState, passed: set[str]
) -> bool:
    """Check if all players have passed priority in order."""
    all_player_ids = {p.player_id for p in game_state.players}
    return all_player_ids == passed


def priority_action_result(
    game_state: GameState,
    passed_players: set[str],
) -> str:
    """Determine what happens after a priority pass.

    Returns:
        'resolve'       — all passed, stack non-empty → resolve top
        'advance_phase' — all passed, stack empty → move to next phase
        'continue'      — not all passed, keep going
    """
    if all_players_passed(game_state, passed_players):
        if stack_is_empty(game_state):
            return "advance_phase"
        return "resolve"
    return "continue"
