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
    player_ids = list(game_state.players.keys())
    if game_state.active_player not in player_ids:
        return player_ids
    idx = player_ids.index(game_state.active_player)
    return player_ids[idx:] + player_ids[:idx]


def advance_priority(game_state: GameState) -> GameState:
    """Move priority to the next player in APNAP order."""
    order = get_priority_order(game_state)
    if game_state.priority_player not in order:
        game_state.priority_player = order[0]
        return game_state
    idx = order.index(game_state.priority_player)
    next_idx = (idx + 1) % len(order)
    game_state.priority_player = order[next_idx]
    return game_state


def all_players_passed(
    game_state: GameState, passed: set[str]
) -> bool:
    """Check if all players have passed priority in order."""
    return set(game_state.players.keys()) == passed


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
