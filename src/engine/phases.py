"""
Phase/step transitions for the turn structure.

Reference: open-mtg (MIT) phases as enum with current_phase_index.
"""

from __future__ import annotations

from .game_state import GameState, Phase

# Ordered list of phases within a turn
PHASE_ORDER: list[Phase] = [
    Phase.UNTAP,
    Phase.UPKEEP,
    Phase.DRAW,
    Phase.MAIN_1,
    Phase.COMBAT_BEGIN,
    Phase.COMBAT_ATTACKERS,
    Phase.COMBAT_BLOCKERS,
    Phase.COMBAT_DAMAGE,
    Phase.COMBAT_END,
    Phase.MAIN_2,
    Phase.END_STEP,
    Phase.CLEANUP,
]


def advance_phase(state: GameState) -> Phase:
    """Advance to the next phase/step. Returns the new phase."""
    current_idx = PHASE_ORDER.index(state.phase)
    next_idx = current_idx + 1

    if next_idx >= len(PHASE_ORDER):
        # End of turn — advance to next player's untap
        next_idx = 0
        state.active_player_index = (state.active_player_index + 1) % len(state.players)
        state.turn_number += 1
        # Reset per-turn state
        active = state.active_player
        active.land_plays_remaining = 1
        active.has_drawn_for_turn = False
        state.log(f"--- Turn {state.turn_number}: {active.name}'s turn ---")

    state.phase = PHASE_ORDER[next_idx]
    state.priority_player_index = state.active_player_index

    # Reset priority passing for all players
    for p in state.players:
        p.passed_priority = False

    return state.phase


def is_main_phase(phase: Phase) -> bool:
    return phase in (Phase.MAIN_1, Phase.MAIN_2)


def is_combat_phase(phase: Phase) -> bool:
    return phase in (
        Phase.COMBAT_BEGIN,
        Phase.COMBAT_ATTACKERS,
        Phase.COMBAT_BLOCKERS,
        Phase.COMBAT_DAMAGE,
        Phase.COMBAT_END,
    )
