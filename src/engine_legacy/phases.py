"""
Phase/step transitions for the turn structure.

Reference: open-mtg (MIT) phases as enum with current_phase_index.
"""

from __future__ import annotations

from .game_state import GameState, Phase, StackItem, Trigger, TriggerType, Zone

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


def _scan_phase_triggers(state: GameState, phrase: str, kind: TriggerType) -> None:
    """Scan battlefield permanents for "at the beginning of <phrase>" triggers.

    Pushes a stub trigger onto the stack for the active player's permanents
    (the most common case; "your upkeep", "each player's upkeep" etc. are
    treated identically for now).
    """
    import re

    active_id = state.active_player.player_id
    pattern = re.compile(
        r"at the beginning of (?:your |each player['\u2019]?s? )?" + phrase
        + r"[, ]+\s*(.+?)(?:\.|$)",
        re.IGNORECASE,
    )
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        if card.controller_id != active_id and "each player" not in (card.oracle_text or "").lower():
            # Only fire for active player's permanents unless text says "each".
            continue
        text = card.oracle_text or ""
        m = pattern.search(text)
        if not m:
            continue
        effect = m.group(1).strip()
        trig = Trigger(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            trigger_type=kind,
            description=f"{card.name}: {effect}",
        )
        state.triggered_abilities.append(trig)
        state.stack.append(StackItem(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            is_spell=False,
            card_data={
                "name": f"[{kind.value}] {card.name}",
                "type_line": "Ability",
                "oracle_text": effect,
            },
        ))
        state.log(f"[TRIGGER ({kind.value})] {card.name}: {effect}")


def _on_enter_phase(state: GameState, phase: Phase) -> None:
    """Hooks fired immediately upon entering a step."""
    if phase == Phase.UNTAP:
        # Active player's permanents untap (CR 502). Stun counters skip
        # untap and decrement instead (CR 701.49).
        from .counters import apply_stun_on_untap
        active_id = state.active_player.player_id
        for c in state.cards:
            if c.zone != Zone.BATTLEFIELD or c.controller_id != active_id:
                continue
            if not c.tapped:
                continue
            if apply_stun_on_untap(c):
                state.log(f"{c.name} stays tapped (stun counter)")
                continue
            c.tapped = False
            c.summoning_sick = False  # Summoning sickness wears off at untap.
    elif phase == Phase.UPKEEP:
        _scan_phase_triggers(state, "upkeep", TriggerType.UPKEEP)
    elif phase == Phase.COMBAT_BEGIN:
        _scan_phase_triggers(state, "combat", TriggerType.BEGINNING_OF_COMBAT)
    elif phase == Phase.END_STEP:
        _scan_phase_triggers(state, "end step", TriggerType.END_STEP)


def advance_phase(state: GameState) -> Phase:
    """Advance to the next phase/step. Returns the new phase."""
    current_idx = PHASE_ORDER.index(state.phase)

    # Cleanup step (CR 514): clear damage from creatures and end-of-turn
    # buffs (prowess, +X/+X "until end of turn"). Also revert vehicles
    # that were crewed this turn.
    if state.phase == Phase.CLEANUP:
        from .keywords import clear_eot_buffs
        from .equip import clear_crew_eot
        for c in state.cards:
            c.damage_marked = 0
        clear_eot_buffs(state)
        clear_crew_eot(state)

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

    _on_enter_phase(state, state.phase)

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
