"""
Combat system — declare attackers, declare blockers, assign damage.

References:
- open-mtg (MIT): CR 509/510-compliant damage assignment ordering
- mtg-python-engine (MIT): handle_combat_phase() pattern
"""

from __future__ import annotations

from .game_state import CardInstance, CombatState, GameState, Zone


def begin_combat(state: GameState) -> CombatState:
    """Initialize combat state for the active player's combat phase."""
    combat = CombatState()
    state.combat = combat
    state.log("Combat begins")
    return combat


def declare_attackers(
    state: GameState, attacker_assignments: dict[str, str]
) -> None:
    """Declare attackers. attacker_assignments: {card_instance_id: defending_player_id}"""
    if state.combat is None:
        return

    for attacker_id, defender_id in attacker_assignments.items():
        card = next((c for c in state.cards if c.instance_id == attacker_id), None)
        if card is None:
            continue
        if card.zone != Zone.BATTLEFIELD or card.tapped or card.summoning_sick:
            continue
        if not card.is_creature():
            continue

        card.tapped = True  # Attacking taps creatures (unless vigilance)
        state.combat.attackers[attacker_id] = defender_id
        state.log(f"{card.name} attacks {defender_id}")


def declare_blockers(
    state: GameState, blocker_assignments: dict[str, list[str]]
) -> None:
    """Declare blockers. blocker_assignments: {attacker_id: [blocker_card_ids]}"""
    if state.combat is None:
        return

    for attacker_id, blocker_ids in blocker_assignments.items():
        if attacker_id not in state.combat.attackers:
            continue
        state.combat.blockers[attacker_id] = blocker_ids
        for bid in blocker_ids:
            blocker = next((c for c in state.cards if c.instance_id == bid), None)
            if blocker:
                state.log(f"{blocker.name} blocks {_card_name(state, attacker_id)}")


def resolve_combat_damage(state: GameState) -> None:
    """Assign and resolve combat damage."""
    if state.combat is None:
        return

    for attacker_id, defender_id in state.combat.attackers.items():
        attacker = _get_card(state, attacker_id)
        if attacker is None:
            continue

        blockers = state.combat.blockers.get(attacker_id, [])
        atk_power = _parse_pt(attacker.power)

        if not blockers:
            # Unblocked — damage goes to defending player
            defender = next((p for p in state.players if p.player_id == defender_id), None)
            if defender:
                defender.life_total -= atk_power
                state.log(f"{attacker.name} deals {atk_power} damage to {defender.name}")
        else:
            # Blocked — assign damage to blockers in order, blockers hit back
            remaining_power = atk_power
            for blocker_id in blockers:
                blocker = _get_card(state, blocker_id)
                if blocker is None:
                    continue
                blk_toughness = _parse_pt(blocker.toughness)
                damage_to_blocker = min(remaining_power, blk_toughness)
                blocker.damage_marked += damage_to_blocker
                remaining_power -= damage_to_blocker

                # Blocker deals damage back to attacker
                blk_power = _parse_pt(blocker.power)
                attacker.damage_marked += blk_power

    state.combat = None


def end_combat(state: GameState) -> None:
    """Clean up combat state."""
    state.combat = None
    state.log("Combat ends")


# Helpers
def _get_card(state: GameState, cid: str) -> CardInstance | None:
    return next((c for c in state.cards if c.instance_id == cid), None)


def _card_name(state: GameState, cid: str) -> str:
    card = _get_card(state, cid)
    return card.name if card else "Unknown"


def _parse_pt(value: str | None) -> int:
    """Parse power/toughness — returns 0 for * or None."""
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0
