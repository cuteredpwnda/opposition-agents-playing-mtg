"""
Zone management — moving cards between zones.

Reference: mtg-python-engine (MIT) gameobject.py zone transitions.
"""

from __future__ import annotations

from .game_state import CardInstance, GameState, Zone


def move_card(
    state: GameState, 
    card_instance_id: str, 
    from_zone: Zone, 
    to_zone: Zone,
    player_id: str
) -> GameState:
    """Move a card to a new zone, resetting zone-specific state."""
    card = next((c for c in state.cards if c.instance_id == card_instance_id), None)
    if card is None:
        return state
    
    card.zone = to_zone

    # Reset battlefield-specific state when leaving the battlefield
    if from_zone == Zone.BATTLEFIELD:
        card.tapped = False
        card.damage_marked = 0
        card.summoning_sick = True
        card.attached_to = None
        card.counters.clear()

    # Cards entering the battlefield are summoning-sick
    if to_zone == Zone.BATTLEFIELD:
        card.summoning_sick = True

    # Human-readable, narrative log message based on transition.
    pname = _player_label(state, player_id) or _player_label(state, card.controller_id)
    is_land = "Land" in (card.card_data.get("type_line") or "")
    msg: str
    if from_zone == Zone.LIBRARY and to_zone == Zone.HAND:
        msg = f"    {pname} draws {card.name}"
    elif from_zone == Zone.HAND and to_zone == Zone.BATTLEFIELD and is_land:
        msg = f"    {pname} plays {card.name}"
    elif from_zone == Zone.HAND and to_zone == Zone.GRAVEYARD:
        msg = f"    {pname} discards {card.name}"
    elif from_zone == Zone.HAND and to_zone == Zone.STACK:
        # Cast event is logged by rules_engine with cost/targets — keep silent.
        msg = ""
    elif from_zone == Zone.STACK and to_zone == Zone.BATTLEFIELD:
        msg = f"    ✚ {card.name} enters the battlefield under {pname}'s control"
    elif from_zone == Zone.STACK and to_zone == Zone.GRAVEYARD:
        # Resolution already logged by stack/rules_engine.
        msg = ""
    elif from_zone == Zone.BATTLEFIELD and to_zone == Zone.GRAVEYARD:
        msg = f"    ✖ {card.name} → {pname}'s graveyard"
    elif to_zone == Zone.EXILE:
        # Exile from any zone (CR 406). Use a distinct glyph (⦻) so
        # it's easy to grep for in the pretty log.
        msg = f"    ⦻ {card.name} is exiled from {from_zone.value} ({pname})"
    elif from_zone == Zone.BATTLEFIELD and to_zone == Zone.HAND:
        msg = f"    ↩ {card.name} returns to {pname}'s hand"
    elif from_zone == Zone.LIBRARY and to_zone == Zone.BATTLEFIELD:
        msg = f"    ✚ {pname} puts {card.name} onto the battlefield from library"
    elif from_zone == Zone.COMMAND_ZONE and to_zone == Zone.STACK:
        msg = ""  # cast-from-command-zone logged by rules_engine
    elif from_zone == Zone.BATTLEFIELD and to_zone == Zone.COMMAND_ZONE:
        msg = f"    ⌂ {card.name} returns to the command zone"
    else:
        msg = f"    {card.name}: {from_zone.value} → {to_zone.value}"

    if msg:
        state.log(msg)
    return state


def _player_label(state: GameState, player_id: str) -> str:
    """Best-effort display name for a player (falls back to player_id)."""
    if not player_id:
        return ""
    p = next((pl for pl in state.players if pl.player_id == player_id), None)
    if p is None:
        return player_id
    return p.name or p.player_id


def get_cards_in_zone(state: GameState, player_id: str, zone: Zone) -> list[CardInstance]:
    """Get all cards a player controls in a given zone."""
    return state.cards_in_zone(player_id, zone)


def shuffle_library(state: GameState, player_id: str) -> None:
    """Shuffle ``player_id``'s library in place (CR 103.2 / 701.20).

    Uses the global :mod:`random` module so :func:`random.seed` (or the
    project's :func:`src.utils.seeding.set_global_seed`) makes shuffles
    reproducible. Logs a one-liner at the standard event-indent so the
    pretty game log shows when a library has been re-randomised
    (after fetches, tutors, ramp, mulligans-in-game, etc.).
    """
    import random

    library = [c for c in state.cards if c.zone == Zone.LIBRARY and c.owner_id == player_id]
    if len(library) < 2:
        return
    random.shuffle(library)
    # Splice the new order back into ``state.cards`` at the original
    # positions of the library cards so other-zone cards keep their slots.
    lib_iter = iter(library)
    new_cards: list[CardInstance] = []
    lib_idset = {c.instance_id for c in library}
    for c in state.cards:
        if c.instance_id in lib_idset:
            new_cards.append(next(lib_iter))
        else:
            new_cards.append(c)
    state.cards = new_cards
    label = _player_label(state, player_id)
    state.log(f"    \u21bb {label} shuffles their library")
