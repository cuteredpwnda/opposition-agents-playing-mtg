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
        msg = f"{pname} draws {card.name}"
    elif from_zone == Zone.HAND and to_zone == Zone.BATTLEFIELD and is_land:
        msg = f"{pname} plays {card.name}"
    elif from_zone == Zone.HAND and to_zone == Zone.GRAVEYARD:
        msg = f"{pname} discards {card.name}"
    elif from_zone == Zone.HAND and to_zone == Zone.STACK:
        # Cast event is logged by rules_engine with cost/targets — keep silent.
        msg = ""
    elif from_zone == Zone.STACK and to_zone == Zone.BATTLEFIELD:
        msg = f"{card.name} enters the battlefield under {pname}'s control"
    elif from_zone == Zone.STACK and to_zone == Zone.GRAVEYARD:
        # Resolution already logged by stack/rules_engine.
        msg = ""
    elif from_zone == Zone.BATTLEFIELD and to_zone == Zone.GRAVEYARD:
        msg = f"{card.name} is put into {pname}'s graveyard"
    elif from_zone == Zone.BATTLEFIELD and to_zone == Zone.HAND:
        msg = f"{card.name} is returned to {pname}'s hand"
    elif from_zone == Zone.LIBRARY and to_zone == Zone.BATTLEFIELD:
        msg = f"{pname} puts {card.name} onto the battlefield from their library"
    elif from_zone == Zone.COMMAND_ZONE and to_zone == Zone.STACK:
        msg = ""  # cast-from-command-zone logged by rules_engine
    elif from_zone == Zone.BATTLEFIELD and to_zone == Zone.COMMAND_ZONE:
        msg = f"{card.name} is moved to the command zone"
    else:
        msg = f"{card.name} moves from {from_zone.value} to {to_zone.value}"

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
