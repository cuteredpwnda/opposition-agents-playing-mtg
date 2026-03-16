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

    state.log(f"{card.name} moved from {from_zone.value} to {to_zone.value}")
    return state


def get_cards_in_zone(state: GameState, player_id: str, zone: Zone) -> list[CardInstance]:
    """Get all cards a player controls in a given zone."""
    return state.cards_in_zone(player_id, zone)
