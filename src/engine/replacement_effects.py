"""
Replacement effects — "instead" effects that modify game events before they happen.

Reference: MTG rules 614.
"""

from __future__ import annotations

from src.engine.game_state import GameState


def apply_replacement_effects(
    game_state: GameState, event: dict
) -> dict:
    """
    Apply replacement effects that modify an event.
    
    For example:
    - "If a card would enter your graveyard from your library, instead
      you may put it into your hand." (Dredge)
    - "If a creature you control would deal combat damage to a player,
      instead that player mills that many cards." (Mill)
    """
    # Stub: would need effect registry + matching by event type
    # For now, pass through unchanged
    return event
