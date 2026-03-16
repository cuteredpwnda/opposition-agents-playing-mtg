"""
Replacement effects — "instead" effects that modify game events before they happen.

Reference: MTG rules 614.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from src.engine.game_state import GameState, Zone


@dataclass
class ReplacementEffect:
    """Representation of a replacement effect."""
    source_card: str  # Card that creates this effect
    event_type: str  # "zone_change", "damage", "draw", etc.
    condition: str  # Description of when it applies
    replacement: str  # What happens instead


class ReplacementEffectRegistry:
    """Registry of active replacement effects on the board."""
    
    def __init__(self):
        self.effects: dict[str, list[ReplacementEffect]] = {}
    
    def add_effect(self, effect: ReplacementEffect) -> None:
        """Register a replacement effect."""
        event_type = effect.event_type
        if event_type not in self.effects:
            self.effects[event_type] = []
        self.effects[event_type].append(effect)
    
    def remove_effect(self, effect: ReplacementEffect) -> None:
        """Unregister a replacement effect."""
        event_type = effect.event_type
        if event_type in self.effects:
            self.effects[event_type] = [
                e for e in self.effects[event_type] if e.source_card != effect.source_card
            ]
    
    def get_effects(self, event_type: str) -> list[ReplacementEffect]:
        """Get all replacement effects for an event type."""
        return self.effects.get(event_type, [])


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
    
    Args:
        game_state: Current game state
        event: Event dict with keys like 'type', 'source_card', 'target', etc.
    
    Returns:
        Modified event dict, or None if event is completely replaced
    """
    if not hasattr(game_state, 'replacement_effects'):
        game_state.replacement_effects = ReplacementEffectRegistry()
    
    event_type = event.get('type', '')
    
    # Get all replacement effects that could apply
    effects = game_state.replacement_effects.get_effects(event_type)
    
    # Apply each replacement effect (in proper order)
    # For simplicity, apply in registration order
    for effect in effects:
        # Check if effect's condition is met
        # (Simplified: assume condition always met for now)
        event['replaced_by'] = effect.source_card
        event['original_event'] = str(event)
        
        # Modify event based on replacement
        if 'zone_change' in event_type:
            # Example: Dredge effect — change destination zone
            if 'destination_zone' in event:
                original = event['destination_zone']
                event['destination_zone'] = Zone.HAND
                event['replacement_text'] = f"changed destination zone from {original} to hand"
    
    return event
