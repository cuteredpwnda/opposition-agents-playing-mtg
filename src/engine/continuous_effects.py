"""
Continuous effects — static effects modifying game rules (layer system).

Reference: MTG rules 611.
"""

from __future__ import annotations

from dataclasses import dataclass
from src.engine.game_state import GameState, CardInstance


@dataclass
class ContinuousEffect:
    """Representation of a continuous effect."""
    source_card: str
    layer: int  # 1-10 according to CR 611
    effect_type: str  # "power_toughness", "type", "ability", etc.
    modification: dict  # What to modify
    timestamp: float  # For tie-breaking


class ContinuousEffectRegistry:
    """Registry of active continuous effects on the board."""
    
    def __init__(self):
        self.effects: list[ContinuousEffect] = []
    
    def add_effect(self, effect: ContinuousEffect) -> None:
        """Register a continuous effect."""
        self.effects.append(effect)
        # Sort by layer then timestamp
        self.effects.sort(key=lambda e: (e.layer, e.timestamp))
    
    def remove_effect(self, source_card: str) -> None:
        """Remove all effects from a source card."""
        self.effects = [e for e in self.effects if e.source_card != source_card]
    
    def get_effects_by_layer(self, layer: int) -> list[ContinuousEffect]:
        """Get all effects for a specific layer."""
        return [e for e in self.effects if e.layer == layer]


def apply_continuous_effects(game_state: GameState) -> GameState:
    """
    Apply all continuous effects using the layer system.
    
    Layers (in order, timestamp tiebreaker):
    1. Copy effects
    2-6. Type/subtype/supertype changes
    7. Power/toughness changes
    8. Ability changes
    9. Special actions (timestamp order)
    10. Effect order (timestamp order)
    """
    if not hasattr(game_state, 'continuous_effects'):
        game_state.continuous_effects = ContinuousEffectRegistry()
    
    # Process each layer in order
    for layer in range(1, 11):
        effects = game_state.continuous_effects.get_effects_by_layer(layer)
        
        for effect in effects:
            # Apply effect modifications
            if effect.effect_type == "power_toughness":
                # Layer 7: Modify power/toughness
                for card in game_state.cards:
                    if card.card_name == effect.source_card:
                        if 'power_bonus' in effect.modification:
                            card.power = (int(card.power or 0) + effect.modification['power_bonus'])
                        if 'toughness_bonus' in effect.modification:
                            card.toughness = (int(card.toughness or 0) + effect.modification['toughness_bonus'])
            
            elif effect.effect_type == "type":
                # Layer 4: Add/remove types
                for card in game_state.cards:
                    if card.card_name == effect.source_card:
                        if 'add_type' in effect.modification:
                            card.type_line = f"{card.type_line} {effect.modification['add_type']}"
            
            elif effect.effect_type == "ability":
                # Layer 6: Add/remove abilities
                for card in game_state.cards:
                    if card.card_name == effect.source_card:
                        if 'add_ability' in effect.modification:
                            card.oracle_text = f"{card.oracle_text}. {effect.modification['add_ability']}"
    
    return game_state
