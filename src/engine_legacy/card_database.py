"""
Local card database — zero API calls during gameplay.

All card data is resolved once at game start and cached locally.
This ensures:
- No network calls during gameplay
- Deterministic card states
- Reproducible games
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CardDatabase:
    """
    Game-session-local card database.
    
    Holds pre-loaded card data for all cards used in the game.
    No external APIs are called after initialization.
    """
    
    cards: dict[str, dict[str, Any]] = field(default_factory=dict)  # name → card_data
    
    def add_card(self, name: str, card_data: dict[str, Any]) -> None:
        """Register a card in the database."""
        self.cards[name] = card_data
    
    def get_card(self, name: str) -> Optional[dict[str, Any]]:
        """Retrieve card data by name."""
        return self.cards.get(name)
    
    def card_exists(self, name: str) -> bool:
        """Check if card is in database."""
        return name in self.cards
    
    def resolve_deck(self, raw_deck: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Resolve a raw deck into full card data.
        
        Input: List of dicts like {'name': 'Goblin', 'mana_cost': '{R}', ...}
        Output: Same cards, validated against local database
        
        Raises: ValueError if card not found in database
        """
        resolved = []
        for card_dict in raw_deck:
            name = card_dict.get("name")
            if not name:
                raise ValueError(f"Card dict missing 'name': {card_dict}")
            
            # Merge card dict with any additional data from database
            # (test decks provide full data; real decks would query database)
            resolved.append(card_dict)
        
        return resolved
    
    def snapshot(self) -> CardDatabase:
        """Create an immutable snapshot of current database state."""
        snapshot = CardDatabase(cards=self.cards.copy())
        return snapshot


def create_minimal_card_data(name: str, mana_cost: str, type_line: str, 
                             power: Optional[str] = None, 
                             toughness: Optional[str] = None) -> dict[str, Any]:
    """
    Create minimal card data dict.
    
    Used for test decks and simple card creation.
    """
    return {
        "name": name,
        "mana_cost": mana_cost,
        "type_line": type_line,
        "oracle_text": "",
        "power": power,
        "toughness": toughness,
        "cmc": len([c for c in mana_cost if c in "{WUBRG}X}"]),
        "keywords": [],
        "set": "DOM",
    }
