"""
Deck utilities for training and benchmarking.
"""

from __future__ import annotations

from typing import List, Dict, Any


def create_mock_deck() -> List[Dict[str, Any]]:
    """Create a simplified mock deck for testing and benchmarking."""
    # Balanced 60-card deck with lands and a mix of creatures/spells
    deck = []
    for i in range(24):
        deck.append({
            "name": "Mountain",
            "type_line": "Land",
            "mana_cost": "",
            "cmc": 0,
            "oracle_text": "{T}: Add {R}",
            "power": None,
            "toughness": None,
        })

    for i in range(18):
        deck.append({
            "name": f"Goblin_{i}",
            "type_line": "Creature — Goblin",
            "mana_cost": "{R}",
            "cmc": 1,
            "oracle_text": "Haste",
            "power": "1",
            "toughness": "1",
        })

    for i in range(10):
        deck.append({
            "name": f"LightningBolt_{i}",
            "type_line": "Instant",
            "mana_cost": "{R}",
            "cmc": 1,
            "oracle_text": "Deal 3 damage to target creature or player.",
            "power": None,
            "toughness": None,
        })

    for i in range(8):
        deck.append({
            "name": f"BurnSpell_{i}",
            "type_line": "Sorcery",
            "mana_cost": "{1}{R}",
            "cmc": 2,
            "oracle_text": "Deal 2 damage divided as you choose.",
            "power": None,
            "toughness": None,
        })

    # Ensure 60 cards
    if len(deck) < 60:
        deck.extend(deck[:60 - len(deck)])

    return deck
