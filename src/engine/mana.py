"""
Mana system — tap lands, pay costs, mana pool management.

Built fresh (no existing Python engine handles full complexity well).
Reference: Forge (Java, GPL — behavioral reference only).
"""

from __future__ import annotations

import re
from typing import Optional

from .game_state import GameState, PlayerState


# Mana cost parsing: {2}{W}{U} → generic=2, W=1, U=1
MANA_SYMBOL_PATTERN = re.compile(r"\{([WUBRGCX0-9]+)\}")


def parse_mana_cost(mana_cost_text: str) -> dict[str, int]:
    """Parse a Scryfall mana cost string into components.

    Example: "{2}{W}{U}" → {"generic": 2, "W": 1, "U": 1}
    """
    result: dict[str, int] = {"generic": 0}
    for symbol in MANA_SYMBOL_PATTERN.findall(mana_cost_text):
        if symbol.isdigit():
            result["generic"] += int(symbol)
        elif symbol == "X":
            pass  # X costs resolved at cast time
        else:
            result[symbol] = result.get(symbol, 0) + 1
    return result


def can_pay(player: PlayerState, cost: dict[str, int]) -> bool:
    """Check if a player can pay a mana cost with their current pool."""
    pool = dict(player.mana_pool)

    # Pay colored costs first
    for color in "WUBRGC":
        required = cost.get(color, 0)
        if pool.get(color, 0) < required:
            return False
        pool[color] = pool.get(color, 0) - required

    # Pay generic with remainder
    generic_needed = cost.get("generic", 0)
    total_remaining = sum(pool.values())
    return total_remaining >= generic_needed


def pay_cost(player: PlayerState, cost: dict[str, int]) -> bool:
    """Attempt to pay a mana cost. Returns True if successful."""
    if not can_pay(player, cost):
        return False

    # Pay colored costs
    for color in "WUBRGC":
        required = cost.get(color, 0)
        player.mana_pool[color] -= required

    # Pay generic — consume cheapest available mana
    generic_needed = cost.get("generic", 0)
    for color in "CUBWRG":  # Prefer colorless, then least useful colors
        while generic_needed > 0 and player.mana_pool.get(color, 0) > 0:
            player.mana_pool[color] -= 1
            generic_needed -= 1

    return True


def add_mana(player: PlayerState, color: str, amount: int = 1) -> None:
    """Add mana to a player's mana pool."""
    player.mana_pool[color] = player.mana_pool.get(color, 0) + amount


def empty_mana_pool(player: PlayerState) -> None:
    """Empty a player's mana pool (happens at end of each phase)."""
    for color in player.mana_pool:
        player.mana_pool[color] = 0


def tap_land_for_mana(
    state: GameState, player: PlayerState, card_instance_id: str
) -> Optional[str]:
    """Tap a land to produce mana. Returns the color produced, or None."""
    card = next(
        (c for c in state.cards if c.instance_id == card_instance_id), None
    )
    if card is None or card.tapped:
        return None

    card.tapped = True
    # Determine mana production from card data
    color = _land_produces(card.card_data)
    if color:
        add_mana(player, color)
    return color


def _land_produces(card_data: dict) -> str:
    """Determine what color a basic land produces."""
    name = card_data.get("name", "")
    mapping = {
        "Plains": "W",
        "Island": "U",
        "Swamp": "B",
        "Mountain": "R",
        "Forest": "G",
    }
    return mapping.get(name, "C")
