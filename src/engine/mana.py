"""
Mana system — tap lands, pay costs, mana pool management.

Built fresh (no existing Python engine handles full complexity well).
Reference: Forge (Java, GPL — behavioral reference only).
"""

from __future__ import annotations

import re
from typing import Optional

from .game_state import GameState, PlayerState, Zone


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


def potential_mana(state: GameState, player: PlayerState) -> dict[str, int]:
    """Sum of current pool plus mana producible by untapped basic lands.

    Used to expose ``CAST_SPELL`` actions whenever the player *could* afford
    the spell after auto-tapping; the engine then auto-taps lands during
    cast resolution. Treating mana abilities as implicit removes a whole
    class of priority-loop livelock that hits naive agents (Random/Heuristic)
    which would otherwise cycle ``ACTIVATE_ABILITY`` forever instead of
    passing priority.
    """
    available = {c: player.mana_pool.get(c, 0) for c in "WUBRGC"}
    for card in state.cards:
        if (
            card.zone == Zone.BATTLEFIELD
            and card.controller_id == player.player_id
            and not card.tapped
            and "Land" in card.type_line
        ):
            color = _land_produces(card.card_data)
            available[color] = available.get(color, 0) + 1
    return available


def can_pay_with_lands(state: GameState, player: PlayerState, cost: dict[str, int]) -> bool:
    """Like ``can_pay`` but also counts producible mana from untapped lands."""
    pool = dict(potential_mana(state, player))
    for color in "WUBRGC":
        required = cost.get(color, 0)
        if pool.get(color, 0) < required:
            return False
        pool[color] = pool.get(color, 0) - required
    return sum(pool.values()) >= cost.get("generic", 0)


def auto_tap_for_cost(state: GameState, player: PlayerState, cost: dict[str, int]) -> bool:
    """Tap untapped lands as needed so that ``can_pay`` returns True.

    Returns True on success, False if even after tapping every land the cost
    cannot be met. Lands are tapped greedily: colored requirements first,
    then generic.
    """
    if can_pay(player, cost):
        return True
    # Index lands by produced color
    untapped: dict[str, list] = {c: [] for c in "WUBRGC"}
    for card in state.cards:
        if (
            card.zone == Zone.BATTLEFIELD
            and card.controller_id == player.player_id
            and not card.tapped
            and "Land" in card.type_line
        ):
            untapped[_land_produces(card.card_data)].append(card)
    # Pay colored requirements
    for color in "WUBRG":
        deficit = cost.get(color, 0) - player.mana_pool.get(color, 0)
        while deficit > 0 and untapped.get(color):
            land = untapped[color].pop()
            tap_land_for_mana(state, player, land.instance_id)
            deficit -= 1
        if deficit > 0:
            return False
    # Pay generic from any remaining untapped lands
    generic_remaining = cost.get("generic", 0) - max(
        0,
        sum(player.mana_pool.values()) - sum(
            cost.get(c, 0) for c in "WUBRGC"
        ),
    )
    if generic_remaining > 0:
        for color in "CWUBRG":
            while generic_remaining > 0 and untapped.get(color):
                land = untapped[color].pop()
                tap_land_for_mana(state, player, land.instance_id)
                generic_remaining -= 1
    return can_pay(player, cost)


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
        pname = player.name or player.player_id
        state.log(f"{pname} taps {card.name} for {{{color}}}")
    return color


def _land_produces(card_data: dict) -> str:
    """Determine what color a basic land produces.

    Resolution order:
    1. Explicit oracle text ``"Add {X}"`` where X∈WUBRGC.
    2. Basic-land name (or any name starting with one).
    3. Fallback to colorless ``C``.
    """
    name = card_data.get("name", "") or ""
    oracle = card_data.get("oracle_text", "") or ""

    # 1) Oracle-text scan: "Add {R}", "{T}: Add {U}", etc.
    match = re.search(r"add\s*\{([WUBRGC])\}", oracle, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2) Basic-land name prefix match (handles "Mountain_8", "Plains 12", ...)
    mapping = {
        "Plains": "W",
        "Island": "U",
        "Swamp": "B",
        "Mountain": "R",
        "Forest": "G",
        "Wastes": "C",
    }
    for basic, color in mapping.items():
        if name == basic or name.startswith(basic):
            return color

    return "C"
