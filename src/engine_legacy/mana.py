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

    Also includes mana from non-land permanents whose only activation cost
    is ``{T}`` (Sol Ring, Birds of Paradise, Llanowar Elves, …) — see
    :func:`permanent_mana_production`.
    """
    available = {c: player.mana_pool.get(c, 0) for c in "WUBRGC"}
    for card in state.cards:
        if (
            card.zone == Zone.BATTLEFIELD
            and card.controller_id == player.player_id
            and not card.tapped
        ):
            if "Land" in card.type_line:
                color = _land_produces(card.card_data)
                available[color] = available.get(color, 0) + 1
                continue
            produced = permanent_mana_production(card)
            if produced is None:
                continue
            # Summoning-sick creatures without haste can't tap for mana.
            if (
                "Creature" in card.type_line
                and card.summoning_sick
                and "haste" not in (card.oracle_text or "").lower()
            ):
                continue
            for color, n in produced.items():
                available[color] = available.get(color, 0) + n
    return available


# Pattern: "{T}: Add {C}{C}" / "{T}: Add {G}" / "{T}: Add {U}{B}"
_TAP_ADD_PATTERN = re.compile(
    r"(?:^|\n|\.)\s*\{t\}\s*:\s*add\s+((?:\{[wubrgc]\}\s*)+)",
    re.IGNORECASE,
)
_ANY_COLOR_PATTERN = re.compile(
    r"\{t\}\s*:\s*add\s+one\s+mana\s+of\s+any\s+color", re.IGNORECASE
)


def permanent_mana_production(card) -> Optional[dict[str, int]]:
    """Return the mana produced by ``card``'s simplest ``{T}: Add …`` ability.

    Returns ``None`` if the card has no recognisable plain ``{T}`` mana
    ability. Recognises:

    * ``{T}: Add {C}{C}`` → ``{"C": 2}``
    * ``{T}: Add {G}`` → ``{"G": 1}``
    * ``{T}: Add {U}{B}`` → ``{"U": 1, "B": 1}``
    * ``{T}: Add one mana of any color`` → ``{"C": 1}`` (treated as
      colorless for cast affordability — caller will deal with the colour
      requirement at cast time).

    Lands are excluded — use :func:`_land_produces` for those. Abilities
    requiring extra costs (sacrifice, mana, life payment, etc.) are
    rejected to keep the surface deterministic.
    """
    if "Land" in card.type_line:
        return None
    text = card.oracle_text or ""
    if not text:
        return None
    # Reject any ability that bundles an additional cost (e.g. Mana Vault's
    # `{T}: Add {C}{C}{C}` is fine, but Treasure's `{T}, Sacrifice …` is not).
    # We do this by scanning for a comma-separated extra cost in front of the
    # add clause: `{T}, Sacrifice CARDNAME: Add ...`.
    if re.search(r"\{t\}\s*,\s*[^:]+:\s*add\s+", text, re.IGNORECASE):
        return None
    m = _TAP_ADD_PATTERN.search(text)
    if m:
        symbols = re.findall(r"\{([wubrgc])\}", m.group(1), re.IGNORECASE)
        out: dict[str, int] = {}
        for s in symbols:
            out[s.upper()] = out.get(s.upper(), 0) + 1
        return out or None
    if _ANY_COLOR_PATTERN.search(text):
        return {"C": 1}
    return None


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
    then generic. Also taps non-land permanents with simple ``{T}`` mana
    abilities (Sol Ring, Birds of Paradise, Llanowar Elves, …) when basics
    alone aren't enough.
    """
    if can_pay(player, cost):
        return True
    # Index untapped sources by produced color. Lands and permanents share
    # the same color → list[card] mapping; we drain lands first because
    # they have no summoning-sickness or alt-use concerns.
    untapped: dict[str, list] = {c: [] for c in "WUBRGC"}
    rocks: list[tuple[str, int, "object"]] = []  # (primary_color, total, card)
    for card in state.cards:
        if (
            card.zone != Zone.BATTLEFIELD
            or card.controller_id != player.player_id
            or card.tapped
        ):
            continue
        if "Land" in card.type_line:
            untapped[_land_produces(card.card_data)].append(card)
            continue
        produced = permanent_mana_production(card)
        if produced is None:
            continue
        if (
            "Creature" in card.type_line
            and card.summoning_sick
            and "haste" not in (card.oracle_text or "").lower()
        ):
            continue
        # Multi-mana rocks (e.g. Sol Ring → {C}{C}) need to fire as a unit.
        if sum(produced.values()) > 1:
            primary = max(produced, key=produced.get)
            rocks.append((primary, sum(produced.values()), card))
            # Only the primary slot can satisfy a colored requirement; the
            # rest spills into generic. Track via the rocks list.
        else:
            color = next(iter(produced))
            untapped[color].append(card)
    # Pay colored requirements
    for color in "WUBRG":
        deficit = cost.get(color, 0) - player.mana_pool.get(color, 0)
        while deficit > 0 and untapped.get(color):
            land = untapped[color].pop()
            _tap_for_mana(state, player, land)
            deficit -= 1
        if deficit > 0:
            return False
    # Pay generic from any remaining untapped lands / single-color rocks /
    # multi-mana rocks.
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
                _tap_for_mana(state, player, land)
                generic_remaining -= 1
        # Multi-mana rocks last (their mana is harder to colour-match).
        while generic_remaining > 0 and rocks:
            _, total, card = rocks.pop()
            _tap_for_mana(state, player, card)
            generic_remaining -= total
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
        state.log(f"    \u25cb {pname} taps {card.name} for {{{color}}}")
    return color


def _tap_for_mana(state: GameState, player: PlayerState, card) -> None:
    """Tap any untapped land or simple mana-rock and pour its mana in.

    Internal helper used by :func:`auto_tap_for_cost`. Splits cleanly on
    "Land" vs other-permanent so the existing land-tap log still fires.
    """
    if card.tapped:
        return
    if "Land" in card.type_line:
        tap_land_for_mana(state, player, card.instance_id)
        return
    produced = permanent_mana_production(card)
    if produced is None:
        return
    card.tapped = True
    pname = player.name or player.player_id
    pretty = "".join(f"{{{c}}}" * n for c, n in produced.items())
    state.log(f"    \u25cb {pname} taps {card.name} for {pretty}")
    for c, n in produced.items():
        add_mana(player, c, n)


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
