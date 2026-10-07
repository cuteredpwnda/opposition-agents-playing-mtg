"""Cycling — CR 702.32.

Cycling is a special action: pay the cycling cost, discard the card
from hand, then draw a card. It is not a triggered ability and does
not use the stack (the discard does, however, allow other "when you
cycle" triggers to fire — left for a future pass).
"""

from __future__ import annotations

import re
from typing import Optional

from .game_state import CardInstance, GameState, PlayerState, Zone
from .mana import parse_mana_cost, can_pay, pay_cost, auto_tap_for_cost


_CYCLING_RE = re.compile(
    r"\b(?:typecycling\s+)?cycling\s+(\{[^\}]+\}(?:\s*\{[^\}]+\})*)",
    re.IGNORECASE,
)


def parse_cycling_cost(card: CardInstance) -> Optional[str]:
    """Return the raw cycling cost string (e.g. "{1}{U}") or None."""
    text = card.oracle_text or ""
    m = _CYCLING_RE.search(text)
    if not m:
        return None
    return m.group(1)


def can_pay_cycling(state: GameState, player: PlayerState, cost_text: str) -> bool:
    cost = parse_mana_cost(cost_text)
    if can_pay(player, cost):
        return True
    # Allow auto-tap from untapped basics
    snapshot = dict(player.mana_pool)
    auto_tap_for_cost(state, player, cost)
    ok = can_pay(player, cost)
    # Restore — execute_cycle will re-tap on actual execution
    player.mana_pool = snapshot
    return ok


def execute_cycle(state: GameState, card: CardInstance, player: PlayerState) -> GameState:
    """Pay cycling cost, discard the card, draw a card."""
    from .zones import move_card

    cost_text = parse_cycling_cost(card)
    if cost_text is None:
        return state
    cost = parse_mana_cost(cost_text)

    if not can_pay(player, cost):
        auto_tap_for_cost(state, player, cost)
    if not can_pay(player, cost):
        state.log(f"{player.name} cannot pay cycling cost for {card.name}")
        return state

    pay_cost(player, cost)

    # Discard
    state = move_card(state, card.instance_id, Zone.HAND, Zone.GRAVEYARD, player.player_id)
    state.log(f"{player.name} cycles {card.name}")

    # Draw a card
    library = [c for c in state.cards if c.zone == Zone.LIBRARY and c.owner_id == player.player_id]
    if library:
        drawn = library[0]
        drawn.zone = Zone.HAND
        drawn.known_to.add(player.player_id)
        state.log(f"  -> {player.name} draws {drawn.name}")
    else:
        # Empty library on draw is handled by SBAs at end of phase.
        state.log(f"  -> {player.name} attempted to draw from empty library")

    return state
