"""Channel — Kamigawa: Neon Dynasty mechanic (CR 702.x).

Channel is an activated ability you activate from your hand. The
canonical wording is::

    Channel — {cost}, Discard this card: <effect>

Activating channel pays the mana cost, discards the card, and puts the
``<effect>`` on the stack as an ability (so it can be responded to,
unlike cycling which is a special action).
"""

from __future__ import annotations

import re
from typing import Optional

from .game_state import CardInstance, GameState, PlayerState, StackItem, Zone
from .mana import parse_mana_cost, can_pay, pay_cost, auto_tap_for_cost


_CHANNEL_RE = re.compile(
    r"channel\s*[\u2014\-]\s*"                 # "Channel —"
    r"(\{[^\}]+\}(?:\s*\{[^\}]+\})*)"           # mana cost (group 1)
    r"\s*,\s*discard\s+[^:]+?"                  # ", Discard <this card / name>"
    r"\s*:\s*(.+?)(?=\Z|\n\n)",                  # ": <effect>" (group 2)
    re.IGNORECASE | re.DOTALL,
)


def parse_channel(card: CardInstance) -> Optional[tuple[str, str]]:
    """Return ``(mana_cost, effect_text)`` or ``None`` if the card has no
    channel ability."""
    text = card.oracle_text or ""
    m = _CHANNEL_RE.search(text)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def can_pay_channel(state: GameState, player: PlayerState, cost_text: str) -> bool:
    cost = parse_mana_cost(cost_text)
    if can_pay(player, cost):
        return True
    snapshot = dict(player.mana_pool)
    auto_tap_for_cost(state, player, cost)
    ok = can_pay(player, cost)
    player.mana_pool = snapshot
    return ok


def execute_channel(
    state: GameState, card: CardInstance, player: PlayerState
) -> GameState:
    """Pay the channel cost, discard the card, and put its effect on
    the stack as an ability."""
    from .zones import move_card

    parsed = parse_channel(card)
    if parsed is None:
        return state
    cost_text, effect_text = parsed
    cost = parse_mana_cost(cost_text)

    if not can_pay(player, cost):
        auto_tap_for_cost(state, player, cost)
    if not can_pay(player, cost):
        state.log(f"{player.name} cannot pay channel cost for {card.name}")
        return state

    pay_cost(player, cost)

    # Discard the channel card from hand to graveyard.
    state = move_card(
        state, card.instance_id, Zone.HAND, Zone.GRAVEYARD, player.player_id,
    )
    state.log(f"{player.name} channels {card.name} ({cost_text})")

    # Push the channel effect onto the stack as an ability so opponents
    # may respond. The resolver reads ``oracle_text`` from card_data.
    item = StackItem(
        source_card_id=card.instance_id,
        controller_id=player.player_id,
        is_spell=False,
        card_data={
            "name": f"{card.name} (channel)",
            "oracle_text": effect_text,
            "type_line": "Ability",
        },
    )
    # Auto-pick targets the same way modal sub-items do.
    from .spell_effects import auto_pick_targets
    item.targets = auto_pick_targets(state, card, player.player_id)
    state.stack.append(item)
    state.log(f"  -> [Channel effect] {effect_text} added to stack")
    return state
