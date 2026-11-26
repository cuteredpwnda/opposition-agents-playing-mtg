"""Activated keywords & alternative-cost mechanics.

Implements:

- **Equip {N}** (CR 702.6): activated ability of an Equipment that
  attaches it to a creature you control. Sorcery speed.
- **Crew {N}** (CR 702.121): tap N+ power's worth of creatures to
  turn a Vehicle into an artifact creature until end of turn.
- **Flashback {cost}** (CR 702.34): cast from graveyard for the
  flashback cost; on resolution, exile instead of going to graveyard.
- **Kicker {cost}** (CR 702.33): pay an additional cost as the spell
  is cast. We surface a second CAST_SPELL action with
  ``metadata['kicked'] = True`` and add the kicker mana to the
  effective cost.

These hooks are surfaced as legal actions by ``rules_engine``; this
module is the parser + executor.
"""

from __future__ import annotations

import re
from typing import Optional

from .game_state import CardInstance, GameState, PlayerState, Zone
from .mana import auto_tap_for_cost, can_pay, parse_mana_cost, pay_cost


# ---------------------------------------------------------------------------
# Cost parsers
# ---------------------------------------------------------------------------


_EQUIP_RE = re.compile(r"\bequip\s+(\{[^\}]+\}(?:\s*\{[^\}]+\})*|\d+)", re.IGNORECASE)
_CREW_RE = re.compile(r"\bcrew\s+(\d+)", re.IGNORECASE)
_FLASHBACK_RE = re.compile(
    r"\bflashback\s+(\{[^\}]+\}(?:\s*\{[^\}]+\})*)", re.IGNORECASE,
)
_KICKER_RE = re.compile(
    r"\bkicker\s+(\{[^\}]+\}(?:\s*\{[^\}]+\})*)", re.IGNORECASE,
)


def parse_equip_cost(card: CardInstance) -> Optional[str]:
    text = card.oracle_text or ""
    if "Equipment" not in card.type_line:
        return None
    m = _EQUIP_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    # Equip 2 → {2}
    if raw.isdigit():
        return "{" + raw + "}"
    return raw


def parse_crew_cost(card: CardInstance) -> Optional[int]:
    if "Vehicle" not in card.type_line:
        return None
    m = _CREW_RE.search(card.oracle_text or "")
    return int(m.group(1)) if m else None


def parse_flashback_cost(card: CardInstance) -> Optional[str]:
    m = _FLASHBACK_RE.search(card.oracle_text or "")
    return m.group(1) if m else None


def parse_kicker_cost(card: CardInstance) -> Optional[str]:
    m = _KICKER_RE.search(card.oracle_text or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Equip
# ---------------------------------------------------------------------------


def can_equip(state: GameState, equipment: CardInstance, target: CardInstance) -> bool:
    if equipment.zone != Zone.BATTLEFIELD or target.zone != Zone.BATTLEFIELD:
        return False
    if "Equipment" not in equipment.type_line:
        return False
    if not target.is_creature():
        return False
    if equipment.controller_id != target.controller_id:
        return False
    return True


def execute_equip(
    state: GameState, equipment: CardInstance, target: CardInstance,
) -> bool:
    """Pay the equip cost and attach. Returns True on success."""
    cost_text = parse_equip_cost(equipment)
    if cost_text is None or not can_equip(state, equipment, target):
        return False
    player = next((p for p in state.players if p.player_id == equipment.controller_id), None)
    if player is None:
        return False
    cost = parse_mana_cost(cost_text)
    if not can_pay(player, cost):
        auto_tap_for_cost(state, player, cost)
    if not can_pay(player, cost):
        state.log(f"{player.name} cannot pay equip {cost_text} for {equipment.name}")
        return False
    pay_cost(player, cost)
    # Clear bonuses on the creature this equipment was previously attached to.
    if equipment.attached_to and equipment.attached_to != target.instance_id:
        prev = next(
            (c for c in state.cards if c.instance_id == equipment.attached_to),
            None,
        )
        if prev is not None:
            prev.counters.pop("equip_pwr", None)
            prev.counters.pop("equip_tou", None)
    equipment.attached_to = target.instance_id
    state.log(f"{equipment.name} equipped to {target.name}")
    # Apply +X/+Y from oracle text "Equipped creature gets +N/+M".
    m = re.search(
        r"equipped creature gets \+(\d+)/\+(\d+)",
        equipment.oracle_text or "", re.IGNORECASE,
    )
    if m:
        target.counters["equip_pwr"] = int(m.group(1))
        target.counters["equip_tou"] = int(m.group(2))
    return True


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------


def can_crew(
    state: GameState, vehicle: CardInstance, crew_creatures: list[CardInstance],
) -> bool:
    crew_n = parse_crew_cost(vehicle)
    if crew_n is None or vehicle.zone != Zone.BATTLEFIELD:
        return False
    total = 0
    for c in crew_creatures:
        if c.zone != Zone.BATTLEFIELD or not c.is_creature() or c.tapped:
            return False
        if c.controller_id != vehicle.controller_id:
            return False
        try:
            total += int(c.power) if c.power is not None else 0
        except (TypeError, ValueError):
            total += 0
    return total >= crew_n


def execute_crew(
    state: GameState, vehicle: CardInstance, crew_creatures: list[CardInstance],
) -> bool:
    if not can_crew(state, vehicle, crew_creatures):
        return False
    for c in crew_creatures:
        c.tapped = True
    # Vehicle becomes an artifact creature until end of turn.
    if "Creature" not in vehicle.type_line:
        vehicle.card_data["type_line"] = vehicle.type_line + " Creature"
    vehicle.counters["crewed_eot"] = 1
    state.log(f"{vehicle.name} is crewed")
    return True


def clear_crew_eot(state: GameState) -> None:
    """Cleanup step: revert vehicles that were crewed this turn."""
    for c in state.cards:
        if c.zone != Zone.BATTLEFIELD:
            continue
        if c.counters.pop("crewed_eot", 0) > 0:
            tl = c.card_data.get("type_line", "")
            if " Creature" in tl and "Vehicle" in tl:
                c.card_data["type_line"] = tl.replace(" Creature", "")


# ---------------------------------------------------------------------------
# Flashback
# ---------------------------------------------------------------------------


def can_flashback(card: CardInstance, player: PlayerState, state: GameState) -> bool:
    if card.zone != Zone.GRAVEYARD:
        return False
    cost_text = parse_flashback_cost(card)
    if cost_text is None:
        return False
    cost = parse_mana_cost(cost_text)
    if can_pay(player, cost):
        return True
    snapshot = dict(player.mana_pool)
    auto_tap_for_cost(state, player, cost)
    ok = can_pay(player, cost)
    player.mana_pool = snapshot
    return ok


# ---------------------------------------------------------------------------
# Kicker
# ---------------------------------------------------------------------------


def kicker_cost_dict(card: CardInstance) -> Optional[dict]:
    cost_text = parse_kicker_cost(card)
    return parse_mana_cost(cost_text) if cost_text else None
