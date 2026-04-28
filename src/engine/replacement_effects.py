"""Replacement effects framework (CR 614).

This module provides a small but actually-wired replacement-effects
pipeline. It handles a handful of common patterns:

* **Damage prevention** — "Prevent all combat damage that would be dealt
  to you this turn." (Holy Day, Fog-class effects).
* **Death replacement** — "If a creature would die, exile it instead."
  (Anafenza, Rest in Peace-style for creatures.)
* **Lifegain doubling** — "If you would gain life, you gain twice that
  much life instead." (Boon Reflection, Rhox Faithmender.)

The intent is *not* to model every replacement in Magic — there are
hundreds. The intent is to give the engine a single hook
(`apply_replacements`) that other modules can call, and a registry the
ETB/LTB machinery scans to install effects automatically as permanents
enter / leave the battlefield.

Reference: CR 614 (replacement effects).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.engine.game_state import CardInstance, GameState, Zone


@dataclass
class ReplacementEffect:
    """A registered replacement effect (CR 614.1)."""

    source_card_id: str
    controller_id: str
    event_type: str
    apply: Callable[[GameState, dict], Optional[dict]]
    description: str = ""


@dataclass
class ReplacementRegistry:
    by_event: dict[str, list[ReplacementEffect]] = field(default_factory=dict)

    def add(self, effect: ReplacementEffect) -> None:
        self.by_event.setdefault(effect.event_type, []).append(effect)

    def remove_for_card(self, card_id: str) -> None:
        for et, effects in list(self.by_event.items()):
            self.by_event[et] = [e for e in effects if e.source_card_id != card_id]

    def for_event(self, event_type: str) -> list[ReplacementEffect]:
        return list(self.by_event.get(event_type, []))


def _registry(state: GameState) -> ReplacementRegistry:
    reg = getattr(state, "replacement_registry", None)
    if reg is None or not isinstance(reg, ReplacementRegistry):
        reg = ReplacementRegistry()
        state.replacement_registry = reg
    return reg


# ---------------------------------------------------------------------------
# Pattern parsers
# ---------------------------------------------------------------------------


def _parse_damage_prevention(card: CardInstance) -> Optional[ReplacementEffect]:
    text = (card.oracle_text or "").lower()
    if "prevent all combat damage that would be dealt to you" in text:
        owner = card.controller_id

        def _apply(state: GameState, event: dict) -> Optional[dict]:
            if event.get("target_player") == owner and event.get("combat"):
                event["amount"] = 0
                event["replaced_by"] = card.instance_id
            return event

        return ReplacementEffect(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            event_type="damage_to_player",
            apply=_apply,
            description=f"{card.name}: prevent combat damage to you",
        )
    if "prevent all damage that would be dealt to you this turn" in text:
        owner = card.controller_id

        def _apply(state: GameState, event: dict) -> Optional[dict]:
            if event.get("target_player") == owner:
                event["amount"] = 0
                event["replaced_by"] = card.instance_id
            return event

        return ReplacementEffect(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            event_type="damage_to_player",
            apply=_apply,
            description=f"{card.name}: prevent all damage to you",
        )
    return None


def _parse_death_to_exile(card: CardInstance) -> Optional[ReplacementEffect]:
    pattern = re.compile(
        r"if (?:a |another )?creature[^.]*?would die[^.]*?,\s*exile it instead",
        re.IGNORECASE,
    )
    if pattern.search(card.oracle_text or ""):
        def _apply(state: GameState, event: dict) -> Optional[dict]:
            event["destination_zone"] = Zone.EXILE
            event["replaced_by"] = card.instance_id
            return event

        return ReplacementEffect(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            event_type="creature_dies",
            apply=_apply,
            description=f"{card.name}: dying creatures are exiled instead",
        )
    return None


def _parse_lifegain_double(card: CardInstance) -> Optional[ReplacementEffect]:
    text = (card.oracle_text or "").lower()
    if "if you would gain life" in text and (
        "twice that much" in text or "that much plus" in text
    ):
        owner = card.controller_id

        def _apply(state: GameState, event: dict) -> Optional[dict]:
            if event.get("target_player") == owner:
                event["amount"] = int(event.get("amount", 0)) * 2
                event["replaced_by"] = card.instance_id
            return event

        return ReplacementEffect(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            event_type="lifegain",
            apply=_apply,
            description=f"{card.name}: doubles your lifegain",
        )
    return None


_PARSERS: list[Callable[[CardInstance], Optional[ReplacementEffect]]] = [
    _parse_damage_prevention,
    _parse_death_to_exile,
    _parse_lifegain_double,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_replacements_for(state: GameState, card: CardInstance) -> None:
    """Scan ``card``'s oracle text and register any replacement effects."""
    reg = _registry(state)
    reg.remove_for_card(card.instance_id)
    for parser in _PARSERS:
        eff = parser(card)
        if eff is not None:
            reg.add(eff)


def remove_replacements_for(state: GameState, card_id: str) -> None:
    _registry(state).remove_for_card(card_id)


def apply_replacements(state: GameState, event: dict) -> Optional[dict]:
    """Run all registered replacements for ``event['type']`` over ``event``."""
    if not isinstance(event, dict) or "type" not in event:
        return event
    reg = _registry(state)
    for effect in reg.for_event(event["type"]):
        result = effect.apply(state, event)
        if result is None:
            return None
        event = result
    return event


def apply_lifegain(state: GameState, player_id: str, amount: int) -> int:
    """Apply lifegain replacements then mutate life. Returns gained amount."""
    if amount <= 0:
        return 0
    event = {"type": "lifegain", "target_player": player_id, "amount": amount}
    event = apply_replacements(state, event)
    if event is None:
        return 0
    final = int(event.get("amount", 0))
    if final <= 0:
        return 0
    player = next((p for p in state.players if p.player_id == player_id), None)
    if player is None:
        return 0
    player.life_total += final
    return final


def apply_damage_to_player(
    state: GameState,
    player_id: str,
    amount: int,
    *,
    combat: bool = False,
    source_card_id: str = "",
) -> int:
    """Apply damage replacements then mutate life. Returns damage dealt."""
    if amount <= 0:
        return 0
    event = {
        "type": "damage_to_player",
        "target_player": player_id,
        "amount": amount,
        "combat": combat,
        "source_card_id": source_card_id,
    }
    event = apply_replacements(state, event)
    if event is None:
        return 0
    final = int(event.get("amount", 0))
    if final <= 0:
        return 0
    player = next((p for p in state.players if p.player_id == player_id), None)
    if player is None:
        return 0
    player.life_total -= final
    return final
