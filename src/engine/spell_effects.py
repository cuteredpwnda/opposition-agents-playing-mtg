"""Spell- and ability-effect resolver.

Translates the simplified oracle-text vocabulary used by the benchmark
archetype decks (and many real Scryfall cards) into actual game-state
mutations: damage, draw, destroy, bounce, counter, lifegain, ramp,
discard, anthem-style +1/+1.

Targets are picked greedily at *cast time* (see
:func:`auto_pick_targets`) so naive agents (Random, Heuristic) don't have
to enumerate every legal target combination — keeping the action space
small while still letting the engine evaluate non-trivial spells.

This module is intentionally additive: it does not remove or change the
behaviour of ``triggers.resolve_trigger``; it is invoked from
``rules_engine.resolve_spell`` for non-creature spells and from
ETB/cast-trigger handlers when their effect text matches a known pattern.
"""
from __future__ import annotations

import re
from typing import Optional

from .game_state import (
    CardInstance,
    GameState,
    PlayerState,
    StackItem,
    Zone,
)
from .keywords import can_be_targeted, has as has_kw, ward_cost


# ---------------------------------------------------------------------------
# Effect detection (oracle-text → kind)
# ---------------------------------------------------------------------------


def detect_effect_kind(oracle_text: str) -> str:
    """Return a coarse effect kind for the spell.

    Order matters: a "counter target spell" effect must beat the
    plain "spell" word; a "destroy target creature" beats a generic
    "creature" mention.
    """
    text = (oracle_text or "").lower()
    if "counter target spell" in text or "counter target" in text:
        return "counter"
    if "exile target" in text:
        return "exile"
    if "destroy target" in text:
        return "destroy"
    if "fight" in text and "target" in text:
        return "fight"
    if (re.search(r"return target .* to (its|their) owner['\u2019]s hand", text)
            or ("return target" in text and "hand" in text)):
        return "bounce"
    if "tap target" in text:
        return "tap"
    if "deal" in text and "damage" in text:
        return "damage"
    if "mill" in text or ("put" in text and "top" in text and "graveyard" in text):
        return "mill"
    if "create" in text and "token" in text:
        return "token"
    if "scry" in text:
        return "scry"
    if "surveil" in text:
        return "surveil"
    if "draw" in text and "card" in text and "discard" not in text:
        return "draw"
    if "search your library" in text and "land" in text:
        return "ramp"
    if "discard" in text:
        return "discard"
    if "gain" in text and "life" in text:
        return "lifegain"
    return "noop"

# ---------------------------------------------------------------------------
# Target picking
# ---------------------------------------------------------------------------


def _opponent(state: GameState, player_id: str) -> Optional[PlayerState]:
    return next((p for p in state.players if p.player_id != player_id), None)


def _opponent_creatures(state: GameState, player_id: str, source: CardInstance | None = None) -> list[CardInstance]:
    return [
        c for c in state.cards
        if c.zone == Zone.BATTLEFIELD
        and c.controller_id != player_id
        and c.is_creature()
        and can_be_targeted(c, source, player_id)
    ]


def _opponent_noncreature_perms(state: GameState, player_id: str, source: CardInstance | None = None) -> list[CardInstance]:
    out: list[CardInstance] = []
    for c in state.cards:
        if c.zone != Zone.BATTLEFIELD:
            continue
        if c.controller_id == player_id:
            continue
        type_line = c.type_line.lower()
        if ("artifact" in type_line or "enchantment" in type_line) and can_be_targeted(c, source, player_id):
            out.append(c)
    return out


def _power(card: CardInstance) -> int:
    try:
        return int(card.power) if card.power is not None else 0
    except ValueError:
        return 0


def auto_pick_targets(
    state: GameState, source_card: CardInstance, controller_id: str
) -> list[str]:
    """Pick reasonable default targets for ``source_card``.

    Returned ids are either ``CardInstance.instance_id`` or a
    ``PlayerState.player_id`` (``"player_2"`` etc.). The resolver
    distinguishes by lookup.
    """
    kind = detect_effect_kind(source_card.oracle_text)
    opp = _opponent(state, controller_id)
    if not opp:
        return []

    if kind == "counter":
        # Top spell on stack that isn't ours.
        for item in reversed(state.stack):
            if item.is_spell and item.controller_id != controller_id:
                return [item.item_id]
        return []

    if kind == "destroy" or kind == "exile":
        text = source_card.oracle_text.lower()
        if "creature" in text:
            creatures = _opponent_creatures(state, controller_id, source_card)
            if creatures:
                creatures.sort(key=_power, reverse=True)
                return [creatures[0].instance_id]
        if "artifact" in text or "enchantment" in text:
            perms = _opponent_noncreature_perms(state, controller_id, source_card)
            if perms:
                return [perms[0].instance_id]
        return []

    if kind == "bounce" or kind == "tap":
        creatures = _opponent_creatures(state, controller_id, source_card)
        if creatures:
            creatures.sort(key=_power, reverse=True)
            return [creatures[0].instance_id]
        return []

    if kind == "fight":
        # Need our own creature + opponent creature; pick our biggest.
        my_creatures = [
            c for c in state.cards
            if c.zone == Zone.BATTLEFIELD and c.controller_id == controller_id and c.is_creature()
        ]
        opp_creatures = _opponent_creatures(state, controller_id, source_card)
        if my_creatures and opp_creatures:
            mine = max(my_creatures, key=_power)
            theirs = max(opp_creatures, key=_power)
            return [mine.instance_id, theirs.instance_id]
        return []

    if kind == "damage":
        amount = _parse_amount(source_card.oracle_text, default=1)
        creatures = _opponent_creatures(state, controller_id, source_card)
        threats = [c for c in creatures if _power(c) >= 3]
        if threats and opp.life_total > amount * 2:
            threats.sort(key=_power, reverse=True)
            return [threats[0].instance_id]
        return [opp.player_id]

    if kind in ("draw", "ramp", "lifegain", "discard", "mill", "token", "scry", "surveil", "noop"):
        return []
    return []


# ---------------------------------------------------------------------------
# Effect application
# ---------------------------------------------------------------------------


def _parse_amount(text: str, default: int = 1) -> int:
    m = re.search(r"\b(\d+)\b", text or "")
    return int(m.group(1)) if m else default


def _find_card(state: GameState, instance_id: str) -> Optional[CardInstance]:
    return next((c for c in state.cards if c.instance_id == instance_id), None)


def _find_player(state: GameState, player_id: str) -> Optional[PlayerState]:
    return next((p for p in state.players if p.player_id == player_id), None)


def apply_spell_effect(
    state: GameState, stack_item: StackItem
) -> GameState:
    """Resolve the effect of ``stack_item`` and return the mutated state.

    Caller is responsible for removing the source card from the stack
    (e.g., moving it to graveyard) — this function only applies effects.
    """
    from .zones import move_card

    source = _find_card(state, stack_item.source_card_id) if stack_item.source_card_id else None
    oracle = (source.oracle_text if source else stack_item.card_data.get("oracle_text", "")) or ""
    name = source.name if source else stack_item.card_data.get("name", "Spell")
    controller_id = stack_item.controller_id
    targets = list(stack_item.targets or [])
    kind = detect_effect_kind(oracle)

    if kind == "counter":
        for tid in targets:
            target = next((it for it in state.stack if it.item_id == tid), None)
            if target and target is not stack_item:
                state.stack.remove(target)
                # Move the countered spell from STACK to its owner's graveyard.
                if target.source_card_id:
                    countered = _find_card(state, target.source_card_id)
                    if countered and countered.zone == Zone.STACK:
                        move_card(
                            state, countered.instance_id, Zone.STACK,
                            Zone.GRAVEYARD, countered.owner_id,
                        )
                state.log(f"{name} counters {target.card_data.get('name', 'spell')}")
        return state

    if kind == "damage":
        amount = _parse_amount(oracle, default=1)
        for tid in targets:
            tplayer = _find_player(state, tid)
            if tplayer is not None:
                tplayer.life_total -= amount
                state.log(f"{name} deals {amount} damage to {tplayer.name}")
                continue
            tcard = _find_card(state, tid)
            if tcard is not None and tcard.zone == Zone.BATTLEFIELD:
                tcard.damage_marked += amount
                state.log(f"{name} deals {amount} damage to {tcard.name}")
        if not targets:
            opp = _opponent(state, controller_id)
            if opp:
                opp.life_total -= amount
                state.log(f"{name} deals {amount} damage to {opp.name}")
        return state

    if kind == "destroy":
        for tid in targets:
            tcard = _find_card(state, tid)
            if tcard and tcard.zone == Zone.BATTLEFIELD:
                if "indestructible" in tcard.oracle_text.lower():
                    state.log(f"{name} cannot destroy indestructible {tcard.name}")
                    continue
                move_card(
                    state, tcard.instance_id, Zone.BATTLEFIELD,
                    Zone.GRAVEYARD, tcard.owner_id,
                )
                state.log(f"{name} destroys {tcard.name}")
        return state

    if kind == "bounce":
        for tid in targets:
            tcard = _find_card(state, tid)
            if tcard and tcard.zone == Zone.BATTLEFIELD:
                move_card(
                    state, tcard.instance_id, Zone.BATTLEFIELD,
                    Zone.HAND, tcard.owner_id,
                )
                state.log(f"{name} returns {tcard.name} to {tcard.owner_id}'s hand")
        return state

    if kind == "draw":
        controller = _find_player(state, controller_id)
        if controller:
            count = _parse_amount(oracle, default=1)
            library = [
                c for c in state.cards
                if c.zone == Zone.LIBRARY and c.owner_id == controller_id
            ]
            for _ in range(count):
                if not library:
                    break
                drawn = library.pop(0)
                move_card(state, drawn.instance_id, Zone.LIBRARY, Zone.HAND, controller_id)
            state.log(f"{name}: {controller.name} draws {count} card(s)")
        return state

    if kind == "ramp":
        # Search library for up to N basic lands, put onto battlefield tapped.
        count = 2 if "two" in oracle.lower() else _parse_amount(oracle, default=1)
        library = [
            c for c in state.cards
            if c.zone == Zone.LIBRARY and c.owner_id == controller_id and c.is_land()
        ]
        for c in library[:count]:
            move_card(state, c.instance_id, Zone.LIBRARY, Zone.BATTLEFIELD, controller_id)
            c.tapped = True
        state.log(f"{name}: {controller_id} ramps {min(count, len(library))} land(s)")
        return state

    if kind == "lifegain":
        amount = _parse_amount(oracle, default=1)
        controller = _find_player(state, controller_id)
        if controller:
            controller.life_total += amount
            state.log(f"{name}: {controller.name} gains {amount} life")
        return state

    if kind == "discard":
        opp = _opponent(state, controller_id)
        if opp:
            opp_hand = [
                c for c in state.cards
                if c.zone == Zone.HAND and c.owner_id == opp.player_id
            ]
            if opp_hand:
                opp_hand.sort(key=lambda c: c.cmc, reverse=True)
                victim = opp_hand[0]
                move_card(state, victim.instance_id, Zone.HAND, Zone.GRAVEYARD, opp.player_id)
                state.log(f"{name}: {opp.name} discards {victim.name}")
        return state

    # noop / unrecognised — just log
    if oracle.strip():
        state.log(f"{name} resolves (no effect implemented)")
    return state
