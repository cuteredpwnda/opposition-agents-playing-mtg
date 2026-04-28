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
    if "proliferate" in text:
        return "proliferate"
    if re.search(r"put\s+(?:a|one|two|three|\d+)\s+\+1/\+1\s+counter", text):
        return "plus_counter"
    if "exile target" in text:
        return "exile"
    if "destroy target" in text:
        return "destroy"
    if "fight" in text and "target" in text:
        return "fight"
    if (re.search(r"return target .* to (its|their) owner['\u2019]s hand", text)
            or ("return target" in text and "hand" in text)):
        return "bounce"
    if "untap target" in text or re.search(r"untap (all|each)", text):
        return "untap"
    if "tap target" in text:
        return "tap"
    if "each opponent loses" in text and "life" in text:
        return "drain_each"
    if "each opponent" in text and "damage" in text:
        return "damage_each"
    if "deal" in text and "damage" in text:
        return "damage"
    if "sacrifice" in text and "target" not in text:
        # Self-sac as cost is handled at cast time; here "each player
        # sacrifices" is the on-resolve case.
        if "each" in text or "all" in text:
            return "edict"
    if "target" in text and "sacrifice" in text:
        return "edict"
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
    if "search your library" in text:
        return "tutor"
    if "discard" in text:
        return "discard"
    if "gain" in text and "life" in text:
        return "lifegain"
    if re.search(r"creatures? you control get \+\d+/\+\d+", text):
        return "anthem_pump"
    if re.search(r"target creature gets \+\d+/\+\d+", text):
        return "pump"
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
                if has_kw(tcard, "indestructible"):
                    state.log(f"{name} cannot destroy indestructible {tcard.name}")
                    continue
                move_card(
                    state, tcard.instance_id, Zone.BATTLEFIELD,
                    Zone.GRAVEYARD, tcard.owner_id,
                )
                state.log(f"{name} destroys {tcard.name}")
        return state

    if kind == "exile":
        for tid in targets:
            tcard = _find_card(state, tid)
            if tcard and tcard.zone == Zone.BATTLEFIELD:
                # Exile bypasses indestructible (CR 701.18).
                move_card(
                    state, tcard.instance_id, Zone.BATTLEFIELD,
                    Zone.EXILE, tcard.owner_id,
                )
                state.log(f"{name} exiles {tcard.name}")
        return state

    if kind == "tap":
        for tid in targets:
            tcard = _find_card(state, tid)
            if tcard and tcard.zone == Zone.BATTLEFIELD:
                tcard.tapped = True
                state.log(f"{name} taps {tcard.name}")
        return state

    if kind == "fight":
        if len(targets) >= 2:
            a = _find_card(state, targets[0])
            b = _find_card(state, targets[1])
            if a and b and a.zone == Zone.BATTLEFIELD and b.zone == Zone.BATTLEFIELD:
                ap, bp = _power(a), _power(b)
                a.damage_marked += bp
                b.damage_marked += ap
                state.log(f"{name}: {a.name} ({ap}) fights {b.name} ({bp})")
        return state

    if kind == "mill":
        # Default: mill 1 from opponent.
        opp_player = _opponent(state, controller_id)
        amount = _parse_amount(oracle, default=1)
        if opp_player:
            library = [
                c for c in state.cards
                if c.zone == Zone.LIBRARY and c.owner_id == opp_player.player_id
            ]
            for c in library[:amount]:
                move_card(state, c.instance_id, Zone.LIBRARY, Zone.GRAVEYARD, opp_player.player_id)
            state.log(f"{name}: {opp_player.name} mills {min(amount, len(library))}")
        return state

    if kind == "token":
        # "Create N X/Y <type> creature tokens" — minimal parser.
        m_count = re.search(r"create (\d+|a|an|two|three|four)", oracle.lower())
        word_to_int = {"a": 1, "an": 1, "two": 2, "three": 3, "four": 4}
        if m_count:
            raw = m_count.group(1)
            count = int(raw) if raw.isdigit() else word_to_int.get(raw, 1)
        else:
            count = 1
        m_pt = re.search(r"(\d+)/(\d+)", oracle)
        if m_pt:
            tp, tt = m_pt.group(1), m_pt.group(2)
        else:
            tp, tt = "1", "1"
        # Try to identify the token's creature type ("Soldier", "Goblin", ...).
        m_type = re.search(r"(\d+)/(\d+)\s+(?:white|blue|black|red|green|colorless)?\s*([A-Z][a-z]+)", source.oracle_text if source else "")
        token_subtype = m_type.group(3) if m_type else "Spirit"
        for _ in range(count):
            tok = CardInstance(
                card_data={
                    "name": f"{token_subtype} Token",
                    "type_line": f"Token Creature \u2014 {token_subtype}",
                    "mana_cost": "",
                    "cmc": 0,
                    "oracle_text": "",
                    "power": tp,
                    "toughness": tt,
                    "is_token": True,
                },
                zone=Zone.BATTLEFIELD,
                owner_id=controller_id,
                controller_id=controller_id,
                summoning_sick=True,
                turn_entered=state.turn_number,
            )
            state.cards.append(tok)
        state.log(f"{name}: creates {count} {tp}/{tt} {token_subtype} token(s)")
        return state

    if kind == "scry":
        amount = _parse_amount(oracle, default=1)
        # Greedy: keep cards with cmc <= turn_number, send the rest to bottom.
        controller = _find_player(state, controller_id)
        if controller:
            library = [
                c for c in state.cards
                if c.zone == Zone.LIBRARY and c.owner_id == controller_id
            ]
            top = library[:amount]
            keep_top: list[CardInstance] = []
            send_bottom: list[CardInstance] = []
            for c in top:
                if c.is_land() and controller.land_plays_remaining > 0:
                    keep_top.append(c)
                elif c.cmc <= max(1, state.turn_number):
                    keep_top.append(c)
                else:
                    send_bottom.append(c)
            # Reorder library: keep_top, then unchanged middle, then send_bottom.
            middle = library[amount:]
            new_order = keep_top + middle + send_bottom
            # Rebuild library zone in order: pop existing, push in new order.
            state.cards = [c for c in state.cards if c not in library] + new_order
            state.log(f"{name}: scries {amount} (keep {len(keep_top)}, bottom {len(send_bottom)})")
        return state

    if kind == "surveil":
        amount = _parse_amount(oracle, default=1)
        controller = _find_player(state, controller_id)
        if controller:
            library = [
                c for c in state.cards
                if c.zone == Zone.LIBRARY and c.owner_id == controller_id
            ]
            top = library[:amount]
            for c in top:
                # Heuristic: graveyard cards we don't want; lands stay on top.
                if c.is_land() and controller.land_plays_remaining > 0:
                    continue
                # Send first non-land to graveyard, rest stay.
                move_card(state, c.instance_id, Zone.LIBRARY, Zone.GRAVEYARD, controller_id)
                state.log(f"{name}: surveil sends {c.name} to graveyard")
                break
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
        from .zones import shuffle_library
        count = 2 if "two" in oracle.lower() else _parse_amount(oracle, default=1)
        library = [
            c for c in state.cards
            if c.zone == Zone.LIBRARY and c.owner_id == controller_id and c.is_land()
        ]
        for c in library[:count]:
            move_card(state, c.instance_id, Zone.LIBRARY, Zone.BATTLEFIELD, controller_id)
            c.tapped = True
        # CR 701.20: searching a library shuffles it.
        shuffle_library(state, controller_id)
        state.log(f"{name}: {controller_id} ramps {min(count, len(library))} land(s)")
        return state

    if kind == "tutor":
        # Generic tutor: pull the highest-CMC non-land card to hand.
        from .zones import shuffle_library
        library = [
            c for c in state.cards
            if c.zone == Zone.LIBRARY and c.owner_id == controller_id and not c.is_land()
        ]
        if library:
            library.sort(key=lambda c: c.cmc or 0, reverse=True)
            picked = library[0]
            move_card(state, picked.instance_id, Zone.LIBRARY, Zone.HAND, controller_id)
            state.log(f"{name}: tutors up {picked.name}")
        shuffle_library(state, controller_id)
        return state

    if kind == "plus_counter":
        amt_match = re.search(
            r"put\s+(a|one|two|three|four|\d+)\s+\+1/\+1\s+counter", oracle.lower()
        )
        word = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4}
        amount = 1
        if amt_match:
            raw = amt_match.group(1)
            amount = int(raw) if raw.isdigit() else word.get(raw, 1)
        # Default target: our biggest creature; or the explicit target.
        creatures: list[CardInstance] = []
        if targets:
            for tid in targets:
                tc = _find_card(state, tid)
                if tc and tc.zone == Zone.BATTLEFIELD and tc.is_creature():
                    creatures.append(tc)
        else:
            mine = [
                c for c in state.cards
                if c.zone == Zone.BATTLEFIELD
                and c.controller_id == controller_id and c.is_creature()
            ]
            if mine:
                creatures.append(max(mine, key=_power))
        for tc in creatures:
            tc.counters["+1/+1"] = tc.counters.get("+1/+1", 0) + amount
            state.log(f"{name}: puts {amount} +1/+1 counter(s) on {tc.name}")
        return state

    if kind == "proliferate":
        # CR 701.27: choose any number of permanents/players with counters,
        # add one of each kind they already have. Greedy: do all of ours.
        bumped = 0
        for c in state.cards:
            if c.zone != Zone.BATTLEFIELD or c.controller_id != controller_id:
                continue
            for kctype in list(c.counters.keys()):
                if c.counters[kctype] > 0:
                    c.counters[kctype] += 1
                    bumped += 1
        state.log(f"{name}: proliferates ({bumped} counter(s) added)")
        return state

    if kind == "untap":
        if targets:
            for tid in targets:
                tc = _find_card(state, tid)
                if tc and tc.zone == Zone.BATTLEFIELD:
                    tc.tapped = False
                    state.log(f"{name} untaps {tc.name}")
        else:
            # "Untap all permanents you control"
            n = 0
            for c in state.cards:
                if (c.zone == Zone.BATTLEFIELD
                        and c.controller_id == controller_id and c.tapped):
                    c.tapped = False
                    n += 1
            state.log(f"{name} untaps {n} permanent(s)")
        return state

    if kind == "edict":
        # "Each player sacrifices a creature" — keep it simple: each player
        # sacrifices their lowest-power creature.
        for p in state.players:
            mine = [
                c for c in state.cards
                if c.zone == Zone.BATTLEFIELD
                and c.controller_id == p.player_id and c.is_creature()
            ]
            if not mine:
                continue
            victim = min(mine, key=_power)
            move_card(
                state, victim.instance_id, Zone.BATTLEFIELD,
                Zone.GRAVEYARD, victim.owner_id,
            )
            state.log(f"{name}: {p.name} sacrifices {victim.name}")
        return state

    if kind == "damage_each":
        amount = _parse_amount(oracle, default=1)
        for p in state.players:
            if p.player_id == controller_id:
                continue
            p.life_total -= amount
            state.log(f"{name} deals {amount} damage to {p.name}")
        return state

    if kind == "drain_each":
        amount = _parse_amount(oracle, default=1)
        gained = 0
        for p in state.players:
            if p.player_id == controller_id:
                continue
            p.life_total -= amount
            gained += amount
            state.log(f"{name}: {p.name} loses {amount} life")
        # "and you gain that much life" — common rider; only apply if present.
        if "you gain" in oracle.lower():
            you = _find_player(state, controller_id)
            if you:
                you.life_total += gained
                state.log(f"{name}: {you.name} gains {gained} life")
        return state

    if kind == "anthem_pump":
        m = re.search(r"\+(\d+)/\+(\d+)", oracle)
        if m:
            dp, dt = int(m.group(1)), int(m.group(2))
            n = 0
            for c in state.cards:
                if (c.zone == Zone.BATTLEFIELD
                        and c.controller_id == controller_id and c.is_creature()):
                    # Until end of turn: stash on the card; cleanup clears later.
                    c.eot_power_bonus = c.__dict__.get("eot_power_bonus", 0) + dp
                    c.eot_toughness_bonus = c.__dict__.get("eot_toughness_bonus", 0) + dt
                    n += 1
            state.log(f"{name}: pumps {n} creature(s) +{dp}/+{dt} EOT")
        return state

    if kind == "pump":
        m = re.search(r"\+(\d+)/\+(\d+)", oracle)
        if m and targets:
            dp, dt = int(m.group(1)), int(m.group(2))
            for tid in targets:
                tc = _find_card(state, tid)
                if tc and tc.zone == Zone.BATTLEFIELD:
                    tc.eot_power_bonus = tc.__dict__.get("eot_power_bonus", 0) + dp
                    tc.eot_toughness_bonus = tc.__dict__.get("eot_toughness_bonus", 0) + dt
                    state.log(f"{name}: {tc.name} gets +{dp}/+{dt} EOT")
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
        state.log(f"    \u25c0 {name} resolves (no effect implemented)")
    return state
