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
    # Permanent spells (creature, planeswalker, artifact, enchantment, land,
    # battle) don't choose targets when *cast* — they just enter the
    # battlefield. Their activated / triggered / loyalty abilities pick
    # targets later, when those abilities go on the stack. Bailing out here
    # prevents e.g. Grist from "targeting Mogg War Marshal" on cast just
    # because its −2 loyalty text contains "destroy target creature".
    type_line = (source_card.type_line or "").lower()
    permanent_types = ("creature", "planeswalker", "artifact",
                       "enchantment", "land", "battle")
    is_permanent = any(t in type_line for t in permanent_types)
    is_spell = "instant" in type_line or "sorcery" in type_line
    if is_permanent and not is_spell:
        return []

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


# Mode kinds we consider "valuable" for greedy modal pick-rank.
_MODE_KIND_RANK = {
    "destroy": 9, "exile": 9, "counter": 8, "bounce": 7, "edict": 7,
    "damage": 6, "drain_each": 6, "damage_each": 6,
    "token": 5, "ramp": 5, "tutor": 5, "draw": 5,
    "anthem_pump": 4, "pump": 3, "plus_counter": 4,
    "lifegain": 2, "scry": 2, "surveil": 2, "tap": 2, "untap": 2,
    "discard": 3, "mill": 2, "fight": 4, "proliferate": 3,
    "noop": 0,
}


def _split_modes(oracle: str) -> list[str]:
    """Split a modal spell into its individual mode clauses.

    Modes are separated by a bullet ``•`` (sometimes ``·`` or ``|``).
    The "Choose one — " preamble is stripped off the first mode.
    Trailing entwine / fuse clauses are skipped.
    """
    # Common bullet separators on Scryfall oracle text.
    parts = re.split(r"\s*[\u2022\u00b7|]\s*", oracle)
    if len(parts) < 2:
        return []
    # First part is the preamble before the first bullet — drop it.
    parts = [p.strip() for p in parts[1:] if p.strip()]
    # Strip rider clauses ("Entwine {2}", "Fuse", reminder text in parens).
    cleaned = []
    for p in parts:
        # Drop entwine/fuse riders that follow the last mode.
        if p.lower().startswith("entwine") or p.lower().startswith("fuse"):
            continue
        # Strip trailing reminder text in parentheses.
        p = re.sub(r"\(.*?\)", "", p).strip()
        if p:
            cleaned.append(p)
    return cleaned


def _pick_modes(
    state: GameState,
    source: CardInstance | None,
    controller_id: str,
    modes: list[str],
    pick_count: int,
) -> list[str]:
    """Rank modes by `_MODE_KIND_RANK` and return the top ``pick_count``.

    Filters out modes whose kind would have no legal effect (e.g. "destroy
    target creature" with no opposing creatures).
    """
    scored: list[tuple[int, str]] = []
    for m in modes:
        kind = detect_effect_kind(m)
        score = _MODE_KIND_RANK.get(kind, 0)
        # Penalise modes that need a target we don't have.
        if kind in ("destroy", "exile", "bounce", "tap", "fight"):
            opp_creatures = _opponent_creatures(state, controller_id, source)
            if not opp_creatures and "creature" in m.lower():
                score -= 5
        if kind == "counter":
            top = state.stack[-1] if state.stack else None
            if not top or not top.is_spell or top.controller_id == controller_id:
                score -= 5
        scored.append((score, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:max(1, pick_count)]]


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
    # Prefer the stack item's own oracle text when set (modal sub-items
    # carry only the chosen mode's clause). Falling back to source.oracle_text
    # would re-trigger the modal split and recurse infinitely.
    item_oracle = stack_item.card_data.get("oracle_text", "") if stack_item.card_data else ""
    oracle = item_oracle or (source.oracle_text if source else "") or ""
    name = source.name if source else stack_item.card_data.get("name", "Spell")
    controller_id = stack_item.controller_id
    targets = list(stack_item.targets or [])

    # Modal spells (CR 700.2): "Choose one — • A • B • C". Pick the mode(s)
    # the controller would prefer and run only those clauses. Without this,
    # the resolver sees the entire concatenated oracle and picks whichever
    # detect_effect_kind matches first — usually the wrong mode.
    oracle_lower_full = oracle.lower()
    if "choose one" in oracle_lower_full or "choose two" in oracle_lower_full \
            or "choose one or both" in oracle_lower_full:
        modes = _split_modes(oracle)
        if modes:
            n_modes = 2 if ("choose two" in oracle_lower_full
                            or "choose one or both" in oracle_lower_full) else 1
            # Entwine lets you choose all modes; we don't pay extra here, so
            # default to the standard pick count.
            picks = _pick_modes(state, source, controller_id, modes, n_modes)
            for mode_text in picks:
                state.log(f"    \u2022 {name} — mode: {mode_text.strip()}")
                sub_item = StackItem(
                    source_card_id=stack_item.source_card_id,
                    controller_id=controller_id,
                    is_spell=False,
                    card_data={
                        "name": name,
                        "oracle_text": mode_text,
                    },
                    targets=list(stack_item.targets or []),
                )
                # Pick fresh targets for the mode using its own text.
                if source is not None and not sub_item.targets:
                    original = source.card_data.get("oracle_text", "")
                    source.card_data["oracle_text"] = mode_text
                    try:
                        sub_item.targets = auto_pick_targets(state, source, controller_id)
                    finally:
                        source.card_data["oracle_text"] = original
                apply_spell_effect(state, sub_item)
            return state

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
        from . import tokens as _tok
        # "Create N X/Y <type> creature tokens" — minimal parser.
        m_count = re.search(r"create (\d+|a|an|two|three|four)", oracle.lower())
        word_to_int = {"a": 1, "an": 1, "two": 2, "three": 3, "four": 4}
        if m_count:
            raw = m_count.group(1)
            count = int(raw) if raw.isdigit() else word_to_int.get(raw, 1)
        else:
            count = 1
        # Predefined utility tokens.
        ol = oracle.lower()
        if "treasure" in ol:
            for _ in range(count):
                _tok.create_treasure_token(state, controller_id)
            state.log(f"{name}: creates {count} Treasure token(s)")
            return state
        if "food" in ol:
            for _ in range(count):
                _tok.create_food_token(state, controller_id)
            state.log(f"{name}: creates {count} Food token(s)")
            return state
        if "clue" in ol:
            for _ in range(count):
                _tok.create_clue_token(state, controller_id)
            state.log(f"{name}: creates {count} Clue token(s)")
            return state
        if "blood" in ol:
            for _ in range(count):
                _tok.create_blood_token(state, controller_id)
            state.log(f"{name}: creates {count} Blood token(s)")
            return state
        m_pt = re.search(r"(\d+)/(\d+)", oracle)
        tp, tt = (int(m_pt.group(1)), int(m_pt.group(2))) if m_pt else (1, 1)
        # Extract color words and subtype name from the X/Y ... token clause.
        m_clause = re.search(
            r"(\d+)/(\d+)\s+([\w\s,]*?)\s*(?:creature\s+)?token",
            oracle.lower(),
        )
        color_map = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G"}
        token_colors: list[str] = []
        token_subtype = "Spirit"
        if m_clause:
            middle = m_clause.group(3).strip()
            words = re.split(r"[\s,]+", middle)
            token_colors = [color_map[w] for w in words if w in color_map]
            # First non-color, non-noise word becomes the subtype.
            for w in words:
                if w in color_map or w in ("and", "or", "the", ""):
                    continue
                token_subtype = w.title()
                break
        for _ in range(count):
            _tok.create_creature_token(
                state, controller_id,
                power=tp, toughness=tt,
                subtypes=(token_subtype,),
                colors=token_colors or None,
            )
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
