"""Canonical keyword detection and validity helpers (CR 702).

This module is the single source of truth for "does card X have keyword Y?"
and for queries like "is this permanent a legal target for `controller`?".

Keywords supported (per CR 702 evergreen + a handful of common deciduous):

Combat:
    flying, reach, menace, vigilance, haste, defender,
    first strike, double strike, trample, lifelink,
    deathtouch, indestructible

Targeting / interaction:
    hexproof, shroud, protection from <color>, ward {N}

Cast-time / triggered:
    flash, prowess

The detector is permissive: it scans ``oracle_text`` (case-insensitive) and
``card_data["keywords"]`` (Scryfall-style list). Keyword text inside reminder
parens still counts — Scryfall reminder text uses the same wording.
"""
from __future__ import annotations

import re
from typing import Optional

from .game_state import CardInstance, GameState, Zone


_EVERGREEN = (
    "flying", "reach", "menace", "vigilance", "haste", "defender",
    "first strike", "double strike", "trample", "lifelink",
    "deathtouch", "indestructible", "hexproof", "shroud",
    "flash", "prowess",
)


def has(card: Optional[CardInstance], keyword: str) -> bool:
    """Case-insensitive membership in oracle text or keywords list."""
    if card is None:
        return False
    kw = keyword.lower()
    text = (card.oracle_text or "").lower()
    if kw in text:
        return True
    raw = card.card_data.get("keywords") or []
    if isinstance(raw, str):
        raw = [raw]
    return any(kw == str(k).lower() for k in raw)


# Backwards-compat alias for older call sites.
def has_keyword(card: Optional[CardInstance], keyword: str) -> bool:
    return has(card, keyword)


# ---------------------------------------------------------------------------
# Protection
# ---------------------------------------------------------------------------


_COLOR_WORDS = ("white", "blue", "black", "red", "green")
_COLOR_LETTER = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G"}


def protections(card: Optional[CardInstance]) -> list[str]:
    """Color letters this permanent has protection from."""
    if card is None:
        return []
    text = (card.oracle_text or "").lower()
    out: list[str] = []
    for word in _COLOR_WORDS:
        if f"protection from {word}" in text:
            out.append(_COLOR_LETTER[word])
    return out


def _spell_colors(card: CardInstance) -> set[str]:
    raw = card.card_data.get("colors") or []
    if isinstance(raw, str):
        raw = [raw]
    out: set[str] = set()
    for c in raw:
        s = str(c).upper()
        if s in {"W", "U", "B", "R", "G"}:
            out.add(s)
    if not out:
        for letter in re.findall(r"\{([WUBRG])\}", card.mana_cost or ""):
            out.add(letter)
    return out


# ---------------------------------------------------------------------------
# Ward
# ---------------------------------------------------------------------------


def ward_cost(card: Optional[CardInstance]) -> int:
    """Generic ward cost in mana (CR 702.21), or 0 if absent.

    Recognised forms (oracle-text excerpts):

    * ``Ward {2}`` / ``Ward {1}{U}`` — total mana value.
    * ``Ward—Pay 3 life.`` — converted to 1 generic for the purpose of
      auto-pay (we don't yet model life payment as part of ward).
    * Bare ``Ward`` (no cost) — defaults to 1 (treated as ``ward {1}``).

    The match is anchored so that words like ``wardrobe`` or
    ``forward`` don't false-positive.
    """
    if card is None:
        return 0
    text = (card.oracle_text or "").lower()
    # Numeric mana ward like "ward {2}" or "ward {1}{U}".
    m = re.search(r"\bward\s*((?:\{[^}]+\}\s*)+)", text)
    if m:
        cost_chunk = m.group(1)
        total = 0
        for token in re.findall(r"\{([^}]+)\}", cost_chunk):
            tok = token.strip().upper()
            if tok.isdigit():
                total += int(tok)
            else:
                total += 1  # any coloured/hybrid pip counts as 1
        return total
    # Word-boundary check so "wardrobe" / "forward" don't match.
    if re.search(r"\bward\b", text):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Targeting validity
# ---------------------------------------------------------------------------


def can_be_targeted(
    target: CardInstance,
    source: Optional[CardInstance],
    source_controller_id: str,
) -> bool:
    """True if ``target`` is a legal target for a spell/ability ``source``
    that ``source_controller_id`` controls.

    Implements hexproof, shroud, and protection-from-color.
    """
    if target is None:
        return False
    is_opponent_source = target.controller_id != source_controller_id
    if has(target, "shroud"):
        return False
    if has(target, "hexproof") and is_opponent_source:
        return False
    if source is not None:
        protected = protections(target)
        if protected and (_spell_colors(source) & set(protected)):
            return False
    return True


# ---------------------------------------------------------------------------
# Block legality (used by combat.can_block)
# ---------------------------------------------------------------------------


def can_block_with(attacker: CardInstance, defender: CardInstance) -> bool:
    """Legality wrapper kept for backwards compatibility with older callers."""
    if defender is None or not defender.is_creature():
        return False
    if has(attacker, "flying") and not (has(defender, "flying") or has(defender, "reach")):
        return False
    return True


# ---------------------------------------------------------------------------
# Prowess (triggered when controller casts a non-creature spell)
# ---------------------------------------------------------------------------


def apply_prowess_on_cast(
    state: GameState, casting_player_id: str, cast_card: CardInstance
) -> None:
    """+1/+0 until end of turn for each prowess creature when caster casts a
    non-creature spell (CR 702.108). Tracked via a counter cleared in cleanup."""
    if cast_card.is_creature():
        return
    for c in state.cards:
        if c.zone != Zone.BATTLEFIELD or c.controller_id != casting_player_id:
            continue
        if not c.is_creature() or not has(c, "prowess"):
            continue
        c.counters["prowess_eot"] = c.counters.get("prowess_eot", 0) + 1
        state.log(f"{c.name}'s prowess triggers (+1/+0 until end of turn)")


def clear_eot_buffs(state: GameState) -> None:
    """Cleanup-step helper: clear 'until end of turn' counters."""
    for c in state.cards:
        if "prowess_eot" in c.counters:
            c.counters.pop("prowess_eot", None)


# ---------------------------------------------------------------------------
# Effective P/T (factors prowess and other +X/+X temp counters)
# ---------------------------------------------------------------------------


def effective_power(card: CardInstance) -> int:
    base = _parse_pt(card.power)
    plus = card.counters.get("+1/+1", 0)
    minus = card.counters.get("-1/-1", 0)
    eot = int(getattr(card, "eot_power_bonus", 0) or 0)
    equip = int(card.counters.get("equip_pwr", 0) or 0)
    return base + card.counters.get("prowess_eot", 0) + plus - minus + eot + equip


def effective_toughness(card: CardInstance) -> int:
    base = _parse_pt(card.toughness)
    plus = card.counters.get("+1/+1", 0)
    minus = card.counters.get("-1/-1", 0)
    eot = int(getattr(card, "eot_toughness_bonus", 0) or 0)
    equip = int(card.counters.get("equip_tou", 0) or 0)
    return base + plus - minus + eot + equip


def _parse_pt(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
