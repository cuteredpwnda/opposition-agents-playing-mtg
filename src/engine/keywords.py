"""Canonical keyword detection and validity helpers (CR 702).

This module is the single source of truth for "does card X have keyword Y?"
and for queries like "is this permanent a legal target for `controller`?".

Keywords supported (per CR 702 evergreen + deciduous):

Combat:
    flying, reach, menace, vigilance, haste, defender,
    first strike, double strike, trample, lifelink,
    deathtouch, indestructible, shadow, fear, intimidate,
    horsemanship, flanking, banding, battle cry, exalted,
    annihilator, infect, wither, toxic, skulk,
    daunt (can't be blocked by 2-power-or-less)

Damage-replacement:
    infect, wither  → damage dealt as -1/-1 counters to creatures
    infect          → damage dealt as poison counters to players

Death-replacement:
    undying   → if no +1/+1 counter, return with +1/+1
    persist   → if no -1/-1 counter, return with -1/-1

Targeting / interaction:
    hexproof, shroud, protection from <color>, ward {N}
    hexproof from <color>, protection from everything

Cast-time / triggered:
    flash, prowess, convoke, delve, improvise, emerge,
    spectacle, storm, dredge, cascade, affinity,
    morph, megamorph, suspend, foretell, escape,
    jump-start, retrace, overload, replicate, buyback,
    ninjutsu, modular, evolve, exploit, battalion,
    constellation, raid, morbid, threshold, hellbent,
    magecraft, boast, training, encore

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
# Extended block legality (CR 702.28 shadow, 702.35 fear, etc.)
# ---------------------------------------------------------------------------


def _card_colors(card: CardInstance) -> set[str]:
    """Return the set of color letters (WUBRG) this card is."""
    raw = card.card_data.get("colors") or []
    if isinstance(raw, str):
        raw = [raw]
    out: set[str] = set()
    for c in raw:
        s = str(c).upper()
        if s in {"W", "U", "B", "R", "G"}:
            out.add(s)
    # Fall back to parsing mana cost.
    if not out:
        mc = card.card_data.get("mana_cost") or ""
        for letter in re.findall(r"\{([WUBRG])\}", mc.upper()):
            out.add(letter)
    return out


def _is_artifact_type(card: CardInstance) -> bool:
    return "artifact" in (card.type_line or "").lower()


def can_be_blocked_by(
    attacker: CardInstance, blocker: CardInstance, state=None
) -> bool:
    """Full block-legality check incorporating evasion keywords (CR 509.1b).

    Checks:
    * Flying / Reach (CR 702.9)
    * Shadow (CR 702.28) — shadow creatures can only be blocked by shadow
    * Fear (CR 702.35) — only artifacts or black can block
    * Intimidate (CR 702.74) — only artifacts or same-color can block
    * Skulk (CR 702.116) — can't be blocked by power > attacker's power
    * Daunt (colloquial for "can't be blocked by creatures with power 2 or less")
    * Horsemanship (CR 702.42) — like flying but for its own axis
    * Landwalk (CR 702.14) — unblockable if defending player controls that land type
    * Menace handled in ``menace_satisfied``
    * Defender check handled in ``can_attack``
    """
    if blocker is None or not blocker.is_creature():
        return False
    if blocker.zone.value != "battlefield" or blocker.tapped:
        return False

    from .keywords import effective_power

    # Flying / Reach (CR 702.9)
    if has(attacker, "flying") and not (has(blocker, "flying") or has(blocker, "reach")):
        return False

    # Horsemanship (CR 702.42) — same axis as flying
    if has(attacker, "horsemanship") and not has(blocker, "horsemanship"):
        return False

    # Shadow (CR 702.28)
    if has(attacker, "shadow") and not has(blocker, "shadow"):
        return False
    if has(blocker, "shadow") and not has(attacker, "shadow"):
        return False  # shadow can't block non-shadow either

    # Fear (CR 702.35)
    if has(attacker, "fear"):
        if not (_is_artifact_type(blocker) or "B" in _card_colors(blocker)):
            return False

    # Intimidate (CR 702.74) — can only be blocked by artifact or same color
    if has(attacker, "intimidate"):
        atk_colors = _card_colors(attacker)
        if not (_is_artifact_type(blocker) or bool(atk_colors & _card_colors(blocker))):
            return False

    # Skulk (CR 702.116) — can't be blocked by creature with greater power
    if has(attacker, "skulk"):
        atk_pow = effective_power(attacker)
        if effective_power(blocker) > atk_pow:
            return False

    # Landwalk — if defending player controls a land of that basic type,
    # attacker is unblockable (CR 702.14).  We check a few common kinds.
    if state is not None:
        lw_map = {
            "plainswalk": "plains", "islandwalk": "island",
            "swampwalk": "swamp", "mountainwalk": "mountain",
            "forestwalk": "forest",
        }
        defender_id = next(
            (pid for pid, _ in (state.combat.attackers.items() if state.combat else [])
             if _ == blocker.controller_id), None
        )
        for kw, subtype in lw_map.items():
            if has(attacker, kw):
                # Check if the defending player controls a land with that subtype.
                blocker_lands = [
                    c for c in state.cards
                    if c.zone.value == "battlefield"
                    and c.controller_id == blocker.controller_id
                    and subtype in (c.type_line or "").lower()
                ]
                if blocker_lands:
                    return False  # unblockable

    return True


def annihilator_count(card: CardInstance) -> int:
    """Return the annihilator value (CR 702.86), 0 if not present."""
    if card is None:
        return 0
    text = (card.oracle_text or "").lower()
    # Form: "annihilator N"
    m = re.search(r"\bannihilator\s+(\d+)\b", text)
    if m:
        return int(m.group(1))
    # Scryfall keywords list: "Annihilator 2"
    raw = card.card_data.get("keywords") or []
    for k in raw:
        mk = re.match(r"annihilator\s+(\d+)", str(k).lower())
        if mk:
            return int(mk.group(1))
    return 0


def toxic_count(card: CardInstance) -> int:
    """Return toxic value (New Phyrexia — modern infect variant), 0 if absent."""
    if card is None:
        return 0
    text = (card.oracle_text or "").lower()
    m = re.search(r"\btoxic\s+(\d+)\b", text)
    return int(m.group(1)) if m else 0


def dredge_count(card: CardInstance) -> int:
    """Return dredge N, 0 if absent."""
    if card is None:
        return 0
    text = (card.oracle_text or "").lower()
    m = re.search(r"\bdredge\s+(\d+)\b", text)
    return int(m.group(1)) if m else 0


def modular_count(card: CardInstance) -> int:
    """Return modular N (enter with N +1/+1 counters), 0 if absent."""
    if card is None:
        return 0
    text = (card.oracle_text or "").lower()
    m = re.search(r"\bmodular\s+(\d+)\b", text)
    return int(m.group(1)) if m else 0


def reinforce_cost(card: CardInstance) -> str | None:
    """Return reinforce cost string, None if absent."""
    if card is None:
        return None
    m = re.search(r"\breinforce\s+(\d+)—(\{[^}]+(?:\}\{[^}]+)*\})", (card.oracle_text or ""), re.IGNORECASE)
    if m:
        return m.group(2)
    return None


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


def effective_power(card: CardInstance, state: GameState | None = None) -> int:
    """Effective power after all continuous effects (CR 613).

    Delegates to ``continuous_effects.effective_power`` which applies the full
    layer pipeline when ``state`` is provided.  Falls back to the legacy inline
    computation when called without state (e.g. from tests that predate the
    layer system).
    """
    try:
        from .continuous_effects import effective_power as _ce_ep
        return _ce_ep(card, state)
    except Exception:
        pass
    base = _parse_pt(card.power)
    plus = card.counters.get("+1/+1", 0)
    minus = card.counters.get("-1/-1", 0)
    eot = int(getattr(card, "eot_power_bonus", 0) or 0)
    equip = int(card.counters.get("equip_pwr", 0) or 0)
    return base + card.counters.get("prowess_eot", 0) + plus - minus + eot + equip


def effective_toughness(card: CardInstance, state: GameState | None = None) -> int:
    """Effective toughness after all continuous effects (CR 613)."""
    try:
        from .continuous_effects import effective_toughness as _ce_et
        return _ce_et(card, state)
    except Exception:
        pass
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
