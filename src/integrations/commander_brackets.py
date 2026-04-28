"""Commander Bracket system (WotC, 2025).

Classifies a 100-card singleton decklist into one of five brackets and
also performs basic Commander deck legality checks (singleton, deck
size, color identity, banned list, partner pairings).

Brackets:
    1 — Exhibition: ultra-casual / theme decks
    2 — Core: average precon-level
    3 — Upgraded: tuned precon, some Game Changers allowed
    4 — Optimized: high-power but not necessarily cEDH
    5 — cEDH: competitive, fastest-possible wins, no restrictions

Reference: https://magic.wizards.com/en/news/announcements/introducing-commander-brackets-beta

The reference lists (Game Changers, Mass Land Destruction, Extra Turns,
Tutors) live as plain text under ``data/commander/`` so they can be
edited as the official lists change.

Typical use::

    deck = DecklistLoader().from_text(text)
    db = get_default_db()
    report = classify_bracket(deck, db)
    print(f"Bracket {report.bracket}: {report.label}")
    for note in report.notes:
        print(f"  - {note}")
    if report.violations:
        print("  ! Illegal:")
        for v in report.violations:
            print(f"    {v}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from src.integrations.decklist_loader import Decklist
from src.integrations.offline_card_db import OfflineCardDB

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/commander")


# ---------------------------------------------------------------------------
# Reference list loaders (cached at module load)
# ---------------------------------------------------------------------------


def _load_list(name: str) -> set[str]:
    """Load a card-name list from ``data/commander/<name>.txt`` (lowercased)."""
    path = DATA_DIR / f"{name}.txt"
    if not path.exists():
        logger.warning("Commander list missing: %s", path)
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s.lower())
    return out


GAME_CHANGERS = _load_list("game_changers")
MASS_LAND_DESTRUCTION = _load_list("mass_land_destruction")
EXTRA_TURNS = _load_list("extra_turns")
TUTORS = _load_list("tutors")


# Format-level Commander banned list (subset; full list at
# https://mtgcommander.net/index.php/banned-list/). We only enforce
# the most-cited entries — a full check belongs in a maintained data file.
COMMANDER_BANNED = {
    "ancestral recall", "balance", "biorhythm", "black lotus",
    "channel", "emrakul, the aeons torn", "erayo, soratami ascendant",
    "fastbond", "flash", "gifts ungiven", "griselbrand", "hullbreacher",
    "iona, shield of emeria", "karakas", "leovold, emissary of trest",
    "library of alexandria", "limited resources", "lutri, the spellchaser",
    "mox jet", "mox pearl", "mox ruby", "mox sapphire", "mox emerald",
    "panoptic mirror", "paradox engine", "primeval titan", "prophet of kruphix",
    "recurring nightmare", "rofellos, llanowar emissary", "sundering titan",
    "sway of the stars", "sylvan primordial", "time vault", "time walk",
    "tinker", "tolarian academy", "trade secrets", "upheaval",
    "worldfire", "yawgmoth's bargain", "nadu, winged wisdom",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BracketReport:
    """Outcome of bracket classification."""

    bracket: int                      # 1..5
    label: str
    score: int                        # raw classifier score
    game_changers: list[str] = field(default_factory=list)
    mld: list[str] = field(default_factory=list)
    extra_turns: list[str] = field(default_factory=list)
    tutors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)  # legality issues

    @property
    def is_legal(self) -> bool:
        return not self.violations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat(deck: Decklist) -> Iterable[tuple[str, int]]:
    """Iterate (name, count) over the whole 100-card list."""
    seen: dict[str, int] = {}
    for name, count in deck.mainboard.items():
        seen[name] = seen.get(name, 0) + count
    for name in deck.commander:
        seen[name] = seen.get(name, 0) + 1
    return seen.items()


def _is_basic_land(card: dict) -> bool:
    type_line = (card.get("type_line") or "").lower()
    return "basic" in type_line and "land" in type_line


def _is_snow_land(card: dict) -> bool:
    type_line = (card.get("type_line") or "").lower()
    return "snow" in type_line and "land" in type_line


def _color_identity(card: dict) -> set[str]:
    return set(card.get("color_identity", []) or [])


def _has_partner(card: dict) -> bool:
    text = (card.get("oracle_text") or "").lower()
    return "partner" in text or "choose a background" in text


# ---------------------------------------------------------------------------
# Legality check
# ---------------------------------------------------------------------------


def check_legality(deck: Decklist, db: OfflineCardDB) -> list[str]:
    """Return a list of legality violations (empty if legal)."""
    violations: list[str] = []

    total = sum(c for _, c in _flat(deck))
    if total != 100:
        violations.append(f"deck has {total} cards (Commander requires exactly 100)")

    # Singleton: every non-basic, non-snow card must appear exactly once.
    for name, count in _flat(deck):
        if count <= 1:
            continue
        card = db.get(name)
        if card is None:
            violations.append(f"'{name}' appears {count} times but card not found in cache")
            continue
        if not (_is_basic_land(card) or _is_snow_land(card)):
            text = (card.get("oracle_text") or "").lower()
            if "a deck can have any number of cards named" not in text:
                violations.append(f"singleton violation: {count}x '{name}'")

    # Commander identity
    if not deck.commander:
        violations.append("no commander declared")
    elif len(deck.commander) > 2:
        violations.append(f"too many commanders: {len(deck.commander)}")
    else:
        cmds = [db.get(n) for n in deck.commander]
        cmds = [c for c in cmds if c is not None]
        if len(deck.commander) == 2:
            if not all(_has_partner(c) for c in cmds):
                violations.append("two commanders but at least one lacks Partner / Background")
        for cmd in cmds:
            type_line = (cmd.get("type_line") or "").lower()
            text = (cmd.get("oracle_text") or "").lower()
            is_eligible = (
                ("legendary" in type_line and "creature" in type_line)
                or "can be your commander" in text
                or "choose a background" in type_line
            )
            if not is_eligible:
                violations.append(f"'{cmd.get('name')}' is not a legal commander")

        # Color-identity check on the rest of the deck
        cmd_ci: set[str] = set()
        for c in cmds:
            cmd_ci |= _color_identity(c)
        for name, _ in _flat(deck):
            if name in deck.commander:
                continue
            card = db.get(name)
            if card is None:
                continue
            if not _color_identity(card).issubset(cmd_ci):
                violations.append(
                    f"'{name}' violates color identity {sorted(cmd_ci) or ['C']}"
                )

    # Banned list
    for name, _ in _flat(deck):
        if name.lower() in COMMANDER_BANNED:
            violations.append(f"'{name}' is on the Commander banned list")

    return violations


# ---------------------------------------------------------------------------
# Bracket classification
# ---------------------------------------------------------------------------


# WotC public guidance:
#   B1: 0 GC, 0 MLD, 0 extra-turn, no 2-card infinites, no early game-end
#   B2: 0 GC, 0 MLD, very limited tutors, "precon-level"
#   B3: <= 3 GC, no early MLD, no 2-card infinites
#   B4: optimized — anything legal goes (still no banlist)
#   B5: cEDH — fastest-possible competitive

BRACKET_LABELS = {
    1: "Exhibition (ultra-casual)",
    2: "Core (precon-level)",
    3: "Upgraded",
    4: "Optimized",
    5: "cEDH (competitive)",
}


def classify_bracket(
    deck: Decklist,
    db: OfflineCardDB,
    cedh_hint: Optional[bool] = None,
) -> BracketReport:
    """Classify a deck into a Commander Bracket and report findings.

    Args:
        deck: parsed decklist.
        db: offline Scryfall DB for card-data lookup.
        cedh_hint: if True, force at least Bracket 5 (user-declared cEDH).
    """
    notes: list[str] = []
    gc_hits: list[str] = []
    mld_hits: list[str] = []
    et_hits: list[str] = []
    tutor_hits: list[str] = []

    for name, count in _flat(deck):
        ln = name.lower()
        if ln in GAME_CHANGERS:
            gc_hits.extend([name] * count)
        if ln in MASS_LAND_DESTRUCTION:
            mld_hits.extend([name] * count)
        if ln in EXTRA_TURNS:
            et_hits.extend([name] * count)
        if ln in TUTORS:
            tutor_hits.extend([name] * count)

    # Heuristic score → bracket.
    # Each contributes additively; thresholds map to brackets.
    score = 0
    score += len(gc_hits) * 3
    score += len(mld_hits) * 4
    score += len(et_hits) * 2
    score += max(0, len(tutor_hits) - 2) * 2  # first 2 tutors are "free"

    if score == 0 and len(tutor_hits) <= 1:
        bracket = 1
    elif score <= 2 and len(gc_hits) == 0 and len(mld_hits) == 0:
        bracket = 2
    elif len(gc_hits) <= 3 and len(mld_hits) == 0 and score < 12:
        bracket = 3
    elif score < 24:
        bracket = 4
    else:
        bracket = 5

    if cedh_hint:
        bracket = max(bracket, 5)
        notes.append("cEDH hint set by caller — forcing Bracket 5")

    # Notes
    if gc_hits:
        notes.append(f"{len(gc_hits)} Game Changer(s): {', '.join(sorted(set(gc_hits)))}")
    if mld_hits:
        notes.append(f"{len(mld_hits)} mass land destruction card(s): "
                     f"{', '.join(sorted(set(mld_hits)))}")
    if et_hits:
        notes.append(f"{len(et_hits)} extra-turn card(s): "
                     f"{', '.join(sorted(set(et_hits)))}")
    if tutor_hits:
        notes.append(f"{len(tutor_hits)} unconditional tutor(s)")
    if not notes:
        notes.append("No bracket-affecting cards detected — clean casual list.")

    violations = check_legality(deck, db)

    return BracketReport(
        bracket=bracket,
        label=BRACKET_LABELS[bracket],
        score=score,
        game_changers=sorted(set(gc_hits)),
        mld=sorted(set(mld_hits)),
        extra_turns=sorted(set(et_hits)),
        tutors=sorted(set(tutor_hits)),
        notes=notes,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Bracket-aware matching
# ---------------------------------------------------------------------------


def can_play_together(reports: list[BracketReport], tolerance: int = 1) -> bool:
    """Return True iff all reports fall within ``tolerance`` brackets of each other.

    The official WotC guidance is that all players in a pod should agree on
    the bracket beforehand; in practice ±1 is acceptable.
    """
    if not reports:
        return True
    bs = [r.bracket for r in reports]
    return max(bs) - min(bs) <= tolerance
