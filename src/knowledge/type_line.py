"""Parse Scryfall type lines against the Comprehensive-Rules type system.

Scryfall ships the type line as one string: ``"Legendary Artifact Creature
— Golem Soldier"``. Every consumer then re-derives structure from it by
substring matching, which is how ``"Vehicle"`` ends up recorded as a card
type and how ``"Artifact Land"`` gets mis-tagged.

This module does it once, properly, against the enumerations extracted
from the rules (:mod:`scripts.build_ontology_from_cr`). Because those
enumerations are closed, the parser can do something substring matching
cannot: resolve each subtype to the family it belongs to, and therefore
check the CR 205.3c/d correlation constraint --- *an object can't have a
subtype that doesn't correspond to one of its card types*.

That check is the payoff of having derived the type system from the rules
rather than hand-listing it: it turns a documented constraint into a test
that runs over the real card pool.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED_TTL = REPO_ROOT / "data" / "ontology" / "mtg-cr-types.ttl"

# Scryfall uses an em dash; a few community sources use a hyphen or en dash.
_DASH = re.compile(r"\s+[—–-]\s+")
# Split double-faced / split cards before parsing either half.
_FACE_SEP = re.compile(r"\s+//\s+")

# Which subtype family each card type licenses (CR 205.3g-q).
CARD_TYPE_TO_FAMILY: dict[str, str] = {
    "Artifact": "ArtifactType",
    "Enchantment": "EnchantmentType",
    "Land": "LandType",
    "Planeswalker": "PlaneswalkerType",
    "Instant": "SpellType",
    "Sorcery": "SpellType",
    "Creature": "CreatureType",
    "Kindred": "CreatureType",
    "Plane": "PlanarType",
    "Dungeon": "DungeonType",
    "Battle": "BattleType",
}


@dataclass(frozen=True)
class TypeSystem:
    """The closed vocabularies, as extracted from the rules."""

    card_types: frozenset[str]
    supertypes: frozenset[str]
    # family name -> members
    subtypes: dict[str, frozenset[str]]

    def family_of(self, subtype: str) -> set[str]:
        """Families a subtype belongs to. More than one is legal (CR reuses words)."""
        return {fam for fam, members in self.subtypes.items() if subtype in members}

    @property
    def all_subtypes(self) -> set[str]:
        out: set[str] = set()
        for members in self.subtypes.values():
            out |= members
        return out


@dataclass
class ParsedTypeLine:
    """Structured form of one card face's type line."""

    raw: str
    supertypes: list[str] = field(default_factory=list)
    card_types: list[str] = field(default_factory=list)
    subtypes: list[str] = field(default_factory=list)
    # Subtype -> the families it could belong to, restricted to those the
    # card's own types license. Empty set means a CR 205.3d violation.
    subtype_families: dict[str, set[str]] = field(default_factory=dict)
    unknown_tokens: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[str]:
        """Subtypes not licensed by any of this card's types (CR 205.3d)."""
        return [s for s, fams in self.subtype_families.items() if not fams]

    @property
    def is_well_formed(self) -> bool:
        return bool(self.card_types) and not self.violations and not self.unknown_tokens


# ---------------------------------------------------------------------------
# Loading the vocabularies
# ---------------------------------------------------------------------------

_FAMILIES = (
    "ArtifactType", "EnchantmentType", "LandType", "PlaneswalkerType",
    "SpellType", "CreatureType", "PlanarType", "DungeonType", "BattleType",
)


@functools.lru_cache(maxsize=1)
def load_type_system(path: Path | None = None) -> TypeSystem:
    """Read the generated TTL and index the type vocabularies by family.

    Parsed with a regex rather than rdflib so that importing this module
    stays cheap and dependency-free; the file is machine-generated and its
    shape is fixed by the generator.
    """
    ttl_path = path or DERIVED_TTL
    if not ttl_path.exists():
        raise FileNotFoundError(
            f"{ttl_path} not found. Run: python scripts/build_ontology_from_cr.py"
        )
    text = ttl_path.read_text(encoding="utf-8")

    by_class: dict[str, set[str]] = {}
    # Each individual is emitted as:  <prefix>:<Local> a mtg:<Class> ;
    #                                     rdfs:label "Name"@en ;
    pattern = re.compile(
        r"^\S+ a mtg:(?P<cls>\w+) ;\s*\n\s*rdfs:label \"(?P<label>[^\"]+)\"@en",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        by_class.setdefault(m.group("cls"), set()).add(m.group("label"))

    return TypeSystem(
        card_types=frozenset(by_class.get("CardType", set())),
        supertypes=frozenset(by_class.get("Supertype", set())),
        subtypes={f: frozenset(by_class.get(f, set())) for f in _FAMILIES},
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def split_faces(type_line: str) -> list[str]:
    """Split a double-faced or split card's type line into its faces."""
    return [f.strip() for f in _FACE_SEP.split(type_line or "") if f.strip()]


def parse_type_line(type_line: str, ts: TypeSystem | None = None) -> ParsedTypeLine:
    """Parse one face's type line into supertypes, card types and subtypes.

    Subtypes on the right of the dash are matched greedily against the
    known vocabulary so that two-word creature types (``Time Lord``) and
    multi-word planar types (``Bolas's Meditation Realm``) survive.
    """
    ts = ts or load_type_system()
    raw = (type_line or "").strip()
    parsed = ParsedTypeLine(raw=raw)
    if not raw:
        return parsed

    left, _, right = _partition(raw)

    for token in left.split():
        if token in ts.supertypes:
            parsed.supertypes.append(token)
        elif token in ts.card_types:
            parsed.card_types.append(token)
        else:
            parsed.unknown_tokens.append(token)

    if right:
        parsed.subtypes = _match_subtypes(right, ts, parsed)

    licensed = {
        CARD_TYPE_TO_FAMILY[t] for t in parsed.card_types if t in CARD_TYPE_TO_FAMILY
    }
    for sub in parsed.subtypes:
        parsed.subtype_families[sub] = ts.family_of(sub) & licensed

    return parsed


def _partition(raw: str) -> tuple[str, str, str]:
    parts = _DASH.split(raw, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), "—", parts[1].strip()
    return raw, "", ""


def _match_subtypes(right: str, ts: TypeSystem, parsed: ParsedTypeLine) -> list[str]:
    """Greedy longest-match over the subtype vocabulary.

    A plane's subtype is the whole string after the dash, however many
    words it is (CR 205.3b), so that case short-circuits.
    """
    if "Plane" in parsed.card_types:
        return [right]

    known = ts.all_subtypes
    words = right.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        for span in (2, 1):  # longest real subtype is two words ("Time Lord")
            if i + span > len(words):
                continue
            candidate = " ".join(words[i : i + span])
            if candidate in known:
                out.append(candidate)
                i += span
                break
        else:
            parsed.unknown_tokens.append(words[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# Batch report
# ---------------------------------------------------------------------------


@dataclass
class TypeLineReport:
    """Aggregate result of parsing a card pool."""

    total: int = 0
    well_formed: int = 0
    with_violations: int = 0
    with_unknown: int = 0
    violation_examples: list[tuple[str, str]] = field(default_factory=list)
    unknown_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        pct = (100.0 * self.well_formed / self.total) if self.total else 0.0
        lines = [
            f"cards parsed          {self.total}",
            f"well-formed           {self.well_formed} ({pct:.2f}%)",
            f"CR 205.3d violations  {self.with_violations}",
            f"unknown tokens        {self.with_unknown}",
        ]
        if self.violation_examples:
            lines.append("\nviolation examples:")
            lines += [f"  {n}: {d}" for n, d in self.violation_examples[:15]]
        if self.unknown_counts:
            top = sorted(self.unknown_counts.items(), key=lambda kv: -kv[1])[:15]
            lines.append("\nmost frequent unknown tokens:")
            lines += [f"  {tok:24s} {cnt}" for tok, cnt in top]
        return "\n".join(lines)


def report_over_cards(
    cards: list[dict], ts: TypeSystem | None = None
) -> TypeLineReport:
    """Parse a Scryfall card list and report type-system conformance."""
    ts = ts or load_type_system()
    report = TypeLineReport()
    for card in cards:
        type_line = card.get("type_line") or ""
        name = card.get("name", "?")
        for face in split_faces(type_line) or [""]:
            parsed = parse_type_line(face, ts)
            if not parsed.card_types:
                continue
            report.total += 1
            if parsed.violations:
                report.with_violations += 1
                if len(report.violation_examples) < 50:
                    report.violation_examples.append(
                        (name, f"{face} -> unlicensed {parsed.violations}")
                    )
            if parsed.unknown_tokens:
                report.with_unknown += 1
                for tok in parsed.unknown_tokens:
                    report.unknown_counts[tok] = report.unknown_counts.get(tok, 0) + 1
            if parsed.is_well_formed:
                report.well_formed += 1
    return report
