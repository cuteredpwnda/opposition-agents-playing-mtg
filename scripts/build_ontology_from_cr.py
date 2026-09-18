#!/usr/bin/env python
"""Translate the Magic Comprehensive Rules into OWL, step by step.

The hand-authored ontology (``mtg-ontology-v2.0.ttl``) is the *schema*: it
says what kinds of things exist and how they relate. It is deliberately
small and human-reviewed. This script produces the *content*: the several
thousand individuals that the Comprehensive Rules actually enumerate, which
no one should be hand-copying.

Three stages, each independently runnable and independently inspectable:

**Stage 1 — rule tree.** Every numbered rule becomes an
``mtg:ComprehensiveRule`` individual carrying its number, text, section,
and a link to its parent rule. This makes "which rule licenses this?" a
graph query, and gives every generated term below something to point at.

**Stage 2 — enumerations.** The CR contains closed lists that are the
authoritative answer to "what card types exist", "what artifact subtypes
exist", and so on. Those rules are parsed into typed individuals:

===========  ==========================================  =================
CR rule      Enumeration                                 Emitted as
===========  ==========================================  =================
205.2a       card types (15)                             ``mtg:CardType``
205.3g       artifact types (incl. Vehicle, Spacecraft)  ``mtg:ArtifactType``
205.3h       enchantment types                           ``mtg:EnchantmentType``
205.3i       land types (incl. Planet)                   ``mtg:LandType``
205.3j       planeswalker types                          ``mtg:PlaneswalkerType``
205.3k       spell types                                 ``mtg:SpellType``
205.3m       creature types (incl. Mount)                ``mtg:CreatureType``
205.3n       planar types                                ``mtg:PlanarType``
205.3p       dungeon types                               ``mtg:DungeonType``
205.3q       battle types                                ``mtg:BattleType``
305.6        basic land types                            ``mtg:BasicLandType``
122.1b       keyword counters                            ``mtg:KeywordCounter``
701.x        keyword actions                             ``mtg:KeywordAction``
702.x        keyword abilities                           ``mtg:Keyword``
===========  ==========================================  =================

**Stage 3 — provenance.** Every generated individual gets
``mtg:groundedIn`` the rule it came from, ``mtg:epistemicStatus "curated"``
(the CR is definitional, not inferred), and the whole graph is attributed
to a ``prov:Activity`` recording the source document and its date.

Output is a *generated* file and is marked as such. Never hand-edit it;
change this script or the hand-authored schema instead.

Usage::

    python scripts/build_ontology_from_cr.py
    python scripts/build_ontology_from_cr.py --rules data/rules/latest.txt --stats
    python scripts/build_ontology_from_cr.py --skip-rule-tree   # enumerations only
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = REPO_ROOT / "data" / "rules" / "latest.txt"
ONTOLOGY_DIR = REPO_ROOT / "data" / "ontology"
# Enumerations are facts about the game and are redistributable.
DEFAULT_OUT_TYPES = ONTOLOGY_DIR / "mtg-cr-types.ttl"
# Rule *text* is the publisher's copyrighted expression: generated locally,
# git-ignored, never redistributed. See paper/mtg_ontology.tex, licensing.
DEFAULT_OUT_RULES = ONTOLOGY_DIR / "mtg-cr-rules.ttl"

MTG = "http://purl.org/mtg/ontology#"
SUBTYPE_NS = "http://purl.org/mtg/ontology/subtype#"
KEYWORD_NS = "http://purl.org/mtg/ontology/keyword#"
ACTION_NS = "http://purl.org/mtg/ontology/keyword-action#"

# Which namespace each generated family lives in. Card types stay in the core
# namespace because the hand-authored schema already declares them; everything
# else is namespaced so that, e.g., the keyword action "Exile" cannot collide
# with the zone ``mtg:Exile``.
FAMILY_PREFIX: dict[str, str] = {
    "CardType": "mtg",
    "Supertype": "mtg",
    "ArtifactType": "mtgs",
    "EnchantmentType": "mtgs",
    "LandType": "mtgs",
    "PlaneswalkerType": "mtgs",
    "SpellType": "mtgs",
    "CreatureType": "mtgs",
    "PlanarType": "mtgs",
    "DungeonType": "mtgs",
    "BattleType": "mtgs",
    "KeywordCounter": "mtgk",
    "Keyword": "mtgk",
    "KeywordAction": "mtgka",
}

# CR top-level sections. The numbering is stable across revisions.
SECTIONS: dict[int, str] = {
    1: "Game Concepts",
    2: "Parts of a Card",
    3: "Card Types",
    4: "Zones",
    5: "Turn Structure",
    6: "Spells, Abilities, and Effects",
    7: "Additional Rules",
    8: "Multiplayer Rules",
    9: "Casual Variants",
}

# A numbered rule line: "205.2a Some text" or "205.2. Card Types".
RULE_RE = re.compile(r"^(?P<num>\d{3}\.\d+[a-z]?)\.?\s+(?P<text>\S.*)$")
# Keyword ability / action headers: "702.9. Flying".
KEYWORD_HEADER_RE = re.compile(r"^(?P<num>70[12]\.\d+)\.\s+(?P<name>\S.*?)\s*$")


# ---------------------------------------------------------------------------
# Rule model
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    number: str
    text: str
    section: int
    parent: str | None = None

    @property
    def is_header(self) -> bool:
        """``205.2.`` style rules that title a block rather than state one."""
        return not self.number[-1].isalpha() and self.text.istitle()


@dataclass
class Enumeration:
    """One closed list lifted out of a rule."""

    rule: str
    owl_class: str
    members: list[str] = field(default_factory=list)
    comment: str = ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_rules(text: str) -> list[Rule]:
    """Extract the numbered-rule tree, stopping at the Glossary."""
    rules: list[Rule] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if line == "Glossary" and len(rules) > 100:
            break
        m = RULE_RE.match(line)
        if not m:
            continue
        num = m.group("num")
        # The table of contents repeats rule numbers before the body; keep
        # the first *substantive* occurrence only.
        if num in seen:
            continue
        body = m.group("text").strip()
        if len(body) < 3:
            continue
        seen.add(num)
        rules.append(Rule(number=num, text=body, section=int(num[0])))

    by_number = {r.number: r for r in rules}
    for r in rules:
        r.parent = _parent_of(r.number, by_number)
    return rules


def _parent_of(number: str, index: dict[str, Rule]) -> str | None:
    """``205.3m`` -> ``205.3``; ``205.3`` -> ``205.1`` block head if present."""
    if number[-1].isalpha():
        candidate = number[:-1]
        return candidate if candidate in index else None
    major = number.split(".")[0]
    candidate = f"{major}.1"
    return candidate if candidate != number and candidate in index else None


def _split_enumeration(sentence: str) -> list[str]:
    """Split ``"A, B (see rule 1), C, and D"`` into members.

    Handles the CR's three quirks: parenthetical cross-references, the
    Oxford ``and``/``or`` before the last item, and multi-word members
    (``Time Lord``, ``The Abyss``, ``Bolas's Meditation Realm``).
    """
    sentence = re.sub(r"\s*\([^)]*\)", "", sentence)  # drop "(see rule 301.5)"
    sentence = sentence.rstrip(". ")
    parts = [p.strip() for p in sentence.split(",")]
    out: list[str] = []
    for part in parts:
        part = re.sub(r"^(?:and|or)\s+", "", part).strip()
        if part:
            out.append(part)
    return [p for p in out if p]


def extract_enumeration(
    rules: dict[str, Rule], rule_no: str, owl_class: str, anchor: str
) -> Enumeration | None:
    """Pull the closed list out of a rule.

    ``anchor`` is the phrase the list follows, e.g. ``"The artifact types are"``.
    """
    rule = rules.get(rule_no)
    if rule is None:
        return None
    idx = rule.text.find(anchor)
    if idx < 0:
        return None
    tail = rule.text[idx + len(anchor):]
    # The list runs to the end of that sentence.
    stop = re.search(r"\.(?:\s|$)", tail)
    sentence = tail[: stop.start()] if stop else tail
    members = _split_enumeration(sentence)
    return Enumeration(rule=rule_no, owl_class=owl_class, members=members, comment=anchor)


def extract_keyword_headers(text: str, major: str) -> list[tuple[str, str]]:
    """Return ``(rule_number, name)`` for each ``70X.N. Name`` header."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        m = KEYWORD_HEADER_RE.match(line)
        if not m or not m.group("num").startswith(major):
            continue
        name = m.group("name")
        # Headers are short title-case names; rule bodies are sentences.
        if len(name) > 40 or name.endswith((".", ",", ":")):
            continue
        num = m.group("num")
        if num in seen:
            continue
        seen.add(num)
        out.append((num, name))
    return out


# ---------------------------------------------------------------------------
# IRI minting
# ---------------------------------------------------------------------------


def slug(name: str) -> str:
    """Mint a stable CamelCase local name from a rules term.

    ``Assembly-Worker`` -> ``AssemblyWorker``; ``Urza's`` -> ``Urzas``;
    ``Bolas's Meditation Realm`` -> ``BolassMeditationRealm``;
    ``Time Lord`` -> ``TimeLord``.
    """
    normalised = unicodedata.normalize("NFKD", name)
    normalised = normalised.replace("\u2019", "'").replace("\u2018", "'")
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    words = re.split(r"[^A-Za-z0-9]+", ascii_only)
    out = "".join(w[:1].upper() + w[1:] for w in words if w)
    if not out:
        out = "T" + hashlib.sha1(name.encode()).hexdigest()[:8]
    if out[0].isdigit():
        out = "N" + out
    return out


def rule_iri(number: str) -> str:
    return f"mtg:CR{number.replace('.', '_')}"


def ttl_string(value: str) -> str:
    """Escape a literal for a Turtle triple-quoted string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

ENUMERATION_SPECS: list[tuple[str, str, str]] = [
    ("205.2a", "CardType", "The card types are"),
    ("205.4a", "Supertype", "The supertypes are"),
    ("205.3g", "ArtifactType", "The artifact types are"),
    ("205.3h", "EnchantmentType", "The enchantment types are"),
    ("205.3i", "LandType", "The land types are"),
    ("205.3j", "PlaneswalkerType", "The planeswalker types are"),
    ("205.3k", "SpellType", "The spell types are"),
    ("205.3m", "CreatureType", "All other creature types are one word long:"),
    ("205.3n", "PlanarType", "The planar types are"),
    ("205.3p", "DungeonType", "That dungeon type is"),
    ("205.3q", "BattleType", "That battle type is"),
    ("122.1b", "KeywordCounter", "The keywords that a keyword counter can be are"),
]

# CR 205.3i names these inside the land-type rule rather than a separate one.
BASIC_LAND_TYPES = ("Forest", "Island", "Mountain", "Plains", "Swamp")

# CR 205.3m carries one two-word creature type outside the main list.
EXTRA_CREATURE_TYPES = ("Time Lord",)


def build(
    rules_text: str, *, include_rule_tree: bool = True
) -> tuple[str, str, dict[str, int]]:
    """Return ``(types_ttl, rules_ttl, stats)``.

    The two outputs are separated because they have different redistribution
    status: enumerations are facts, rule text is the publisher's expression.
    """
    rules = parse_rules(rules_text)
    index = {r.number: r for r in rules}
    stats: dict[str, int] = {"rules": len(rules)}
    # local-name -> families that minted it, for cross-family reuse reporting
    minted: dict[str, set[str]] = {}

    lines: list[str] = []
    emit = lines.append
    rule_lines: list[str] = []
    emit_rule = rule_lines.append

    source_date = _source_date(rules_text)
    emit(_header(source_date, kind="types"))

    # --- Stage 1: rule tree (separate output) ------------------------------
    if include_rule_tree:
        emit_rule(_header(source_date, kind="rules"))
        emit_rule("\n# " + "=" * 74)
        emit_rule("# Stage 1 — Comprehensive Rules tree")
        emit_rule("# " + "=" * 74 + "\n")
        for r in rules:
            emit_rule(f"{rule_iri(r.number)} a mtg:ComprehensiveRule ;")
            emit_rule(f'    mtg:ruleNumber "{r.number}" ;')
            emit_rule(f"    mtg:ruleSection {r.section} ;")
            emit_rule(f'    rdfs:label "CR {r.number}"@en ;')
            emit_rule(f'    mtg:ruleText """{ttl_string(r.text)}"""@en ;')
            if r.parent:
                emit_rule(f"    mtg:subRuleOf {rule_iri(r.parent)} ;")
            emit_rule('    mtg:epistemicStatus "curated" ;')
            emit_rule("    prov:wasDerivedFrom mtg:CRDocument .\n")

    # --- Stage 2: enumerations --------------------------------------------
    emit("\n# " + "=" * 74)
    emit("# Stage 2 — closed enumerations lifted from the rules")
    emit("# " + "=" * 74 + "\n")

    for rule_no, owl_class, anchor in ENUMERATION_SPECS:
        enum = extract_enumeration(index, rule_no, owl_class, anchor)
        if enum is None:
            print(f"  ! rule {rule_no} not found or anchor missing", file=sys.stderr)
            continue
        members = list(enum.members)
        if owl_class == "CreatureType":
            members.extend(EXTRA_CREATURE_TYPES)
        if owl_class in ("CardType", "Supertype"):
            # CR 205.2a/205.4a state these in lower case mid-sentence.
            members = [m[:1].upper() + m[1:] for m in members]
        stats[owl_class] = len(members)
        prefix = FAMILY_PREFIX[owl_class]
        emit(f"### {owl_class} — CR {rule_no} ({len(members)} individuals)\n")
        for name in members:
            local = slug(name)
            minted.setdefault(f"{prefix}:{local}", set()).add(owl_class)
            emit(f"{prefix}:{local} a mtg:{owl_class} ;")
            emit(f'    rdfs:label "{ttl_string(name)}"@en ;')
            emit(f'    skos:prefLabel "{ttl_string(name)}"@en ;')
            if owl_class == "LandType" and name in BASIC_LAND_TYPES:
                emit("    a mtg:BasicLandType ;")
            emit(f"    mtg:groundedIn {rule_iri(rule_no)} ;")
            emit('    mtg:epistemicStatus "curated" .')
        emit("")

    # --- keyword abilities and actions -------------------------------------
    for major, owl_class, label in (
        ("701", "KeywordAction", "Keyword actions — CR 701"),
        ("702", "Keyword", "Keyword abilities — CR 702"),
    ):
        headers = extract_keyword_headers(rules_text, major)
        stats[owl_class] = len(headers)
        prefix = FAMILY_PREFIX[owl_class]
        emit(f"### {label} ({len(headers)} individuals)\n")
        for num, name in headers:
            local = slug(name)
            minted.setdefault(f"{prefix}:{local}", set()).add(owl_class)
            emit(f"{prefix}:{local} a mtg:{owl_class} ;")
            emit(f'    rdfs:label "{ttl_string(name)}"@en ;')
            emit(f'    mtg:crRule "{num}" ;')
            emit(f"    mtg:groundedIn {rule_iri(num)} ;")
            emit('    mtg:epistemicStatus "curated" .')
        emit("")

    _report_reuse(minted)
    rules_ttl = "\n".join(rule_lines) + "\n" if rule_lines else ""
    return "\n".join(lines) + "\n", rules_ttl, stats


def _report_reuse(minted: dict[str, set[str]]) -> None:
    """Report terms the CR reuses across families.

    These are not errors. The rules genuinely reuse words: ``Spacecraft`` is
    both an artifact type and a planar type. Because such an individual ends
    up with two rdf:types, the subtype families must *not* be asserted
    pairwise disjoint in the schema.
    """
    shared = {k: v for k, v in minted.items() if len(v) > 1}
    if not shared:
        return
    print("\nCross-family term reuse (expected; blocks family disjointness):")
    for iri, families in sorted(shared.items()):
        print(f"  {iri:32s} {', '.join(sorted(families))}")


def _source_date(text: str) -> str:
    m = re.search(r"These rules are effective as of (\w+ \d+, \d{4})", text)
    return m.group(1) if m else "unknown"


def _header(source_date: str, kind: str = "types") -> str:
    if kind == "rules":
        title = "MTG Comprehensive Rules tree"
        iri = "cr-rules"
        note = (
            "# THIS FILE IS NOT REDISTRIBUTABLE. It carries the Comprehensive Rules\n"
            "# text, which is copyright Wizards of the Coast. It is generated locally\n"
            "# from your own copy of the rules and is git-ignored. The redistributable\n"
            "# part of the derivation is mtg-cr-types.ttl, which carries only the\n"
            "# enumerations -- facts about the game, not the publisher's expression."
        )
    else:
        title = "MTG type system and keyword vocabulary, derived from the rules"
        iri = "cr-types"
        note = (
            "# Enumerations only: card types, supertypes, subtypes, keyword abilities\n"
            "# and keyword actions. These are facts about the game. The rule *text*\n"
            "# lives in mtg-cr-rules.ttl, which is generated locally and not\n"
            "# redistributed."
        )

    return f"""@prefix mtg:     <{MTG}> .
@prefix mtgs:    <{SUBTYPE_NS}> .
@prefix mtgk:    <{KEYWORD_NS}> .
@prefix mtgka:   <{ACTION_NS}> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix skos:    <http://www.w3.org/2004/02/skos/core#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix prov:    <http://www.w3.org/ns/prov#> .

# =============================================================================
# GENERATED FILE — do not hand-edit.
#
# Produced by scripts/build_ontology_from_cr.py from the Magic Comprehensive
# Rules effective {source_date}. Change the script or the hand-authored schema
# (mtg-ontology-v2.0.ttl) instead; this file is regenerated wholesale.
#
# Everything here is `curated`: the Comprehensive Rules are definitional, so
# these individuals are ground truth rather than induced evidence.
#
# Namespaces are split per family so that CR terms cannot collide with the
# hand-authored schema — the keyword action "Exile" (CR 701) must not land on
# the same IRI as the zone mtg:Exile (CR 406).
#
{note}
# =============================================================================

<http://purl.org/mtg/ontology/{iri}> a owl:Ontology ;
    dcterms:title "{title}"@en ;
    dcterms:source "Magic: The Gathering Comprehensive Rules, effective {source_date}" ;
    dcterms:created "{date.today().isoformat()}"^^xsd:date ;
    owl:imports <http://purl.org/mtg/ontology/2.0> ;
    rdfs:comment "Generated. See scripts/build_ontology_from_cr.py."@en .

mtg:CRDocument a prov:Entity ;
    rdfs:label "Magic Comprehensive Rules"@en ;
    dcterms:issued "{source_date}" ;
    prov:wasAttributedTo mtg:WizardsOfTheCoast .

mtg:WizardsOfTheCoast a prov:Agent ;
    rdfs:label "Wizards of the Coast"@en .
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--out-types", type=Path, default=DEFAULT_OUT_TYPES)
    ap.add_argument("--out-rules", type=Path, default=DEFAULT_OUT_RULES)
    ap.add_argument(
        "--skip-rule-tree",
        action="store_true",
        help="Emit only the redistributable enumerations file.",
    )
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if not args.rules.exists():
        print(
            f"rules file not found: {args.rules}\nRun: python scripts/fetch_rules.py",
            file=sys.stderr,
        )
        return 2

    text = args.rules.read_text(encoding="utf-8", errors="replace")
    types_ttl, rules_ttl, stats = build(
        text, include_rule_tree=not args.skip_rule_tree
    )

    args.out_types.parent.mkdir(parents=True, exist_ok=True)
    args.out_types.write_text(types_ttl, encoding="utf-8")
    print(f"wrote {args.out_types} ({len(types_ttl.splitlines())} lines)")

    if rules_ttl:
        args.out_rules.write_text(rules_ttl, encoding="utf-8")
        print(
            f"wrote {args.out_rules} ({len(rules_ttl.splitlines())} lines) "
            "— git-ignored, carries copyrighted rule text"
        )

    print("\nExtracted:")
    for key, value in stats.items():
        print(f"  {key:20s} {value:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
