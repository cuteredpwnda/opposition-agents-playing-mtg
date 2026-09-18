#!/usr/bin/env python
"""Check a card pool against the Comprehensive-Rules type system.

Parses every card's type line against the vocabularies extracted from the
rules and reports three things:

* **CR 205.3d violations** --- a subtype not licensed by any of the card's
  own card types. This is the constraint stated as an OWL axiom in
  ``mtg-ontology-v2.0.ttl``; running it over the real pool is what turns a
  documented rule into a measurement.
* **Unknown tokens** --- words in a type line that match no known card
  type, supertype or subtype. A rising count here is the signal that a new
  set shipped vocabulary the ontology has not picked up, i.e. that
  ``fetch_rules.py`` and ``build_ontology_from_cr.py`` need re-running.
* **Coverage** --- the fraction of the pool that parses cleanly.

No database required. Runs entirely offline against the generated TTL and
a Scryfall bulk file.

Usage::

    python scripts/check_card_types.py
    python scripts/check_card_types.py --cards data/scryfall/oracle-cards.json
    python scripts/check_card_types.py --fail-on-violation   # CI mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.knowledge.type_line import (  # noqa: E402
    load_type_system,
    parse_type_line,
    report_over_cards,
)

DEFAULT_CARDS = REPO_ROOT / "data" / "scryfall" / "oracle-cards.json"

# Hand-written probes that must parse correctly regardless of card data.
# These are the cases that motivated the type-system rebuild.
PROBES: list[tuple[str, str]] = [
    ("Smuggler's Copter", "Artifact — Vehicle"),
    ("Lightning Bolt", "Instant"),
    ("Dryad Arbor", "Land Creature — Forest Dryad"),
    ("Snow-Covered Mountain", "Basic Snow Land — Mountain"),
    ("Kellan, Planar Trailblazer", "Legendary Creature — Human Faerie"),
    ("Doctor Who", "Legendary Creature — Time Lord Doctor"),
    ("Saddled Mount", "Creature — Mount"),
    ("Undercity", "Dungeon — Undercity"),
    ("Invasion of Ravnica", "Battle — Siege"),
    ("Kumano Faces Kakkazan", "Enchantment — Saga"),
    ("Bala Ged Recovery", "Sorcery"),
    ("Agatha's Soul Cauldron", "Legendary Artifact"),
]


def run_probes() -> int:
    ts = load_type_system()
    print("Probe cases")
    print("-" * 72)
    failures = 0
    for name, line in PROBES:
        p = parse_type_line(line, ts)
        subs = ", ".join(
            f"{s} [{'/'.join(sorted(p.subtype_families[s])) or 'UNLICENSED'}]"
            for s in p.subtypes
        )
        status = "ok " if p.is_well_formed else "FAIL"
        if not p.is_well_formed:
            failures += 1
        print(f"  [{status}] {name}")
        print(f"         super={p.supertypes} types={p.card_types}")
        if subs:
            print(f"         subtypes={subs}")
        if p.unknown_tokens:
            print(f"         UNKNOWN={p.unknown_tokens}")
    print()
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--probes-only", action="store_true")
    ap.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit non-zero if any CR 205.3d violation or unknown token is found.",
    )
    args = ap.parse_args()

    try:
        ts = load_type_system()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(
        f"Type system: {len(ts.card_types)} card types, "
        f"{len(ts.supertypes)} supertypes, "
        f"{len(ts.all_subtypes)} subtypes across {len(ts.subtypes)} families\n"
    )

    probe_failures = run_probes()

    if args.probes_only:
        return 1 if (probe_failures and args.fail_on_violation) else 0

    if not args.cards.exists():
        print(
            f"card pool not found: {args.cards}\n"
            "Run: python -m scripts.import_scryfall  (downloads the bulk file)",
            file=sys.stderr,
        )
        return 0 if not args.fail_on_violation else 2

    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    report = report_over_cards(cards, ts)
    print("Card pool")
    print("-" * 72)
    print(report.summary())

    if args.fail_on_violation and (
        probe_failures or report.with_violations or report.with_unknown
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
