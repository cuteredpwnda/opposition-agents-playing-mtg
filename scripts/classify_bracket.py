#!/usr/bin/env python
"""Classify a Commander decklist into a WotC Bracket.

Usage::

    python scripts/classify_bracket.py data/decks/my_edh_deck.txt
    python scripts/classify_bracket.py deck.txt --cedh   # force cEDH hint
    python scripts/classify_bracket.py a.txt b.txt c.txt # check pod compatibility
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.commander_brackets import (
    can_play_together,
    classify_bracket,
)
from src.integrations.decklist_loader import DecklistLoader
from src.integrations.offline_card_db import get_default_db


def report_one(path: Path, db, cedh: bool) -> None:
    text = path.read_text(encoding="utf-8")
    deck = DecklistLoader().from_text(text)
    rep = classify_bracket(deck, db, cedh_hint=cedh)
    print(f"\n=== {path.name} ===")
    print(f"  Commander: {', '.join(deck.commander) or '(none)'}")
    print(f"  Bracket {rep.bracket}: {rep.label}  (score={rep.score})")
    for n in rep.notes:
        print(f"    - {n}")
    if rep.violations:
        print("  ! Legality issues:")
        for v in rep.violations:
            print(f"    {v}")
    else:
        print("  Legality: OK")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("decklists", nargs="+", type=Path)
    p.add_argument("--cedh", action="store_true", help="Hint cEDH (Bracket 5)")
    args = p.parse_args()

    db = get_default_db()
    if len(db) == 0:
        print("[error] Local card cache is empty. "
              "Run: python scripts/fetch_card_data.py")
        return 2

    reports = []
    for path in args.decklists:
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        deck = DecklistLoader().from_text(text)
        rep = classify_bracket(deck, db, cedh_hint=args.cedh)
        reports.append(rep)
        report_one(path, db, args.cedh)

    if len(reports) > 1:
        print("\n=== Pod compatibility ===")
        ok = can_play_together(reports, tolerance=1)
        bs = sorted(r.bracket for r in reports)
        print(f"  Brackets: {bs}  -> {'OK (within ±1)' if ok else 'MISMATCHED — talk it out!'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
