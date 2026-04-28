#!/usr/bin/env python
"""Fetch a Commander decklist from Moxfield / EDHREC / Archidekt and cache it.

Usage::

    # Moxfield (public deck URL or just the public ID)
    python scripts/fetch_decklist.py https://www.moxfield.com/decks/abc123

    # EDHREC "average deck" for a commander
    python scripts/fetch_decklist.py https://edhrec.com/commanders/krenko-mob-boss

    # Archidekt
    python scripts/fetch_decklist.py https://archidekt.com/decks/1234567

    # Save to a specific file and immediately classify into a bracket
    python scripts/fetch_decklist.py <url> --out data/decks/my_deck.txt --classify

The fetched deck is written as plain text in the format the local
:class:`DecklistLoader` parses, so it round-trips cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.decklist_loader import Decklist, DecklistLoader


def to_text(deck: Decklist) -> str:
    """Serialize a Decklist back to the plaintext format we parse."""
    out: list[str] = []
    if deck.commander:
        out.append("Commander")
        for name in deck.commander:
            out.append(f"1 {name}")
        out.append("")
    out.append("Mainboard")
    for name, count in sorted(deck.mainboard.items()):
        out.append(f"{count} {name}")
    if deck.sideboard:
        out.append("")
        out.append("Sideboard")
        for name, count in sorted(deck.sideboard.items()):
            out.append(f"{count} {name}")
    out.append("")
    return "\n".join(out)


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "deck"


async def main(args: argparse.Namespace) -> int:
    loader = DecklistLoader()
    bracket = args.bracket
    if isinstance(bracket, str) and bracket.isdigit():
        bracket = int(bracket)
    print(f"Fetching: {args.url}  (bracket={bracket})")
    deck = await loader.from_url(args.url, bracket=bracket)
    total = sum(deck.mainboard.values()) + len(deck.commander) + sum(deck.sideboard.values())
    print(f"  -> {total} cards "
          f"(mainboard={sum(deck.mainboard.values())}, "
          f"commanders={len(deck.commander)}, sideboard={sum(deck.sideboard.values())})")
    if deck.commander:
        print(f"  Commanders: {', '.join(deck.commander)}")

    if args.out:
        out_path = Path(args.out)
    else:
        # Auto name: data/decks/<commander|domain>.txt
        if deck.commander:
            stem = slugify(deck.commander[0])
        else:
            stem = "deck_" + slugify(args.url)[-12:]
        out_path = Path("data/decks") / f"{stem}.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_text(deck), encoding="utf-8")
    print(f"  Wrote: {out_path}")

    if args.classify:
        from src.integrations.commander_brackets import classify_bracket
        from src.integrations.offline_card_db import get_default_db

        db = get_default_db()
        if len(db) == 0:
            print("[warn] Local Scryfall cache is empty, can't classify. "
                  "Run: python scripts/fetch_card_data.py")
            return 0
        rep = classify_bracket(deck, db)
        print(f"\nBracket {rep.bracket}: {rep.label} (score={rep.score})")
        for n in rep.notes:
            print(f"  - {n}")
        if rep.violations:
            print("  ! Legality issues:")
            for v in rep.violations:
                print(f"    {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", help="Moxfield / EDHREC / Archidekt URL or deck ID")
    p.add_argument("--out", default=None, help="Output path (default: data/decks/<slug>.txt)")
    p.add_argument("--classify", action="store_true",
                   help="Run bracket classification after fetching")
    p.add_argument("--bracket", default=None,
                   help="EDHREC bracket filter: 1..5, exhibition, core, "
                        "upgraded, optimized, cedh, budget, expensive")
    return p


if __name__ == "__main__":
    sys.exit(asyncio.run(main(build_parser().parse_args())))
