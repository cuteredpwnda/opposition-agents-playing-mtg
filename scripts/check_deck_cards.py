"""Check whether every card in a decklist is present in the offline DB.

Usage:
    python scripts/check_deck_cards.py data/decks/modern_burn.txt [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.decklist_loader import DecklistLoader
from src.integrations.offline_card_db import get_default_db


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_deck_cards.py DECK [DECK ...]")
        return 2
    db = get_default_db()
    print(f"[db] {len(db)} cards loaded")
    rc = 0
    for path in sys.argv[1:]:
        deck = DecklistLoader().from_text(Path(path).read_text(encoding="utf-8"))
        names = list(deck.commander) + list(deck.mainboard.keys())
        missing = [n for n in names if db.get(n) is None]
        print(f"\n{path}  ({len(names)} unique, {sum(deck.mainboard.values())} mainboard)")
        if missing:
            print(f"  MISSING ({len(missing)}):")
            for m in missing:
                print(f"    - {m}")
            rc = 1
        else:
            print("  all cards present")
    return rc


if __name__ == "__main__":
    sys.exit(main())
