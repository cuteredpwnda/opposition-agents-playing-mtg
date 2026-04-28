"""Offline card-data lookup against the local Scryfall bulk cache.

Loads ``data/scryfall/oracle-cards.json`` (~165 MB) into memory once
and exposes O(1) name lookups returning Scryfall card dicts. Used by
deck loaders so we can build real Standard / Modern decks without any
network calls or rate limits.

Usage::

    db = OfflineCardDB.load_default()
    card = db.get("Lightning Bolt")  # → Scryfall dict
    if card is None:
        print("not found")

The first call is slow (~3-6 s to parse the JSON); subsequent lookups
are dict accesses. Lazy-loaded singleton via :func:`get_default_db`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_ORACLE = Path("data/scryfall/oracle-cards.json")


@dataclass
class OfflineCardDB:
    """In-memory Scryfall card database keyed by lowercase name."""

    by_name: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, oracle_path: Path = DEFAULT_ORACLE) -> "OfflineCardDB":
        db = cls()
        if not oracle_path.exists():
            logger.warning(
                "Offline Scryfall cache missing at %s. "
                "Run: python scripts/fetch_card_data.py",
                oracle_path,
            )
            return db
        cards = json.loads(oracle_path.read_text(encoding="utf-8"))
        for c in cards:
            name = c.get("name")
            if not name:
                continue
            db.by_name[name.lower()] = c
            # Also index split / DFC face names so "Fire" and "Fire // Ice"
            # both resolve.
            for face in c.get("card_faces", []) or []:
                fname = face.get("name")
                if fname:
                    db.by_name.setdefault(fname.lower(), c)
        logger.info("Offline card DB loaded: %d cards", len(db.by_name))
        return db

    @classmethod
    def load_default(cls) -> "OfflineCardDB":
        return cls.load(DEFAULT_ORACLE)

    def get(self, name: str) -> Optional[dict]:
        return self.by_name.get(name.lower())

    def __contains__(self, name: str) -> bool:
        return name.lower() in self.by_name

    def __len__(self) -> int:
        return len(self.by_name)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_DEFAULT: Optional[OfflineCardDB] = None


def get_default_db() -> OfflineCardDB:
    """Return a process-wide cached OfflineCardDB instance."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = OfflineCardDB.load_default()
    return _DEFAULT
