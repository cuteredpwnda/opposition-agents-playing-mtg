"""Local cache for Scryfall rulings + errata.

Reads `data/scryfall/rulings.json` (downloaded by `scripts/fetch_card_data.py`)
and `data/scryfall/by_name.json` to provide zero-latency per-card lookups.

Scryfall's rulings feed already aggregates Wizards' official Gatherer
rulings + judge clarifications + functional errata, so a card like
"Chains of Mephistopheles" returns the modern templated text via
`get_rulings("Chains of Mephistopheles")`.

Usage in the judge:

    from src.judge.errata_loader import LocalRulingsCache
    cache = LocalRulingsCache.load_default()
    for ruling in cache.get_rulings("Chains of Mephistopheles"):
        print(ruling["comment"])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RULINGS = Path("data/scryfall/rulings.json")
DEFAULT_INDEX = Path("data/scryfall/by_name.json")


@dataclass
class LocalRulingsCache:
    """In-memory rulings cache keyed by oracle_id, with name index."""

    rulings_by_oracle_id: dict[str, list[dict]] = field(default_factory=dict)
    name_to_oracle_id: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load_default(cls) -> LocalRulingsCache:
        return cls.load(DEFAULT_RULINGS, DEFAULT_INDEX)

    @classmethod
    def load(cls, rulings_path: Path, index_path: Path) -> LocalRulingsCache:
        cache = cls()
        if not rulings_path.exists() or not index_path.exists():
            logger.warning(
                "Local rulings cache missing (%s, %s). "
                "Run: python scripts/fetch_card_data.py",
                rulings_path, index_path,
            )
            return cache

        rulings = json.loads(rulings_path.read_text(encoding="utf-8"))
        for r in rulings:
            oid = r.get("oracle_id")
            if not oid:
                continue
            cache.rulings_by_oracle_id.setdefault(oid, []).append({
                "source": r.get("source", "wotc"),
                "published_at": r.get("published_at", ""),
                "comment": r.get("comment", ""),
            })

        cache.name_to_oracle_id = json.loads(index_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded %d rulings across %d cards",
            sum(len(v) for v in cache.rulings_by_oracle_id.values()),
            len(cache.rulings_by_oracle_id),
        )
        return cache

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_rulings(self, card_name: str) -> list[dict]:
        """Return all rulings for a card (case-insensitive name match)."""
        oid = self.name_to_oracle_id.get(card_name.lower())
        if not oid:
            return []
        return list(self.rulings_by_oracle_id.get(oid, []))

    def is_loaded(self) -> bool:
        return bool(self.rulings_by_oracle_id)
