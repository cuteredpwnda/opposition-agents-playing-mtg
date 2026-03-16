"""
Local card data cache — SQLite-backed to avoid repeated Scryfall calls.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class CardCache:
    """SQLite cache of Scryfall card data for fast local lookup."""

    def __init__(self, db_path: str = "data/card_cache.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cards (
                name TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, card_name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT data FROM cards WHERE name = ?", (card_name,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, card_name: str, data: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cards (name, data) VALUES (?, ?)",
            (card_name, json.dumps(data)),
        )
        self.conn.commit()

    def bulk_put(self, cards: list[dict[str, Any]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO cards (name, data) VALUES (?, ?)",
            [(c["name"], json.dumps(c)) for c in cards if c.get("name")],
        )
        self.conn.commit()

    def has(self, card_name: str) -> bool:
        return self.get(card_name) is not None

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM cards").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self.conn.close()
