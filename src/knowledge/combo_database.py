"""
Commander Spellbook integration — fetch combo data.

API: https://commanderspellbook.com/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SPELLBOOK_API = "https://backend.commanderspellbook.com"


async def fetch_all_combos(output_path: str = "data/combos.json") -> int:
    """Download the full combo database from Commander Spellbook.

    The endpoint is paginated (DRF style — ``next`` URL in each page).
    We page through all of them and persist the union to ``output_path``.

    Returns the number of combos downloaded.
    """
    combos: list[dict] = []
    next_url: str | None = f"{SPELLBOOK_API}/variants/?limit=500"
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        while next_url:
            resp = await client.get(next_url)
            resp.raise_for_status()
            data = resp.json()
            page = data if isinstance(data, list) else data.get("results", [])
            combos.extend(page)
            next_url = data.get("next") if isinstance(data, dict) else None
            logger.info("Fetched %d combos so far...", len(combos))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(combos, indent=2), encoding="utf-8")
    logger.info(f"Downloaded {len(combos)} combos → {output_path}")
    return len(combos)
