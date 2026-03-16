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

SPELLBOOK_API = "https://backend.commanderspellbook.com/api/v2"


async def fetch_all_combos(output_path: str = "data/combos.json") -> int:
    """Download the full combo database from Commander Spellbook.

    Returns the number of combos downloaded.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(f"{SPELLBOOK_API}/variants/")
        resp.raise_for_status()
        data = resp.json()

    combos = data if isinstance(data, list) else data.get("results", [])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(combos, indent=2), encoding="utf-8")
    logger.info(f"Downloaded {len(combos)} combos → {output_path}")
    return len(combos)
