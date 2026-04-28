"""
Commander Spellbook integration — fetch combo data.

API: https://commanderspellbook.com/
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SPELLBOOK_API = "https://backend.commanderspellbook.com"


async def fetch_all_combos(
    output_path: str = "data/combos.json",
    page_size: int = 100,
    save_every: int = 10,
    max_retries: int = 6,
) -> int:
    """Download the full combo database from Commander Spellbook.

    Persists incrementally to ``output_path`` every ``save_every`` pages so
    a 429 / network blip doesn't lose progress.  Honours
    ``Retry-After`` (or backs off exponentially up to 60 s).  If
    ``output_path`` already exists and is non-empty, the file is reused
    (use ``rm`` to force a refresh).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1024:
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(existing, list) and existing:
                logger.info(
                    "Re-using cached combos from %s (%d entries) — delete the file to refresh",
                    out, len(existing),
                )
                return len(existing)
        except Exception:
            pass

    combos: list[dict] = []
    next_url: str | None = f"{SPELLBOOK_API}/variants/?limit={page_size}"
    page_idx = 0
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        while next_url:
            for attempt in range(max_retries):
                try:
                    resp = await client.get(next_url)
                    if resp.status_code == 429:
                        wait = float(resp.headers.get("Retry-After", min(2 ** attempt, 60)))
                        logger.warning(
                            "429 rate-limited; sleeping %.1fs (attempt %d/%d)",
                            wait, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    break
                except httpx.HTTPStatusError as e:
                    if attempt + 1 >= max_retries:
                        logger.error("Giving up on %s after %d attempts: %s",
                                     next_url, max_retries, e)
                        next_url = None
                        break
                    await asyncio.sleep(min(2 ** attempt, 30))
            else:
                # max_retries exhausted on 429 — persist what we have and return
                break
            if next_url is None:
                break

            data = resp.json()
            page = data if isinstance(data, list) else data.get("results", [])
            combos.extend(page)
            next_url = data.get("next") if isinstance(data, dict) else None
            page_idx += 1
            if page_idx % save_every == 0:
                out.write_text(json.dumps(combos, indent=2), encoding="utf-8")
                logger.info("Fetched %d combos so far (checkpoint saved)...", len(combos))

    out.write_text(json.dumps(combos, indent=2), encoding="utf-8")
    logger.info("Downloaded %d combos → %s", len(combos), out)
    return len(combos)
