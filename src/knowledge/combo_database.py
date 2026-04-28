"""
Combo data fetchers.

Primary source: Commander Spellbook (https://commanderspellbook.com/).
Supplemental: EDHREC commander combo pages (https://json.edhrec.com/).

Both fetchers are deliberately polite:
  * fixed base delay between successful requests
  * full-jitter exponential backoff on 429 / 5xx
  * honours ``Retry-After``
  * incremental persistence so a network blip never loses progress
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SPELLBOOK_API = "https://backend.commanderspellbook.com"
EDHREC_JSON = "https://json.edhrec.com"

# Polite defaults.  We are guests on these APIs.
DEFAULT_BASE_DELAY = 1.0       # seconds between successful page fetches
DEFAULT_MAX_BACKOFF = 120.0    # cap for exponential backoff
DEFAULT_USER_AGENT = (
    "opposition-agents-mtg/0.1 (research; https://github.com/; contact via repo)"
)


async def _polite_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_retries: int = 8,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
) -> httpx.Response | None:
    """GET with full-jitter exponential backoff and Retry-After handling.

    Returns the successful response or ``None`` if all retries were
    exhausted.  Never raises on HTTP status — caller decides what to do.
    """
    for attempt in range(max_retries):
        try:
            resp = await client.get(url)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            wait = min(max_backoff, 2 ** attempt) * (0.5 + random.random() / 2)
            logger.warning("Network error on %s: %s — sleeping %.1fs (%d/%d)",
                           url, e, wait, attempt + 1, max_retries)
            await asyncio.sleep(wait)
            continue

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    wait = float(ra)
                except ValueError:
                    wait = min(max_backoff, 2 ** attempt)
            else:
                wait = min(max_backoff, 2 ** attempt)
            # full jitter
            wait = wait * (0.5 + random.random() / 2)
            logger.warning(
                "%d on %s — backing off %.1fs (attempt %d/%d)",
                resp.status_code, url, wait, attempt + 1, max_retries,
            )
            await asyncio.sleep(wait)
            continue

        if resp.status_code >= 400:
            logger.error("HTTP %d on %s — giving up", resp.status_code, url)
            return None

        return resp

    logger.error("Exhausted %d retries on %s", max_retries, url)
    return None


async def fetch_all_combos(
    output_path: str = "data/combos.json",
    page_size: int = 100,
    save_every: int = 10,
    max_retries: int = 8,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> int:
    """Download the full combo database from Commander Spellbook.

    Persists incrementally so a 429 / network blip never loses progress.
    Reuses an existing ``output_path`` file iff it parses as a non-empty
    JSON list (delete the file to force a refresh).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(existing, list) and len(existing) > 100:
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
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True, headers=headers) as client:
        while next_url:
            resp = await _polite_get(client, next_url, max_retries=max_retries)
            if resp is None:
                logger.warning("Stopping early; persisting %d combos collected so far", len(combos))
                break
            data = resp.json()
            page = data if isinstance(data, list) else data.get("results", [])
            combos.extend(page)
            next_url = data.get("next") if isinstance(data, dict) else None
            page_idx += 1
            if page_idx % save_every == 0:
                out.write_text(json.dumps(combos, indent=2), encoding="utf-8")
                logger.info("Fetched %d combos so far (checkpoint saved)...", len(combos))
            # Polite spacing between successful pages.
            await asyncio.sleep(base_delay * (0.8 + 0.4 * random.random()))

    out.write_text(json.dumps(combos, indent=2), encoding="utf-8")
    logger.info("Downloaded %d combos -> %s", len(combos), out)
    return len(combos)


async def fetch_edhrec_combos(
    output_path: str = "data/combos_edhrec.json",
    max_retries: int = 8,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> int:
    """Fetch the global EDHREC combos page.

    EDHREC exposes a JSON mirror of its rendered pages at
    ``https://json.edhrec.com/pages/<slug>.json``.  The combos index lives
    at ``/pages/combos.json`` and contains a ``cardlists`` payload of the
    most-played commander combos, each with a ``cards`` field of card
    names.  This is *not* exhaustive (Spellbook is the canonical source)
    but it captures the popular / metagame-relevant combos and gives a
    fallback when Spellbook is rate-limited or down.

    Output schema is intentionally compatible with our combo importer:
    a list of ``{"uses": [{"card": {"name": str}}, ...], "results": [...],
    "source": "edhrec"}`` records.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(existing, list) and len(existing) > 10:
                logger.info("Re-using cached EDHREC combos (%d entries)", len(existing))
                return len(existing)
        except Exception:
            pass

    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    url = f"{EDHREC_JSON}/pages/combos.json"
    combos: list[dict] = []
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True, headers=headers) as client:
        resp = await _polite_get(client, url, max_retries=max_retries)
        if resp is None:
            logger.warning("EDHREC combos page unreachable; writing empty file")
            out.write_text("[]", encoding="utf-8")
            return 0
        await asyncio.sleep(base_delay)
        data = resp.json()
        # EDHREC packs lists under container.json_dict.cardlists
        container = data.get("container", {}) if isinstance(data, dict) else {}
        json_dict = container.get("json_dict", container)
        cardlists = (
            json_dict.get("cardlists")
            or data.get("cardlists")
            or []
        )
        for cl in cardlists:
            for entry in cl.get("cardviews", []) or []:
                # EDHREC entry names look like "Card A + Card B + Card C".
                raw_name = entry.get("name", "")
                parts = [p.strip() for p in raw_name.split("+") if p.strip()]
                if len(parts) < 2:
                    continue
                combos.append({
                    "uses": [{"card": {"name": p}} for p in parts],
                    "results": [],
                    "source": "edhrec",
                    "url": entry.get("url"),
                    "popularity": entry.get("num_decks") or entry.get("inclusion"),
                })

    out.write_text(json.dumps(combos, indent=2), encoding="utf-8")
    logger.info("Downloaded %d EDHREC combos -> %s", len(combos), out)
    return len(combos)


async def fetch_combos_merged(
    spellbook_path: str = "data/combos.json",
    edhrec_path: str = "data/combos_edhrec.json",
    merged_path: str = "data/combos_merged.json",
    use_edhrec: bool = True,
) -> int:
    """Convenience: fetch both sources and write a merged file.

    Spellbook entries are kept verbatim; EDHREC entries are appended only
    if their card-set isn't already covered by a Spellbook entry.
    """
    sb_count = await fetch_all_combos(spellbook_path)
    edh_count = 0
    if use_edhrec:
        edh_count = await fetch_edhrec_combos(edhrec_path)

    sb = json.loads(Path(spellbook_path).read_text(encoding="utf-8")) if Path(spellbook_path).exists() else []
    edh = json.loads(Path(edhrec_path).read_text(encoding="utf-8")) if Path(edhrec_path).exists() else []

    def _key(entry: dict) -> frozenset[str]:
        return frozenset(
            (u.get("card") or {}).get("name", "").strip().lower()
            for u in (entry.get("uses") or [])
        )

    seen = {_key(c) for c in sb}
    merged = list(sb)
    new_from_edh = 0
    for c in edh:
        k = _key(c)
        if k and k not in seen:
            merged.append(c)
            seen.add(k)
            new_from_edh += 1

    Path(merged_path).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    logger.info(
        "Merged combos: spellbook=%d edhrec=%d (+%d unique) -> %s (%d total)",
        sb_count, edh_count, new_from_edh, merged_path, len(merged),
    )
    return len(merged)
