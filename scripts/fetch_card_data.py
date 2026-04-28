#!/usr/bin/env python
"""Download Scryfall bulk data + per-card rulings to local JSON cache.

Unlike `scripts/import_scryfall.py`, this requires NO Neo4j — it just
populates `data/scryfall/` with everything the engine needs to play offline:

    data/scryfall/oracle-cards.json   ~150 MB   all unique cards (oracle text, mana cost, etc.)
    data/scryfall/rulings.json         ~50 MB   all judge rulings + errata indexed by oracle_id
    data/scryfall/by_name.json        index    name (lowercase) -> oracle_id

These three files are enough for:
    - Building decks from real card data
    - The judge system to look up errata / rulings (e.g. Chains of Mephistopheles)
    - The mechanic miner to scan oracle text for patterns

Usage:
    python scripts/fetch_card_data.py                  # download everything
    python scripts/fetch_card_data.py --skip-rulings   # only oracle cards
    python scripts/fetch_card_data.py --refresh        # force re-download

Reference: https://scryfall.com/docs/api/bulk-data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scryfall"

ORACLE_PATH = DATA_DIR / "oracle-cards.json"
RULINGS_PATH = DATA_DIR / "rulings.json"
INDEX_PATH = DATA_DIR / "by_name.json"

USER_AGENT = "OppositionAgentsMTG/1.0 (research)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}


async def fetch_bulk_url(client: httpx.AsyncClient, bulk_type: str) -> tuple[str, int]:
    """Look up the download URL + size for a bulk-data type."""
    resp = await client.get("https://api.scryfall.com/bulk-data")
    resp.raise_for_status()
    for item in resp.json()["data"]:
        if item["type"] == bulk_type:
            return item["download_uri"], item.get("size", 0)
    raise ValueError(f"bulk type '{bulk_type}' not found")


async def download_file(client: httpx.AsyncClient, url: str, out_path: Path) -> None:
    """Stream a large file to disk with a simple progress indicator."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {url}")
    print(f"     writing to {out_path}")

    async with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1 << 20  # 1 MB

        with out_path.open("wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r     {downloaded/1e6:.1f} / {total/1e6:.1f} MB "
                          f"({pct:.1f}%)", end="", flush=True)
        print()


def build_name_index(oracle_path: Path, index_path: Path) -> int:
    """Build a lowercase-name -> oracle_id index for fast lookup."""
    print(f"Building name index from {oracle_path.name}...")
    cards = json.loads(oracle_path.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for c in cards:
        name = c.get("name")
        oracle_id = c.get("oracle_id") or c.get("id")
        if name and oracle_id:
            index[name.lower()] = oracle_id
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {len(index):,} cards indexed -> {index_path}")
    return len(index)


async def main(args) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(headers=HEADERS, timeout=120.0) as client:

        # --- Oracle cards -------------------------------------------------
        if ORACLE_PATH.exists() and not args.refresh:
            print(f"Skipping oracle-cards (exists: {ORACLE_PATH.stat().st_size/1e6:.1f} MB) "
                  f"— use --refresh to redownload")
        else:
            print("Fetching oracle-cards bulk URL...")
            url, size = await fetch_bulk_url(client, "oracle_cards")
            print(f"  size: {size/1e6:.1f} MB")
            await download_file(client, url, ORACLE_PATH)

        # --- Rulings (errata + judge clarifications) ---------------------
        if args.skip_rulings:
            print("Skipping rulings (--skip-rulings)")
        elif RULINGS_PATH.exists() and not args.refresh:
            print(f"Skipping rulings (exists: {RULINGS_PATH.stat().st_size/1e6:.1f} MB) "
                  f"— use --refresh to redownload")
        else:
            print("\nFetching rulings bulk URL...")
            url, size = await fetch_bulk_url(client, "rulings")
            print(f"  size: {size/1e6:.1f} MB")
            await download_file(client, url, RULINGS_PATH)

        # --- Name index --------------------------------------------------
        if ORACLE_PATH.exists():
            build_name_index(ORACLE_PATH, INDEX_PATH)

    print("\nDone! Cached files:")
    for p in (ORACLE_PATH, RULINGS_PATH, INDEX_PATH):
        if p.exists():
            print(f"  {p.relative_to(PROJECT_ROOT)} ({p.stat().st_size/1e6:.1f} MB)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--refresh", action="store_true",
                   help="Force re-download even if files exist")
    p.add_argument("--skip-rulings", action="store_true",
                   help="Skip rulings download (saves ~50 MB)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(asyncio.run(main(args)))
