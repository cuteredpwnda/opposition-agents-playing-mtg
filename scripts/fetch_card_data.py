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
import tempfile
from pathlib import Path

import httpx

from src.integrations.scryfall_bulk import (
    HEADERS,
    bulk_download_info,
    download_bulk,
    fetch_bulk_metadata,
    file_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scryfall"

ORACLE_PATH = DATA_DIR / "oracle-cards.json"
RULINGS_PATH = DATA_DIR / "rulings.json"
INDEX_PATH = DATA_DIR / "by_name.json"

async def fetch_bulk_url(client: httpx.AsyncClient, bulk_type: str) -> tuple[str, int]:
    """Look up the download URL + size for a bulk-data type."""
    url, size, _ = bulk_download_info(await fetch_bulk_metadata(client, bulk_type))
    return url, size


async def download_file(client: httpx.AsyncClient, url: str, out_path: Path) -> None:
    """Download either legacy JSON or the current gzip JSONL representation."""
    field = "jsonl_download_uri" if url.endswith(".jsonl.gz") else "download_uri"
    await download_bulk(client, {"type": "bulk", field: url}, out_path)


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

    manifest_path = DATA_DIR / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    entries = manifest.get("files", {})
    with tempfile.TemporaryDirectory(prefix=".refresh-", dir=DATA_DIR) as directory:
        stage = Path(directory)
        replacements: list[tuple[Path, Path]] = []
        async with httpx.AsyncClient(headers=HEADERS, timeout=120.0) as client:
            for kind, path in (("oracle_cards", ORACLE_PATH), ("rulings", RULINGS_PATH)):
                if kind == "rulings" and args.skip_rulings:
                    print("Skipping rulings (--skip-rulings)")
                    continue
                if path.exists() and not args.refresh:
                    print(f"Keeping existing {path.name}; use --refresh to update")
                    entries.setdefault(path.name, {"status": "existing; source date unknown"})
                    entries[path.name].update(
                        {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                    )
                    continue
                item = await fetch_bulk_metadata(client, kind)
                url, size, _ = bulk_download_info(item)
                print(f"Downloading {kind}: {size / 1e6:.1f} MB from {url}", flush=True)
                staged = stage / path.name
                entries[path.name] = await download_bulk(client, item, staged)
                replacements.append((staged, path))
        staged_oracle = stage / ORACLE_PATH.name
        source = staged_oracle if staged_oracle.exists() else ORACLE_PATH
        staged_index = stage / INDEX_PATH.name
        build_name_index(source, staged_index)
        entries[INDEX_PATH.name] = {
            "sha256": file_sha256(staged_index), "bytes": staged_index.stat().st_size,
            "oracle_sha256": entries[ORACLE_PATH.name]["sha256"],
        }
        replacements.append((staged_index, INDEX_PATH))
        staged_manifest = stage / manifest_path.name
        staged_manifest.write_text(
            json.dumps({"schema_version": 1, "files": entries}, indent=2) + "\n",
            encoding="utf-8",
        )
        # All downloads and the dependent index are validated before publication.
        # The manifest is published last; verify hashes before consuming a snapshot.
        for staged, destination in replacements:
            staged.replace(destination)
        staged_manifest.replace(manifest_path)

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
