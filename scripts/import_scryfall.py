"""
Download Scryfall bulk data and import into Neo4j.

Usage: python -m scripts.import_scryfall
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx

from src.knowledge.kg_builder import KGBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BULK_DATA_PATH = "data/oracle-cards.json"


async def download_bulk_data(output_path: str = BULK_DATA_PATH) -> str:
    """Download Scryfall oracle-cards bulk data."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get("https://api.scryfall.com/bulk-data")
        resp.raise_for_status()
        for item in resp.json()["data"]:
            if item["type"] == "oracle_cards":
                download_uri = item["download_uri"]
                break
        else:
            raise ValueError("oracle_cards bulk data not found")

        logger.info(f"Downloading from {download_uri}")
        resp = await client.get(download_uri, follow_redirects=True)
        resp.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(resp.content)
    logger.info(f"Saved to {output_path}")
    return output_path


async def main() -> None:
    path = BULK_DATA_PATH
    if not Path(path).exists():
        logger.info("Bulk data not found locally — downloading...")
        path = await download_bulk_data(path)
    else:
        logger.info(f"Using existing bulk data: {path}")

    builder = KGBuilder()
    try:
        count = await builder.import_scryfall_cards(path)
        await builder.import_legalities(path)
        logger.info(f"Import complete: {count} cards + legalities")
    finally:
        await builder.close()


if __name__ == "__main__":
    asyncio.run(main())
