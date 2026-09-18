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

BULK_DATA_PATH = "data/scryfall/oracle-cards.json"


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


async def main(shape: str = "ontology") -> None:
    path = BULK_DATA_PATH
    if not Path(path).exists():
        logger.info("Bulk data not found locally — downloading...")
        path = await download_bulk_data(path)
    else:
        logger.info(f"Using existing bulk data: {path}")

    if shape == "ontology":
        from src.knowledge.abox_builder import AboxBuilder

        abox = AboxBuilder()
        try:
            report = await abox.import_from_bulk(path)
            logger.info("ABox import complete: %s", report.summary())
        finally:
            await abox.close()
        # Legalities still come from the legacy builder; they are unaffected
        # by the design/printing split.
        builder = KGBuilder()
        try:
            await builder.import_legalities(path)
        finally:
            await builder.close()
        return

    builder = KGBuilder()
    try:
        count = await builder.import_scryfall_cards(path)
        await builder.import_legalities(path)
        logger.info(f"Import complete: {count} cards + legalities")
    finally:
        await builder.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shape",
        choices=["ontology", "legacy"],
        default="ontology",
        help=(
            "ontology: CardDesign/CardPrinting with classified types, subtypes "
            "and quality/region values (matches mtg-ontology-v2.0). "
            "legacy: the flat (:Card) node with type-derived labels."
        ),
    )
    asyncio.run(main(ap.parse_args().shape))
