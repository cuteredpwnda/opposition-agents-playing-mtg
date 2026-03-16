"""
Download Commander Spellbook combos and import into Neo4j.

Usage: python -m scripts.import_combos
"""

from __future__ import annotations

import asyncio
import logging

from src.knowledge.combo_database import fetch_all_combos
from src.knowledge.kg_builder import KGBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMBOS_PATH = "data/combos.json"


async def main() -> None:
    logger.info("Fetching combos from Commander Spellbook...")
    count = await fetch_all_combos(COMBOS_PATH)
    logger.info(f"Downloaded {count} combos")

    builder = KGBuilder()
    try:
        imported = await builder.import_combos(COMBOS_PATH)
        logger.info(f"Imported {imported} combos into Neo4j")
    finally:
        await builder.close()


if __name__ == "__main__":
    asyncio.run(main())
