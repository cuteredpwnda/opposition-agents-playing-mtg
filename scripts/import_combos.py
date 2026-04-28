"""
Download Commander Spellbook + EDHREC combos and import into Neo4j.

Usage: python -m scripts.import_combos [--no-edhrec]
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from src.knowledge.combo_database import fetch_combos_merged
from src.knowledge.kg_builder import KGBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MERGED_PATH = "data/combos_merged.json"


async def main(use_edhrec: bool = True) -> None:
    logger.info("Fetching combos (spellbook + edhrec=%s)...", use_edhrec)
    count = await fetch_combos_merged(merged_path=MERGED_PATH, use_edhrec=use_edhrec)
    logger.info("Total merged combos: %d", count)

    builder = KGBuilder()
    try:
        imported = await builder.import_combos(MERGED_PATH)
        logger.info("Imported %d combos into Neo4j", imported)
    finally:
        await builder.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-edhrec", action="store_true", help="Skip EDHREC fallback source")
    args = p.parse_args()
    asyncio.run(main(use_edhrec=not args.no_edhrec))
