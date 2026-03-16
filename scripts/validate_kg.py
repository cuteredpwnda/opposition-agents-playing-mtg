"""
Run n10s SHACL validation against the knowledge graph.

Usage: python -m scripts.validate_kg
"""

from __future__ import annotations

import asyncio
import logging

from src.knowledge.n10s_setup import N10sSetup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    setup = N10sSetup()
    try:
        logger.info("Importing SHACL shapes...")
        await setup.import_shacl_shapes()

        logger.info("Running SHACL validation...")
        violations = await setup.validate()

        if violations:
            logger.warning(f"Found {len(violations)} violations:")
            for v in violations:
                logger.warning(f"  {v}")
        else:
            logger.info("Validation passed — no violations found.")
    finally:
        await setup.close()


if __name__ == "__main__":
    asyncio.run(main())
