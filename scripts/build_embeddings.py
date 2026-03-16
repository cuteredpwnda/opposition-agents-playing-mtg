"""
Build GNN embeddings and write back to Neo4j vector index.

Usage: python -m scripts.build_embeddings
"""

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Export graph from Neo4j → train GraphSAGE → write embeddings back."""
    from neo4j import AsyncGraphDatabase
    from src.config import settings
    from src.knowledge.n10s_setup import N10sSetup

    # 1. Create vector index if not exists
    setup = N10sSetup()
    try:
        await setup.create_vector_index(dimensions=384)
    except Exception:
        logger.info("Vector index already exists (or error creating)")
    finally:
        await setup.close()

    # 2. Export card graph from Neo4j
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        async with driver.session() as session:
            result = await session.run("MATCH (c:Card) RETURN count(c) AS n")
            record = await result.single()
            card_count = record["n"] if record else 0
            logger.info(f"Found {card_count} cards in Neo4j")

        if card_count == 0:
            logger.warning("No cards in Neo4j — run import_scryfall.py first")
            return

        # TODO: Export graph structure, build PyG Data, train GraphSAGE,
        #       write embeddings back via:
        #       MATCH (c:Card {cardName: $name}) SET c.embedding = $vector
        logger.info("Embedding pipeline not yet implemented — stub only")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
