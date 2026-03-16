"""
n10s (neosemantics) setup — import OWL ontology into Neo4j.

This module handles:
1. Initializing the n10s graph config
2. Importing the OWL ontology as TBox (class hierarchy, properties)
3. Importing seed ABox instances from the OWL file
4. Loading SHACL shapes for validation
5. Running SHACL validation

Reference: https://neo4j.com/labs/neosemantics/
"""

from __future__ import annotations

import logging
from pathlib import Path

from neo4j import AsyncGraphDatabase

from src.config import settings

logger = logging.getLogger(__name__)


class N10sSetup:
    """Manages neosemantics (n10s) ontology lifecycle in Neo4j."""

    def __init__(
        self,
        uri: str = settings.neo4j_uri,
        user: str = settings.neo4j_user,
        password: str = settings.neo4j_password,
    ):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self.driver.close()

    async def init_graph_config(self) -> None:
        """Initialize n10s graph configuration.

        Must be called once on a fresh database before any RDF import.
        handleVocabUris: 'MAP' — maps full URIs to short names.
        handleMultival: 'ARRAY' — multi-valued properties become arrays.
        handleRDFTypes: 'LABELS_AND_NODES' — RDF types become Neo4j labels.
        """
        async with self.driver.session() as session:
            # Drop existing config if any (idempotent setup)
            await session.run("CALL n10s.graphconfig.drop() YIELD value RETURN value")
            await session.run(
                """
                CALL n10s.graphconfig.init({
                    handleVocabUris: 'MAP',
                    handleMultival: 'ARRAY',
                    handleRDFTypes: 'LABELS_AND_NODES'
                })
                """
            )
            logger.info("n10s graph config initialized")

    async def import_ontology(self, ontology_path: str | None = None) -> dict:
        """Import the OWL ontology TBox into Neo4j.

        Uses n10s.onto.import.fetch() which creates:
        - (:Class) nodes for owl:Class
        - (:Relationship) nodes for owl:ObjectProperty
        - [:SCO] edges for rdfs:subClassOf
        - [:SPO] edges for rdfs:subPropertyOf
        - [:DOMAIN] / [:RANGE] edges
        """
        path = ontology_path or settings.ontology_path
        uri = self._file_uri(path)

        async with self.driver.session() as session:
            result = await session.run(
                "CALL n10s.onto.import.fetch($uri, 'RDF/XML')",
                uri=uri,
            )
            record = await result.single()
            stats = dict(record) if record else {}
            logger.info(f"Ontology TBox imported: {stats}")
            return stats

    async def import_instances(self, ontology_path: str | None = None) -> dict:
        """Import ABox instances (seed data) from the OWL file.

        Uses n10s.rdf.import.fetch() which creates actual card/combo nodes
        from the NamedIndividual declarations in the ontology.
        """
        path = ontology_path or settings.ontology_path
        uri = self._file_uri(path)

        async with self.driver.session() as session:
            result = await session.run(
                "CALL n10s.rdf.import.fetch($uri, 'RDF/XML')",
                uri=uri,
            )
            record = await result.single()
            stats = dict(record) if record else {}
            logger.info(f"ABox instances imported: {stats}")
            return stats

    async def import_shacl_shapes(self, shapes_path: str | None = None) -> dict:
        """Import SHACL shapes for data validation."""
        path = shapes_path or settings.shacl_shapes_path
        uri = self._file_uri(path)

        async with self.driver.session() as session:
            result = await session.run(
                "CALL n10s.validation.shacl.import.fetch($uri, 'Turtle')",
                uri=uri,
            )
            record = await result.single()
            stats = dict(record) if record else {}
            logger.info(f"SHACL shapes imported: {stats}")
            return stats

    async def validate(self) -> list[dict]:
        """Run SHACL validation against loaded shapes.

        Returns list of violations (empty = valid).
        """
        async with self.driver.session() as session:
            result = await session.run(
                "CALL n10s.validation.shacl.validate()"
            )
            violations = [dict(record) async for record in result]
            if violations:
                logger.warning(f"SHACL validation found {len(violations)} violations")
            else:
                logger.info("SHACL validation passed — no violations")
            return violations

    async def create_indexes(self) -> None:
        """Create Neo4j indexes for efficient querying."""
        async with self.driver.session() as session:
            # Unique constraint on card name
            await session.run(
                "CREATE CONSTRAINT card_name_unique IF NOT EXISTS "
                "FOR (c:Card) REQUIRE c.cardName IS UNIQUE"
            )
            # Index on scryfall ID
            await session.run(
                "CREATE CONSTRAINT scryfall_id_unique IF NOT EXISTS "
                "FOR (c:Card) REQUIRE c.scryfallId IS UNIQUE"
            )
            # Full-text search index
            await session.run(
                "CALL db.index.fulltext.createNodeIndex("
                "'cardSearch', ['Card'], ['cardName', 'oracleText', 'typeLine']"
                ")"
            )
            logger.info("Neo4j indexes created")

    async def create_vector_index(self, dimensions: int = 384) -> None:
        """Create Neo4j native vector index for card embeddings."""
        async with self.driver.session() as session:
            await session.run(
                "CALL db.index.vector.createNodeIndex("
                "'cardEmbeddings', 'Card', 'embedding', $dim, 'cosine'"
                ")",
                dim=dimensions,
            )
            logger.info(f"Vector index created (dim={dimensions})")

    async def full_setup(self) -> None:
        """Run complete setup: config → ontology → instances → indexes."""
        await self.init_graph_config()
        await self.import_ontology()
        await self.import_instances()
        await self.create_indexes()
        logger.info("Full n10s setup complete")

    @staticmethod
    def _file_uri(path: str) -> str:
        """Convert a relative or absolute path to a file:// URI for Neo4j import.

        When running in Docker, the ontology is mounted at /import/ontology/.
        """
        p = Path(path)
        if p.is_absolute():
            return p.as_uri()
        # Docker mount: ./data/ontology/ → /import/ontology/
        if path.startswith("data/ontology/"):
            docker_path = path.replace("data/ontology/", "/import/ontology/")
            return f"file://{docker_path}"
        return f"file:///{p.resolve().as_posix()}"
