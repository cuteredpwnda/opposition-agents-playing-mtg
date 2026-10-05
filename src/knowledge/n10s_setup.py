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
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote

from neo4j import AsyncGraphDatabase
from src.config import settings

logger = logging.getLogger(__name__)

GRAPH_CONFIG = {
    "handleVocabUris": "MAP",
    "handleMultival": "ARRAY",
    "handleRDFTypes": "LABELS_AND_NODES",
}


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
        await self.driver.verify_connectivity()
        async with self.driver.session() as session:
            result = await session.run("CALL n10s.graphconfig.show()")
            current = {record["param"]: record["value"] async for record in result}
            if current:
                mismatches = {
                    key: current.get(key) for key, value in GRAPH_CONFIG.items()
                    if current.get(key) != value
                }
                if mismatches:
                    raise RuntimeError(
                        f"Incompatible n10s graph config {mismatches}; "
                        "review migration explicitly. Existing config was not changed."
                    )
            result = await session.run(
                "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS "
                "FOR (r:Resource) REQUIRE r.uri IS UNIQUE"
            )
            await result.consume()
            if current:
                logger.info("Compatible n10s graph config retained")
                return
            result = await session.run("CALL n10s.graphconfig.init($config)", config=GRAPH_CONFIG)
            await result.consume()
            logger.info("n10s graph config initialized")

    async def _import(
        self, procedure: str, path: str, format_: str | None = None,
    ) -> dict[str, Any]:
        uri = self._file_uri(path)
        format_ = format_ or self._rdf_format(path)
        async with self.driver.session() as session:
            result = await session.run(
                f"CALL {procedure}($uri, $format)", uri=uri, format=format_,
            )
            record = await result.single()
            stats = dict(record) if record else {}
            if stats.get("terminationStatus") != "OK":
                raise RuntimeError(f"{procedure} failed: {stats}")
            for key in ("triplesParsed", "triplesLoaded"):
                if type(stats.get(key)) is not int or stats[key] <= 0:
                    raise RuntimeError(f"{procedure} returned no verified {key}: {stats}")
            logger.info("%s completed: %s", procedure, stats)
            return stats

    async def import_ontology(self, ontology_path: str | None = None) -> dict[str, Any]:
        """Import the OWL ontology TBox into Neo4j.

        Uses n10s.onto.import.fetch() which creates:
        - (:Class) nodes for owl:Class
        - (:Relationship) nodes for owl:ObjectProperty
        - [:SCO] edges for rdfs:subClassOf
        - [:SPO] edges for rdfs:subPropertyOf
        - [:DOMAIN] / [:RANGE] edges
        """
        path = ontology_path or settings.ontology_path
        return await self._import("n10s.onto.import.fetch", path)

    async def import_instances(self, ontology_path: str | None = None) -> dict[str, Any]:
        """Import RDF statements, including schema and seed individuals.

        This is not a bulk card/combo import; n10s.rdf imports all source triples.
        """
        path = ontology_path or settings.ontology_path
        return await self._import("n10s.rdf.import.fetch", path)

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
        from src.knowledge.abox_builder import CONSTRAINTS

        async with self.driver.session() as session:
            # Unique constraint on card name
            result = await session.run(
                "CREATE CONSTRAINT card_name_unique IF NOT EXISTS "
                "FOR (c:Card) REQUIRE c.cardName IS UNIQUE"
            )
            await result.consume()
            for statement in CONSTRAINTS:
                result = await session.run(statement)
                await result.consume()
            # Index on scryfall ID
            result = await session.run(
                "CREATE CONSTRAINT scryfall_id_unique IF NOT EXISTS "
                "FOR (c:Card) REQUIRE c.scryfallId IS UNIQUE"
            )
            await result.consume()
            # Combo + outcome uniqueness (added with the outcome ontology
            # so MERGE on (:Combo {comboId}) / (:Outcome {outcomeId})
            # stays O(1)).
            result = await session.run(
                "CREATE CONSTRAINT combo_id_unique IF NOT EXISTS "
                "FOR (c:Combo) REQUIRE c.comboId IS UNIQUE"
            )
            await result.consume()
            result = await session.run(
                "CREATE CONSTRAINT outcome_id_unique IF NOT EXISTS "
                "FOR (o:Outcome) REQUIRE o.outcomeId IS UNIQUE"
            )
            await result.consume()
            # Full-text search index (idempotent — Neo4j 5+ syntax)
            result = await session.run(
                "CREATE FULLTEXT INDEX cardSearch IF NOT EXISTS "
                "FOR (c:Card) ON EACH [c.cardName, c.oracleText, c.typeLine]"
            )
            await result.consume()
            result = await session.run(
                "CREATE FULLTEXT INDEX cardDesignSearch IF NOT EXISTS "
                "FOR (c:CardDesign) ON EACH [c.cardName, c.oracleText, c.typeLine]"
            )
            await result.consume()
            logger.info("Neo4j indexes created")

    async def create_vector_index(self, dimensions: int = 128) -> None:
        """Create Neo4j native vector index for card embeddings (idempotent)."""
        async with self.driver.session() as session:
            existing = await session.run(
                "SHOW INDEXES YIELD name WHERE name = 'cardEmbeddings' RETURN name"
            )
            if [r async for r in existing]:
                logger.info("Vector index 'cardEmbeddings' already exists")
                return
            await session.run(
                "CALL db.index.vector.createNodeIndex("
                "'cardEmbeddings', 'Card', 'embedding', $dim, 'cosine'"
                ")",
                dim=dimensions,
            )
            logger.info(f"Vector index created (dim={dimensions})")

    async def full_setup(self) -> dict[str, dict[str, Any]]:
        """Run complete setup: config → ontology → instances → indexes."""
        await self.init_graph_config()
        reports = {
            "schema": await self.import_ontology(),
            "schema_rdf": await self.import_instances(),
            "vocabulary": await self.import_instances(settings.ontology_vocabulary_path),
        }
        await self.create_indexes()
        logger.info("Full n10s setup complete")
        return reports

    @staticmethod
    def _rdf_format(path: str) -> str:
        formats = {".ttl": "Turtle", ".owl": "RDF/XML", ".rdf": "RDF/XML",
                   ".xml": "RDF/XML", ".nt": "N-Triples", ".jsonld": "JSON-LD"}
        suffix = Path(path).suffix.lower()
        if suffix not in formats:
            raise ValueError(f"Unsupported RDF file extension: {path}")
        return formats[suffix]

    @staticmethod
    def _file_uri(path: str) -> str:
        """Convert a relative or absolute path to a file:// URI for Neo4j import.

        When running in Docker, the ontology is mounted at /import/ontology/.
        """
        p = Path(path)
        if p.is_absolute():
            return p.as_uri()
        # Docker mount: ./data/ontology/ → /import/ontology/
        parts = PureWindowsPath(path).parts
        if parts[:2] == ("data", "ontology") and ".." not in parts:
            return "file:///import/ontology/" + quote("/".join(parts[2:]))
        return f"file:///{p.resolve().as_posix()}"
