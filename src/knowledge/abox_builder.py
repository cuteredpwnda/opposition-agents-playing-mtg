"""Populate the knowledge graph in the shape the ontology actually declares.

The previous importer (:class:`src.knowledge.kg_builder.KGBuilder`) wrote a
single flat ``(:Card)`` node per oracle entry, derived Neo4j *labels* from
substring matches on the type line, and had no representation for subtypes,
printings, or characteristic-defining power. That was fine for a property
graph and wrong for the ontology: it conflated the card design with its
printings and with in-game objects, used subsumption where the rules use
classification, and could not express ``*`` as a power.

This builder emits the v2.0 patterns instead:

``CardDesign`` / ``CardPrinting``
    The DUL information-realisation pattern. One design, many printings,
    linked by ``REALIZES``.
``hasCardType`` / ``hasSubtype``
    Classification against the Comprehensive-Rules vocabularies rather than
    Neo4j labels, because an in-game object's types can change and a
    subsumption edge cannot be retracted.
``Power`` / ``Toughness`` qualities with ``ValueRegion`` values
    Keeps the lexical value always and the numeric value only when the
    printed value is determinate, so ``*`` and ``1+*`` survive without
    degrading the datatype for the whole pool.

Every node carries ``epistemicStatus: "curated"``; this importer never
writes induced data.

Requires Neo4j. The parsing half is in :mod:`src.knowledge.type_line` and
is deliberately free of any database dependency so it can be tested and
run offline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.knowledge.type_line import (
    ParsedTypeLine,
    TypeSystem,
    load_type_system,
    parse_type_line,
    split_faces,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


@dataclass
class AboxImportReport:
    designs: int = 0
    printings: int = 0
    type_assignments: int = 0
    subtype_assignments: int = 0
    qualities: int = 0
    keyword_assignments: int = 0
    skipped: int = 0
    violations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"designs={self.designs} printings={self.printings} "
            f"types={self.type_assignments} subtypes={self.subtype_assignments} "
            f"qualities={self.qualities} keywords={self.keyword_assignments} "
            f"skipped={self.skipped} violations={len(self.violations)}"
        )


# ---------------------------------------------------------------------------
# Shaping one Scryfall record
# ---------------------------------------------------------------------------


def _numeric(value: Any) -> int | None:
    """Return an int only when the printed value is determinate.

    ``"3"`` -> 3; ``"*"``, ``"1+*"``, ``"X"``, ``None`` -> ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return None


def _quality_payload(kind: str, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    lexical = str(value)
    return {
        "kind": kind,
        "regionId": f"{kind.lower()}:{lexical}",
        "lexical": lexical,
        "numeric": _numeric(value),
    }


def shape_card(card: dict[str, Any], ts: TypeSystem) -> dict[str, Any] | None:
    """Project one Scryfall oracle record onto the ontology's shape."""
    name = card.get("name")
    if not name:
        return None

    faces = split_faces(card.get("type_line") or "")
    parsed_faces: list[ParsedTypeLine] = [parse_type_line(f, ts) for f in faces] or [
        parse_type_line("", ts)
    ]

    card_types: list[str] = []
    supertypes: list[str] = []
    subtypes: list[dict[str, str]] = []
    violations: list[str] = []
    for parsed in parsed_faces:
        card_types.extend(t for t in parsed.card_types if t not in card_types)
        supertypes.extend(t for t in parsed.supertypes if t not in supertypes)
        for sub, families in parsed.subtype_families.items():
            if not families:
                violations.append(f"{name}: {sub!r} not licensed by {parsed.card_types}")
                continue
            for family in sorted(families):
                entry = {"name": sub, "family": family}
                if entry not in subtypes:
                    subtypes.append(entry)

    qualities = [
        q
        for q in (
            _quality_payload("Power", card.get("power")),
            _quality_payload("Toughness", card.get("toughness")),
            _quality_payload("Loyalty", card.get("loyalty")),
            _quality_payload("Defense", card.get("defense")),
        )
        if q is not None
    ]

    return {
        "cardName": name,
        "design": {
            "oracleText": card.get("oracle_text", ""),
            "manaCostString": card.get("mana_cost", ""),
            "manaValue": int(card.get("cmc", 0) or 0),
            "typeLine": card.get("type_line", ""),
            "colors": card.get("colors", []),
            "colorIdentity": card.get("color_identity", []),
            "edhrecRank": card.get("edhrec_rank"),
            "epistemicStatus": "curated",
        },
        "printing": {
            "scryfallId": card.get("id", ""),
            "setCode": card.get("set", ""),
            "collectorNumber": card.get("collector_number", ""),
            "rarity": card.get("rarity", ""),
            "epistemicStatus": "curated",
        }
        if card.get("id")
        else None,
        "cardTypes": card_types,
        "supertypes": supertypes,
        "subtypes": subtypes,
        "qualities": qualities,
        "keywords": card.get("keywords", []),
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

MERGE_BATCH = """
UNWIND $rows AS row

MERGE (d:CardDesign {cardName: row.cardName})
SET d += row.design

// CR 205.2a / 205.4a — classification, not labels.
FOREACH (t IN row.cardTypes |
    MERGE (ct:CardType {name: t})
    MERGE (d)-[:HAS_CARD_TYPE]->(ct))
FOREACH (s IN row.supertypes |
    MERGE (st:Supertype {name: s})
    MERGE (d)-[:HAS_SUPERTYPE]->(st))

// CR 205.3 — subtype carries the family it is correlated to.
FOREACH (sub IN row.subtypes |
    MERGE (s:Subtype {name: sub.name, family: sub.family})
    MERGE (d)-[:HAS_SUBTYPE]->(s))

// DOLCE quality/region: the quality inheres, the region carries the value.
FOREACH (q IN row.qualities |
    MERGE (qu:GameQuality {qualityId: row.cardName + ':' + q.kind})
    SET qu.kind = q.kind
    MERGE (d)-[:HAS_QUALITY]->(qu)
    MERGE (r:ValueRegion {regionId: q.regionId})
    SET r.lexical = q.lexical, r.numeric = q.numeric
    MERGE (qu)-[:HAS_REGION]->(r))

FOREACH (k IN row.keywords |
    MERGE (kw:Keyword {name: k})
    MERGE (d)-[:HAS_KEYWORD]->(kw))

// Information realisation: the printing realizes the design.
FOREACH (_ IN CASE WHEN row.printing IS NULL THEN [] ELSE [1] END |
    MERGE (p:CardPrinting {scryfallId: row.printing.scryfallId})
    SET p += row.printing
    MERGE (p)-[:REALIZES]->(d))
"""

CONSTRAINTS = [
    "CREATE CONSTRAINT card_design_name IF NOT EXISTS "
    "FOR (d:CardDesign) REQUIRE d.cardName IS UNIQUE",
    "CREATE CONSTRAINT card_printing_id IF NOT EXISTS "
    "FOR (p:CardPrinting) REQUIRE p.scryfallId IS UNIQUE",
    "CREATE CONSTRAINT value_region_id IF NOT EXISTS "
    "FOR (r:ValueRegion) REQUIRE r.regionId IS UNIQUE",
    "CREATE CONSTRAINT game_quality_id IF NOT EXISTS "
    "FOR (q:GameQuality) REQUIRE q.qualityId IS UNIQUE",
    "CREATE INDEX subtype_name IF NOT EXISTS FOR (s:Subtype) ON (s.name)",
    "CREATE INDEX card_type_name IF NOT EXISTS FOR (t:CardType) ON (t.name)",
]


class AboxBuilder:
    """Writes ontology-shaped card data into Neo4j."""

    def __init__(self, driver: Any = None, type_system: TypeSystem | None = None):
        if driver is None:
            from neo4j import AsyncGraphDatabase

            from src.config import settings

            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
            )
        self.driver = driver
        self.ts = type_system or load_type_system()

    async def close(self) -> None:
        await self.driver.close()

    async def ensure_constraints(self) -> None:
        async with self.driver.session() as session:
            for stmt in CONSTRAINTS:
                await session.run(stmt)

    async def import_cards(
        self, cards: Iterable[dict[str, Any]], batch_size: int = BATCH_SIZE
    ) -> AboxImportReport:
        report = AboxImportReport()
        batch: list[dict[str, Any]] = []

        for card in cards:
            row = shape_card(card, self.ts)
            if row is None:
                report.skipped += 1
                continue
            report.violations.extend(row.pop("violations"))
            batch.append(row)
            report.designs += 1
            report.printings += 1 if row["printing"] else 0
            report.type_assignments += len(row["cardTypes"])
            report.subtype_assignments += len(row["subtypes"])
            report.qualities += len(row["qualities"])
            report.keyword_assignments += len(row["keywords"])

            if len(batch) >= batch_size:
                await self._write(batch)
                batch = []

        if batch:
            await self._write(batch)

        logger.info("ABox import: %s", report.summary())
        if report.violations:
            logger.warning(
                "%d CR 205.3d violations; first: %s",
                len(report.violations),
                report.violations[0],
            )
        return report

    async def import_from_bulk(self, path: str | Path) -> AboxImportReport:
        bulk = Path(path)
        if not bulk.exists():
            raise FileNotFoundError(f"Scryfall bulk data not found: {bulk}")
        cards = json.loads(bulk.read_text(encoding="utf-8"))
        logger.info("Loaded %d cards from %s", len(cards), bulk.name)
        await self.ensure_constraints()
        return await self.import_cards(cards)

    async def _write(self, rows: list[dict[str, Any]]) -> None:
        async with self.driver.session() as session:
            await session.run(MERGE_BATCH, rows=rows)
