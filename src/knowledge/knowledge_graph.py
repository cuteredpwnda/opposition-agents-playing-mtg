"""
Neo4j-backed MTG Knowledge Graph.

All strategic knowledge lives in Neo4j, with:
- n10s for OWL ontology schema (TBox)
- APOC for graph algorithms (PageRank, community detection, path expansion)
- Native vector indexes for semantic similarity search

Reference: KGPlatform (MIT) — https://github.com/DataScienceLabFHSWF/KGPlatform
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase

from src.config import settings

logger = logging.getLogger(__name__)


class MTGKnowledgeGraph:
    """Neo4j-backed knowledge graph for MTG strategic reasoning.

    Node labels (from OWL): Card, Creature, Instant, Sorcery, Enchantment,
        Artifact, Planeswalker, Land, Combo, Archetype, Keyword, Effect, ...
    Relationship types (from OWL): PART_OF_COMBO, SYNERGIZES_WITH, COUNTERS,
        ENABLES, HAS_KEYWORD, BELONGS_TO_ARCHETYPE, LEGAL_IN, ...
    """

    def __init__(
        self,
        uri: str = settings.neo4j_uri,
        user: str = settings.neo4j_user,
        password: str = settings.neo4j_password,
    ):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self.driver.close()

    # -----------------------------------------------------------------------
    # Combo queries
    # -----------------------------------------------------------------------

    async def get_combos_containing(self, card_name: str) -> list[dict[str, Any]]:
        """Find all combos that include a given card.

        Returns the structured combo data: ``comboName`` (joined card
        names), ``pieces``, ``outcomeCategories`` / ``outcomeMagnitudes``
        from the new ontology, plus a flat ``outcomes`` list with
        per-outcome (category, magnitude, displayName, feature) records.
        """
        query = """
        MATCH (c:Card {cardName: $card_name})-[:PART_OF_COMBO]->(combo:Combo)
        OPTIONAL MATCH (combo)<-[:PART_OF_COMBO]-(piece:Card)
        OPTIONAL MATCH (combo)-[r:PRODUCES]->(o:Outcome)
        RETURN combo.comboId AS comboId,
               combo.comboName AS comboName,
               combo.comboDescription AS description,
               combo.outcomeCategories AS outcomeCategories,
               combo.outcomeMagnitudes AS outcomeMagnitudes,
               combo.cardCount AS cardCount,
               collect(DISTINCT piece.cardName) AS pieces,
               collect(DISTINCT {
                   category: o.category,
                   magnitude: o.magnitude,
                   displayName: o.displayName,
                   feature: r.feature
               }) AS outcomes
        ORDER BY combo.cardCount ASC
        """
        return await self._run_query(query, card_name=card_name)

    async def detect_available_combos(
        self, available_cards: list[str]
    ) -> list[dict[str, Any]]:
        """Find combos where ALL pieces are in the available card list.

        Returns rich payloads identical in shape to
        :meth:`get_combos_containing` so callers can render names,
        outcome categories, and magnitudes uniformly.
        """
        if not available_cards:
            return []
        query = """
        MATCH (combo:Combo)
        WITH combo, [(combo)<-[:PART_OF_COMBO]-(c:Card) | c.cardName] AS pieces
        WHERE size(pieces) > 0 AND ALL(piece IN pieces WHERE piece IN $available)
        OPTIONAL MATCH (combo)-[r:PRODUCES]->(o:Outcome)
        RETURN combo.comboId AS comboId,
               combo.comboName AS comboName,
               combo.comboDescription AS description,
               combo.outcomeCategories AS outcomeCategories,
               combo.outcomeMagnitudes AS outcomeMagnitudes,
               combo.cardCount AS cardCount,
               pieces,
               collect(DISTINCT {
                   category: o.category,
                   magnitude: o.magnitude,
                   displayName: o.displayName,
                   feature: r.feature
               }) AS outcomes
        ORDER BY combo.cardCount ASC
        """
        return await self._run_query(query, available=available_cards)

    async def detect_near_combos(
        self, available_cards: list[str]
    ) -> list[dict[str, Any]]:
        """Find combos where all pieces except one are available."""
        if not available_cards:
            return []
        query = """
        MATCH (combo:Combo)
        WITH combo, [(combo)<-[:PART_OF_COMBO]-(c:Card) | c.cardName] AS pieces
        WITH combo, pieces,
             [p IN pieces WHERE NOT p IN $available] AS missing
        WHERE size(pieces) > 1 AND size(missing) = 1
        OPTIONAL MATCH (combo)-[r:PRODUCES]->(o:Outcome)
        RETURN combo.comboId AS comboId,
               combo.comboName AS comboName,
               combo.comboDescription AS description,
               combo.outcomeCategories AS outcomeCategories,
               combo.outcomeMagnitudes AS outcomeMagnitudes,
               combo.cardCount AS cardCount,
               pieces,
               missing[0] AS missingPiece,
               collect(DISTINCT {
                   category: o.category,
                   magnitude: o.magnitude,
                   displayName: o.displayName,
                   feature: r.feature
               }) AS outcomes
        ORDER BY combo.cardCount ASC
        """
        return await self._run_query(query, available=available_cards)

    # -----------------------------------------------------------------------
    # Outcome-driven combo queries (new ontology)
    # -----------------------------------------------------------------------

    async def get_combos_by_outcome(
        self,
        category: str,
        magnitude: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find all combos producing a given outcome category.

        Example::

            await kg.get_combos_by_outcome("mana", magnitude="infinite")

        ``magnitude`` is one of ``infinite`` / ``arbitrary`` /
        ``near_infinite`` / ``finite``; if ``None`` all magnitudes match.
        """
        query = """
        MATCH (combo:Combo)-[:PRODUCES]->(o:Outcome)
        WHERE o.category = $category
          AND ($magnitude IS NULL OR o.magnitude = $magnitude)
        WITH combo, o
        OPTIONAL MATCH (combo)<-[:PART_OF_COMBO]-(c:Card)
        RETURN combo.comboId AS comboId,
               combo.comboName AS comboName,
               combo.cardCount AS cardCount,
               o.displayName AS outcomeName,
               o.category AS category,
               o.magnitude AS magnitude,
               collect(DISTINCT c.cardName) AS pieces
        ORDER BY combo.cardCount ASC
        LIMIT $limit
        """
        return await self._run_query(
            query, category=category, magnitude=magnitude, limit=limit
        )

    async def get_related_combos_by_outcome(
        self, combo_id: str, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Find combos sharing at least one outcome with ``combo_id``."""
        query = """
        MATCH (src:Combo {comboId: $combo_id})-[:PRODUCES]->(o:Outcome)
              <-[:PRODUCES]-(other:Combo)
        WHERE other.comboId <> $combo_id
        WITH other, collect(DISTINCT o.displayName) AS sharedOutcomes
        OPTIONAL MATCH (other)<-[:PART_OF_COMBO]-(c:Card)
        RETURN other.comboId AS comboId,
               other.comboName AS comboName,
               other.cardCount AS cardCount,
               sharedOutcomes,
               collect(DISTINCT c.cardName) AS pieces
        ORDER BY size(sharedOutcomes) DESC, other.cardCount ASC
        LIMIT $limit
        """
        return await self._run_query(query, combo_id=combo_id, limit=limit)

    async def list_outcome_categories(self) -> list[dict[str, Any]]:
        """Inventory all outcome buckets and how many combos they cover."""
        query = """
        MATCH (combo:Combo)-[:PRODUCES]->(o:Outcome)
        RETURN o.outcomeId AS outcomeId,
               o.category AS category,
               o.magnitude AS magnitude,
               o.displayName AS displayName,
               count(DISTINCT combo) AS comboCount
        ORDER BY comboCount DESC
        """
        return await self._run_query(query)

    # -----------------------------------------------------------------------
    # Synergy & counter-play queries
    # -----------------------------------------------------------------------

    async def get_synergies_for(self, card_name: str) -> list[dict[str, Any]]:
        """Find cards that synergize with a given card."""
        query = """
        MATCH (c:Card {cardName: $card_name})-[s:SYNERGIZES_WITH]-(other:Card)
        RETURN other.cardName AS card,
               s.strength AS strength,
               s.description AS description
        ORDER BY s.strength DESC
        """
        return await self._run_query(query, card_name=card_name)

    async def get_answers_to(self, card_name: str) -> list[dict[str, Any]]:
        """Find cards that counter/answer a given card."""
        query = """
        MATCH (answer:Card)-[:COUNTERS]->(threat:Card {cardName: $card_name})
        OPTIONAL MATCH (answer)-[:LEGAL_IN]->(f:Format)
        RETURN answer.cardName AS card,
               answer.manaCostText AS cost,
               answer.typeLine AS type,
               collect(DISTINCT f.name) AS formats
        """
        return await self._run_query(query, card_name=card_name)

    # -----------------------------------------------------------------------
    # Archetype queries
    # -----------------------------------------------------------------------

    async def infer_archetype_from_cards(
        self, cards_seen: list[str]
    ) -> list[dict[str, Any]]:
        """Given cards observed, rank likely archetypes."""
        if not cards_seen:
            return []
        query = """
        UNWIND $cards AS cardName
        MATCH (c:Card {cardName: cardName})-[:BELONGS_TO_ARCHETYPE]->(a:Archetype)
        WITH a, count(DISTINCT cardName) AS matchCount
        RETURN a.name AS archetype,
               matchCount,
               toFloat(matchCount) / size($cards) AS confidence
        ORDER BY confidence DESC
        LIMIT 5
        """
        results = await self._run_query(query, cards=cards_seen)
        # If no archetype matches, return empty list
        return results if results else []

    async def get_archetype_signature_cards(
        self, archetype_name: str
    ) -> list[dict[str, Any]]:
        """Get the signature/staple cards for an archetype."""
        query = """
        MATCH (a:Archetype {name: $name})<-[:BELONGS_TO_ARCHETYPE]-(c:Card)
        RETURN c.cardName AS card, c.manaCostText AS cost, c.typeLine AS type
        """
        return await self._run_query(query, name=archetype_name)

    # -----------------------------------------------------------------------
    # APOC-powered graph exploration
    # -----------------------------------------------------------------------

    async def subgraph_context(
        self, card_name: str, depth: int = 2
    ) -> dict[str, Any]:
        """APOC subgraph expansion — get strategic context around a card.

        Returns the local neighborhood: combos, synergies, counters, archetypes
        within `depth` hops.
        """
        query = """
        MATCH (start:Card {cardName: $card_name})
        CALL apoc.path.subgraphAll(start, {
          maxLevel: $depth,
          relationshipFilter:
            'PART_OF_COMBO|SYNERGIZES_WITH|COUNTERS|ENABLES|BELONGS_TO_ARCHETYPE|HAS_KEYWORD'
        }) YIELD nodes, relationships
        RETURN
          [n IN nodes |
            labels(n)[0] + ': ' + coalesce(n.cardName, n.name, n.comboDescription, '')
          ] AS entities,
          [r IN relationships | type(r)] AS edge_types
        """
        results = await self._run_query(query, card_name=card_name, depth=depth)
        return results[0] if results else {}

    # -----------------------------------------------------------------------
    # Composite strategic context (used by agents)
    # -----------------------------------------------------------------------

    async def get_strategic_context(
        self,
        hand: list[str],
        battlefield: list[str],
        opponent_cards: list[str],
    ) -> dict[str, Any]:
        """Get full strategic context for agent decision-making."""
        available = list(set(hand + battlefield))
        combos = await self.detect_available_combos(available)
        near = await self.detect_near_combos(available)

        threats = []
        for card in opponent_cards[:5]:  # Limit to 5 opponent cards for perf
            answers = await self.get_answers_to(card)
            if answers:
                threats.append({"threat": card, "answers": answers[:3]})

        return {
            "available_combos": combos[:10],
            "near_combos": near[:10],
            "threat_answers": threats,
        }

    # -----------------------------------------------------------------------
    # Auto-enrichment support (from self-play)
    # -----------------------------------------------------------------------

    async def add_synergy(self, card_a: str, card_b: str, weight: float = 1.0) -> None:
        """Add or boost a SYNERGIZES_WITH relationship between two cards."""
        query = """
        MERGE (a:Card {cardName: $card_a})
        MERGE (b:Card {cardName: $card_b})
        MERGE (a)-[r:SYNERGIZES_WITH]-(b)
        ON CREATE SET r.strength = $weight
        ON MATCH SET r.strength = coalesce(r.strength, 0.0) + $weight
        """
        await self._run_query(query, card_a=card_a, card_b=card_b, weight=weight)

    async def update_card_stats(self, card_name: str, won: bool) -> None:
        """Update win/loss statistics for a card."""
        query = """
        MERGE (c:Card {cardName: $card_name})
        SET c.gamesPlayed = coalesce(c.gamesPlayed, 0) + 1,
            c.gamesWon = coalesce(c.gamesWon, 0) + CASE WHEN $won THEN 1 ELSE 0 END,
            c.winRate = toFloat(coalesce(c.gamesWon, 0)) / coalesce(c.gamesPlayed, 1)
        """
        await self._run_query(query, card_name=card_name, won=won)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    async def _run_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            return [dict(record) async for record in result]
