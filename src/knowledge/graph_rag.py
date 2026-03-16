"""
GraphRAG-style retrieval implemented natively on Neo4j.

Instead of running a separate GraphRAG pipeline (e.g., microsoft/graphrag),
we use Neo4j's native capabilities:
- APOC: subgraph expansion, community detection, PageRank
- n10s:  ontology-aware expansion (class hierarchy traversal)
- Neo4j 5.x: native vector index for semantic search
- Lucene: full-text index over card text

Optionally, GraphQAAgent (MIT — https://github.com/DataScienceLabFHSWF/GraphQAAgent)
can be layered on top for CoT reasoning + answer verification.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase

from src.config import settings

logger = logging.getLogger(__name__)


class MTGGraphRAG:
    """GraphRAG-style retrieval over the MTG knowledge graph in Neo4j.

    Strategies:
        subgraph  — APOC path expansion around an entity
        cypher    — LLM generates Cypher from natural language question
        vector    — Neo4j native vector similarity search
        fulltext  — Lucene full-text search over card text
        community — APOC community detection + cluster lookup
        hybrid    — Combine multiple strategies with RRF
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
    # 1. Subgraph retrieval (APOC)
    # -----------------------------------------------------------------------

    async def subgraph_retrieval(
        self, card_name: str, depth: int = 2
    ) -> dict[str, Any]:
        """APOC subgraph expansion — full neighborhood in N hops."""
        query = """
        MATCH (start:Card {cardName: $card_name})
        CALL apoc.path.subgraphAll(start, {
          maxLevel: $depth,
          relationshipFilter:
            'PART_OF_COMBO|SYNERGIZES_WITH|COUNTERS|ENABLES|'
            + 'BELONGS_TO_ARCHETYPE|HAS_WIN_CONDITION|HAS_KEYWORD'
        }) YIELD nodes, relationships
        RETURN nodes, relationships
        """
        async with self.driver.session() as session:
            result = await session.run(query, card_name=card_name, depth=depth)
            record = await result.single()
            if record is None:
                return {"nodes": [], "relationships": []}
            return {
                "nodes": [
                    {
                        "labels": list(n.labels),
                        "properties": dict(n),
                    }
                    for n in record["nodes"]
                ],
                "relationships": [
                    {
                        "type": r.type,
                        "start": r.start_node["cardName"]
                        if "cardName" in r.start_node
                        else str(r.start_node.id),
                        "end": r.end_node["cardName"]
                        if "cardName" in r.end_node
                        else str(r.end_node.id),
                    }
                    for r in record["relationships"]
                ],
            }

    # -----------------------------------------------------------------------
    # 2. Vector retrieval (Neo4j native)
    # -----------------------------------------------------------------------

    async def vector_retrieval(
        self, query_embedding: list[float], k: int = 10
    ) -> list[dict[str, Any]]:
        """Neo4j native vector index search — find similar cards."""
        query = """
        CALL db.index.vector.queryNodes('cardEmbeddings', $k, $embedding)
        YIELD node, score
        RETURN node.cardName AS card, node.oracleText AS text, score
        """
        async with self.driver.session() as session:
            result = await session.run(query, k=k, embedding=query_embedding)
            return [dict(r) async for r in result]

    # -----------------------------------------------------------------------
    # 3. Full-text retrieval (Lucene)
    # -----------------------------------------------------------------------

    async def fulltext_retrieval(
        self, search_text: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Lucene full-text search over card names and oracle text."""
        query = """
        CALL db.index.fulltext.queryNodes('cardSearch', $text)
        YIELD node, score
        RETURN node.cardName AS card, node.oracleText AS text, score
        LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(query, text=search_text, limit=limit)
            return [dict(r) async for r in result]

    # -----------------------------------------------------------------------
    # 4. Community retrieval (APOC Louvain)
    # -----------------------------------------------------------------------

    async def community_retrieval(self, card_name: str) -> dict[str, Any]:
        """Get the strategic cluster a card belongs to."""
        query = """
        MATCH (c:Card {cardName: $card_name})
        WITH c.strategicCluster AS cluster
        MATCH (member:Card {strategicCluster: cluster})
        RETURN cluster,
               collect(member.cardName)[..20] AS members,
               count(member) AS size
        """
        async with self.driver.session() as session:
            result = await session.run(query, card_name=card_name)
            record = await result.single()
            return dict(record) if record else {}

    # -----------------------------------------------------------------------
    # 5. Matchup analysis (graph traversal)
    # -----------------------------------------------------------------------

    async def matchup_analysis(
        self, my_archetype: str, opp_archetype: str
    ) -> dict[str, Any]:
        """Archetype matchup analysis using graph traversal."""
        query = """
        MATCH (me:Archetype {name: $my_arch})
        MATCH (opp:Archetype {name: $opp_arch})
        OPTIONAL MATCH (me)-[w:WEAK_AGAINST]->(opp)
        OPTIONAL MATCH (me)-[s:STRONG_AGAINST]->(opp)
        OPTIONAL MATCH (me)-[:HAS_WIN_CONDITION]->(myWin:Combo)
        OPTIONAL MATCH (opp)-[:HAS_WIN_CONDITION]->(oppWin:Combo)
        OPTIONAL MATCH (answer:Card)-[:COUNTERS]->(oppKey:Card)
                       <-[:BELONGS_TO_ARCHETYPE]-(opp)
        WHERE (answer)-[:BELONGS_TO_ARCHETYPE]->(me)
        RETURN me.name AS myArchetype, opp.name AS oppArchetype,
               w IS NOT NULL AS isUnfavorable,
               s IS NOT NULL AS isFavorable,
               collect(DISTINCT myWin.comboDescription) AS myWinCons,
               collect(DISTINCT oppWin.comboDescription) AS oppWinCons,
               collect(DISTINCT answer.cardName) AS keyAnswers
        """
        async with self.driver.session() as session:
            result = await session.run(
                query, my_arch=my_archetype, opp_arch=opp_archetype
            )
            record = await result.single()
            return dict(record) if record else {}

    # -----------------------------------------------------------------------
    # 6. Combo finding
    # -----------------------------------------------------------------------

    async def find_combos(
        self, available_cards: list[str]
    ) -> list[dict[str, Any]]:
        """Detect available combos from hand + battlefield."""
        query = """
        MATCH (combo:Combo)
        WITH combo, [(combo)<-[:PART_OF_COMBO]-(c:Card) | c.cardName] AS pieces
        WHERE ALL(piece IN pieces WHERE piece IN $available)
        MATCH (combo)<-[:PART_OF_COMBO]-(c:Card)
        OPTIONAL MATCH (combo)-[:PRODUCES_EFFECT]->(e:Effect)
        RETURN combo.comboDescription AS description,
               collect(DISTINCT c.cardName) AS pieces,
               collect(DISTINCT e.name) AS effects
        """
        async with self.driver.session() as session:
            result = await session.run(query, available=available_cards)
            return [dict(r) async for r in result]

    # -----------------------------------------------------------------------
    # APOC graph algorithm primitives
    # -----------------------------------------------------------------------

    async def run_community_detection(self) -> int:
        """Run Louvain community detection and write clusters to nodes.

        Returns number of communities found.
        """
        query = """
        CALL apoc.periodic.iterate(
          'MATCH (c:Card) RETURN c',
          'CALL apoc.algo.louvain(null, null, {})
           YIELD nodeId, community
           MATCH (c) WHERE id(c) = nodeId
           SET c.strategicCluster = community',
          {batchSize: 1000}
        )
        """
        async with self.driver.session() as session:
            await session.run(query)
            result = await session.run(
                "MATCH (c:Card) RETURN count(DISTINCT c.strategicCluster) AS communities"
            )
            record = await result.single()
            count = record["communities"] if record else 0
            logger.info(f"Community detection complete: {count} communities")
            return count

    async def run_pagerank(self) -> None:
        """Run PageRank and write importance scores to nodes."""
        query = """
        CALL apoc.algo.pageRank(null, null, {iterations: 20, dampingFactor: 0.85})
        YIELD nodeId, score
        MATCH (c:Card) WHERE id(c) = nodeId
        SET c.metagameImportance = score
        """
        async with self.driver.session() as session:
            await session.run(query)
            logger.info("PageRank scores written to Card nodes")
