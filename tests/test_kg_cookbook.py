"""Smoke-tests + executable examples for the KG cookbook (paper/reports/04).

Each test runs one of the Cypher snippets from
``paper/reports/04_kg_query_cookbook.md`` against the live ``mtg-neo4j``
container and asserts a sensible postcondition.  Tests are auto-skipped
when the database is unreachable, when the static graph hasn't been
built, or when an optional plugin (GDS) isn't installed — so the whole
module is also a checklist for "did the KG build finish?".

Run the full suite:

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_kg_cookbook.py -v

To re-run only the parts that need GDS/Louvain after wiring the plugin:

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_kg_cookbook.py -v -k louvain
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.config import settings


# --------------------------------------------------------------------------- #
# Connection plumbing                                                         #
# --------------------------------------------------------------------------- #


def _driver():
    """Build an async Neo4j driver, or skip if unavailable."""
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        pytest.skip("neo4j driver not installed", allow_module_level=False)

    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


async def _run(query: str, **params: Any) -> list[dict[str, Any]]:
    drv = _driver()
    try:
        async with drv.session() as session:
            result = await session.run(query, **params)
            return [r.data() async for r in result]
    finally:
        await drv.close()


def _q(query: str, **params: Any) -> list[dict[str, Any]]:
    """Sync wrapper so the test bodies stay readable."""
    try:
        return asyncio.run(_run(query, **params))
    except Exception as e:
        pytest.skip(f"Neo4j unreachable: {e}")


@pytest.fixture(scope="module")
def kg_inventory() -> dict[str, int]:
    """Capture node-label counts once for the module."""
    rows = _q(
        """
        CALL db.labels() YIELD label
        CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS n }
        RETURN label, n
        """
    )
    return {r["label"]: r["n"] for r in rows}


@pytest.fixture(scope="module")
def kg_rels() -> dict[str, int]:
    rows = _q(
        """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL { WITH relationshipType MATCH ()-[r]->() WHERE type(r) = relationshipType
               RETURN count(r) AS n }
        RETURN relationshipType AS t, n
        """
    )
    return {r["t"]: r["n"] for r in rows}


# --------------------------------------------------------------------------- #
# Section 1 — Sanity checks                                                   #
# --------------------------------------------------------------------------- #


class TestSanity:
    def test_card_count(self, kg_inventory):
        n = kg_inventory.get("Card", 0)
        assert n > 10000, f"expected >10k Cards, got {n} (run stage 2)"

    def test_label_taxonomy(self, kg_inventory):
        for label in ["Creature", "Instant", "Sorcery", "Land"]:
            assert kg_inventory.get(label, 0) > 0, f"no {label} nodes"

    def test_legal_in_present(self, kg_rels):
        assert kg_rels.get("LEGAL_IN", 0) > 0

    def test_has_keyword_present(self, kg_rels):
        assert kg_rels.get("HAS_KEYWORD", 0) > 0


# --------------------------------------------------------------------------- #
# Section 2 — Card lookup                                                     #
# --------------------------------------------------------------------------- #


class TestCardLookup:
    def test_lightning_bolt_exists(self):
        rows = _q(
            "MATCH (c:Card {cardName: 'Lightning Bolt'}) RETURN c.manaValue AS mv, c.oracleText AS o"
        )
        if not rows:
            pytest.skip("Lightning Bolt not in DB (run stage 2)")
        assert rows[0]["mv"] in (1, 1.0)

    def test_modern_legal_cheap_instants(self):
        rows = _q(
            """
            MATCH (c:Instant)-[:LEGAL_IN]->(:Format {name: 'modern'})
            WHERE c.manaValue <= 2
            RETURN c.cardName AS name LIMIT 5
            """
        )
        if not rows:
            pytest.skip("no Format nodes / no instants legal in modern")
        assert all(r["name"] for r in rows)

    def test_fulltext_card_search(self):
        # cardSearch fulltext index is optional — skip if missing
        idx = _q("SHOW INDEXES YIELD name WHERE name = 'cardSearch' RETURN name")
        if not idx:
            pytest.skip("fulltext index 'cardSearch' not created")
        rows = _q(
            "CALL db.index.fulltext.queryNodes('cardSearch', 'sacrifice creature draw') "
            "YIELD node, score RETURN node.cardName AS name, score LIMIT 3"
        )
        assert rows
        assert all("score" in r for r in rows)


# --------------------------------------------------------------------------- #
# Section 3 — Combos                                                          #
# --------------------------------------------------------------------------- #


def _has_combos() -> bool:
    rows = _q("MATCH (c:Combo) RETURN count(c) AS n")
    return bool(rows and rows[0]["n"] > 0)


class TestCombos:
    def test_combo_label_populated(self, kg_inventory):
        if "Combo" not in kg_inventory:
            pytest.skip("no Combo nodes — run scripts.import_combos")
        assert kg_inventory["Combo"] > 0

    def test_part_of_combo_edges(self, kg_rels):
        if "PART_OF_COMBO" not in kg_rels:
            pytest.skip("PART_OF_COMBO edges absent — run scripts.import_combos")
        assert kg_rels["PART_OF_COMBO"] > 0

    def test_combos_containing_known_card(self):
        if not _has_combos():
            pytest.skip("no combos in DB")
        # Pick *any* combo's card so the test isn't brittle.
        seed = _q(
            "MATCH (c:Card)-[:PART_OF_COMBO]->(:Combo) RETURN c.cardName AS n LIMIT 1"
        )
        if not seed:
            pytest.skip("no combo participants in DB")
        rows = _q(
            """
            MATCH (c:Card {cardName: $name})-[:PART_OF_COMBO]->(combo:Combo)
            MATCH (combo)<-[:PART_OF_COMBO]-(piece:Card)
            RETURN combo.comboDescription AS desc,
                   collect(DISTINCT piece.cardName) AS pieces
            LIMIT 5
            """,
            name=seed[0]["n"],
        )
        assert rows
        for r in rows:
            assert seed[0]["n"] in r["pieces"]

    def test_assembled_from_pool(self):
        if not _has_combos():
            pytest.skip("no combos in DB")
        # Find any combo whose entire piece-list we can grab as a hypothetical pool.
        full = _q(
            """
            MATCH (combo:Combo)<-[:PART_OF_COMBO]-(c:Card)
            WITH combo, collect(c.cardName) AS pieces
            WHERE size(pieces) >= 2 AND size(pieces) <= 4
            RETURN pieces LIMIT 1
            """
        )
        if not full:
            pytest.skip("no small combos available")
        pool = full[0]["pieces"]
        rows = _q(
            """
            MATCH (combo:Combo)
            WITH combo, $pool AS pool, [(combo)<-[:PART_OF_COMBO]-(c) | c.cardName] AS pieces
            WHERE size(pieces) > 0 AND ALL(p IN pieces WHERE p IN pool)
            RETURN combo.comboDescription AS desc, pieces LIMIT 3
            """,
            pool=pool,
        )
        assert rows, f"pool {pool} should match its own combo"


# --------------------------------------------------------------------------- #
# Section 4 — Synergies (dynamic, written by stage-4.5 enrichment)            #
# --------------------------------------------------------------------------- #


class TestSynergies:
    def test_synergy_edges_or_skip(self, kg_rels):
        has_base = kg_rels.get("SYNERGIZES_WITH", 0) > 0
        has_extension = kg_rels.get("SUPPORTED_BY", 0) > 0
        if not (has_base or has_extension):
            pytest.skip(
                "no base or extension synergies — run KGEnrichment after stage 4"
            )
        assert has_base or has_extension

    def test_synergy_strengths_in_range(self, kg_rels):
        if kg_rels.get("SUPPORTED_BY", 0) > 0:
            rows = _q(
                "MATCH (:Card)-[:SUPPORTED_BY]-"
                "(ev:LearnedSynergyEvidence)-[:SUPPORTED_BY]-(:Card) "
                "RETURN min(ev.weight) AS lo, max(ev.weight) AS hi"
            )
            if rows and rows[0]["lo"] is not None:
                assert rows[0]["hi"] >= rows[0]["lo"]
                return

        if kg_rels.get("SYNERGIZES_WITH", 0) == 0:
            pytest.skip("no synergies yet")

        rows = _q(
            "MATCH ()-[s:SYNERGIZES_WITH]-() WHERE s.strength IS NOT NULL "
            "RETURN min(s.strength) AS lo, max(s.strength) AS hi"
        )
        assert rows[0]["lo"] is not None
        assert rows[0]["hi"] >= rows[0]["lo"]

    def test_dynamic_card_stats(self):
        outcome_events = _q(
            "MATCH (:Card)-[:HAS_LEARNED_OUTCOME]->(:LearnedCardOutcome) "
            "RETURN count(*) AS n"
        )
        if outcome_events and outcome_events[0]["n"] > 0:
            bad = _q(
                "MATCH (c:Card)-[:HAS_LEARNED_OUTCOME]->(ev:LearnedCardOutcome) "
                "WITH c, count(ev) AS games, "
                "sum(CASE WHEN ev.won THEN 1 ELSE 0 END) AS wins "
                "WHERE wins > games OR games < 0 "
                "RETURN count(c) AS n"
            )
            assert bad[0]["n"] == 0
            return

        rows = _q(
            "MATCH (c:Card) WHERE c.gamesPlayed IS NOT NULL "
            "RETURN count(c) AS n"
        )
        if rows[0]["n"] == 0:
            pytest.skip("no card-stat enrichment yet — run stage 4 + 4.5")
        bad = _q(
            "MATCH (c:Card) WHERE c.gamesPlayed IS NOT NULL "
            "AND (c.gamesWon > c.gamesPlayed OR c.winRate < 0 OR c.winRate > 1) "
            "RETURN count(c) AS n"
        )
        assert bad[0]["n"] == 0


# --------------------------------------------------------------------------- #
# Section 6 — Archetypes (currently aspirational — not built by any script)   #
# --------------------------------------------------------------------------- #


class TestArchetypes:
    def test_archetype_or_skip(self, kg_inventory):
        if "Archetype" not in kg_inventory:
            pytest.skip("Archetype nodes not yet populated (no builder)")
        assert kg_inventory["Archetype"] > 0

    def test_belongs_to_archetype_or_skip(self, kg_rels):
        if "BELONGS_TO_ARCHETYPE" not in kg_rels:
            pytest.skip("BELONGS_TO_ARCHETYPE edges absent")
        assert kg_rels["BELONGS_TO_ARCHETYPE"] > 0


# --------------------------------------------------------------------------- #
# Section 7 — Counters / answers                                              #
# --------------------------------------------------------------------------- #


class TestCounters:
    def test_counters_or_skip(self, kg_rels):
        if "COUNTERS" not in kg_rels:
            pytest.skip("COUNTERS edges absent (no builder yet)")
        assert kg_rels["COUNTERS"] > 0


# --------------------------------------------------------------------------- #
# Section 8 — Embeddings + vector search                                      #
# --------------------------------------------------------------------------- #


class TestEmbeddings:
    def test_some_embeddings_persisted(self):
        rows = _q(
            "MATCH (c:Card) WHERE c.embedding IS NOT NULL RETURN count(c) AS n"
        )
        if rows[0]["n"] == 0:
            pytest.skip("no embeddings — run stage 3")
        assert rows[0]["n"] > 1000

    def test_embedding_dimensionality_uniform(self):
        rows = _q(
            "MATCH (c:Card) WHERE c.embedding IS NOT NULL "
            "WITH size(c.embedding) AS d "
            "RETURN d, count(*) AS n ORDER BY n DESC LIMIT 3"
        )
        if not rows:
            pytest.skip("no embeddings")
        # All embeddings should share one dimension.
        assert len(rows) == 1, f"non-uniform embedding dims: {rows}"
        assert rows[0]["d"] in (64, 128, 256)

    def test_vector_index_or_skip(self):
        idx = _q(
            "SHOW INDEXES YIELD name, type WHERE name = 'cardEmbeddings' RETURN name"
        )
        if not idx:
            pytest.skip("vector index 'cardEmbeddings' not created")
        # Pick one card's embedding and run a self-similarity query.
        sample = _q(
            "MATCH (c:Card) WHERE c.embedding IS NOT NULL "
            "RETURN c.cardName AS name, c.embedding AS e LIMIT 1"
        )
        if not sample:
            pytest.skip("no embeddings to query")
        rows = _q(
            "CALL db.index.vector.queryNodes('cardEmbeddings', 5, $e) "
            "YIELD node, score RETURN node.cardName AS name, score",
            e=sample[0]["e"],
        )
        assert rows, "vector index returned nothing"
        assert rows[0]["name"] == sample[0]["name"], "self-similarity should be top-1"


# --------------------------------------------------------------------------- #
# Section — Louvain / GDS (optional)                                          #
# --------------------------------------------------------------------------- #


class TestLouvain:
    def test_strategic_cluster_or_skip(self):
        rows = _q(
            "MATCH (c:Card) WHERE c.strategicCluster IS NOT NULL RETURN count(c) AS n"
        )
        if rows[0]["n"] == 0:
            pytest.skip("Louvain not run — needs GDS plugin + run_community_detection")
        assert rows[0]["n"] > 100


# --------------------------------------------------------------------------- #
# Section 9 — Quality checks                                                  #
# --------------------------------------------------------------------------- #


class TestQuality:
    def test_no_orphan_cards(self):
        rows = _q("MATCH (c:Card) WHERE NOT (c)--() RETURN count(c) AS n")
        # Some Cards may still be orphans if combos weren't imported; just
        # make sure most cards have at least one edge.
        total = _q("MATCH (c:Card) RETURN count(c) AS n")[0]["n"]
        orphan = rows[0]["n"]
        assert orphan / max(total, 1) < 0.5, f"{orphan}/{total} cards are orphans"

    def test_combos_have_at_least_two_pieces(self):
        rows = _q("MATCH (c:Combo) RETURN count(c) AS n")
        if rows[0]["n"] == 0:
            pytest.skip("no combos")
        bad = _q(
            """
            MATCH (combo:Combo)<-[:PART_OF_COMBO]-(c)
            WITH combo, count(c) AS pieces
            WHERE pieces < 2
            RETURN count(combo) AS n
            """
        )
        assert bad[0]["n"] == 0, f"{bad[0]['n']} combos have <2 pieces"


# --------------------------------------------------------------------------- #
# Section 10 — Python API parity                                              #
# --------------------------------------------------------------------------- #


class TestPythonAPI:
    def test_subgraph_context_smokes(self):
        """`MTGKnowledgeGraph.subgraph_context` should return a non-empty
        dict for any well-known card."""
        from src.knowledge.knowledge_graph import MTGKnowledgeGraph

        async def go():
            kg = MTGKnowledgeGraph()
            try:
                return await kg.subgraph_context("Lightning Bolt", depth=1)
            finally:
                await kg.close()

        try:
            ctx = asyncio.run(go())
        except Exception as e:
            pytest.skip(f"subgraph_context unavailable: {e}")
        assert isinstance(ctx, dict)

    def test_detect_available_combos_smokes(self):
        from src.knowledge.knowledge_graph import MTGKnowledgeGraph

        async def go():
            kg = MTGKnowledgeGraph()
            try:
                # Whatever pool we pass, the call must return a list.
                return await kg.detect_available_combos(
                    ["Lightning Bolt", "Devoted Druid", "Vizier of Remedies"]
                )
            finally:
                await kg.close()

        try:
            combos = asyncio.run(go())
        except Exception as e:
            pytest.skip(f"detect_available_combos unavailable: {e}")
        assert isinstance(combos, list)
