"""Tests for knowledge graph queries (requires running Neo4j)."""

import pytest


# These tests require a running Neo4j instance with data loaded.
# Mark them with a custom marker and skip if Neo4j is unavailable.

pytestmark = pytest.mark.skipif(
    True,  # Set to False when Neo4j is available for integration tests
    reason="Neo4j not available for integration tests",
)


class TestKnowledgeGraph:
    """Integration tests for the MTG knowledge graph."""

    @pytest.mark.asyncio
    async def test_get_combos_containing(self):
        from src.knowledge.knowledge_graph import MTGKnowledgeGraph

        kg = MTGKnowledgeGraph()
        try:
            combos = await kg.get_combos_containing("Thassa's Oracle")
            assert isinstance(combos, list)
        finally:
            await kg.close()

    @pytest.mark.asyncio
    async def test_infer_archetype(self):
        from src.knowledge.knowledge_graph import MTGKnowledgeGraph

        kg = MTGKnowledgeGraph()
        try:
            archetypes = await kg.infer_archetype_from_cards(
                ["Sol Ring", "Mana Crypt", "Rhystic Study"]
            )
            assert isinstance(archetypes, list)
        finally:
            await kg.close()
