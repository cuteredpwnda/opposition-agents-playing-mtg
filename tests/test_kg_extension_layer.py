"""Tests for append-only KG extension writes and merged synergy queries."""

from __future__ import annotations

import pytest

from src.knowledge.knowledge_graph import MTGKnowledgeGraph


@pytest.mark.asyncio
async def test_add_synergy_uses_extension_event_nodes():
    kg = MTGKnowledgeGraph.__new__(MTGKnowledgeGraph)
    captured: dict[str, object] = {}

    async def fake_run(query: str, **params):
        captured["query"] = query
        captured["params"] = params
        return []

    kg._run_query = fake_run  # type: ignore[method-assign]

    await kg.add_synergy(
        "Lightning Bolt",
        "Goblin Guide",
        weight=0.4,
        run_id="test_run",
        source="unit_test",
        metadata={"lift": 1.5},
    )

    query = str(captured["query"])
    params = captured["params"]

    assert "LearnedSynergyEvidence" in query
    assert "CREATE (ev:LearnedSynergyEvidence:KGExtensionEvent" in query
    assert "SYNERGIZES_WITH" not in query
    assert params["run_id"] == "test_run"
    assert params["source"] == "unit_test"
    assert params["metadata"]["lift"] == 1.5


@pytest.mark.asyncio
async def test_update_card_stats_writes_outcome_events():
    kg = MTGKnowledgeGraph.__new__(MTGKnowledgeGraph)
    captured: dict[str, object] = {}

    async def fake_run(query: str, **params):
        captured["query"] = query
        captured["params"] = params
        return []

    kg._run_query = fake_run  # type: ignore[method-assign]

    await kg.update_card_stats(
        "Lightning Bolt",
        won=True,
        run_id="stats_run",
        source="unit_test",
        metadata={"gamesObserved": 10},
    )

    query = str(captured["query"])
    params = captured["params"]

    assert "LearnedCardOutcome" in query
    assert "CREATE (ev:LearnedCardOutcome:KGExtensionEvent" in query
    assert "gamesPlayed" not in query
    assert params["won"] is True
    assert params["run_id"] == "stats_run"


@pytest.mark.asyncio
async def test_get_synergies_for_unions_base_and_extension():
    kg = MTGKnowledgeGraph.__new__(MTGKnowledgeGraph)
    captured: dict[str, object] = {}

    async def fake_run(query: str, **params):
        captured["query"] = query
        return []

    kg._run_query = fake_run  # type: ignore[method-assign]

    await kg.get_synergies_for("Lightning Bolt")

    query = str(captured["query"])
    assert "SYNERGIZES_WITH" in query
    assert "LearnedSynergyEvidence" in query
    assert "deterministic_base+learned_extension" in query
