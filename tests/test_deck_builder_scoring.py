"""Tests for the deck-builder card scorer (G2 / G10).

Uses a stubbed KG to avoid Neo4j.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.agents.deck_builder.scorer import CardScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card(name: str, oracle: str = "", type_line: str = "Instant") -> dict:
    return {
        "name": name,
        "oracle_text": oracle,
        "type_line": type_line,
        "color_identity": [],
        "cmc": 2.0,
        "legalities": {"commander": "legal"},
    }


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Archetype score (synchronous, no KG)
# ---------------------------------------------------------------------------


def test_archetype_basic_land_zero():
    scorer = CardScorer()
    score = scorer._archetype_score(
        _card("Forest", type_line="Basic Land — Forest")
    )
    assert score == 0.0


def test_archetype_nonbasic_land():
    scorer = CardScorer()
    score = scorer._archetype_score(_card("Command Tower", type_line="Land"))
    assert score == pytest.approx(0.3)


def test_archetype_mana_rock():
    scorer = CardScorer()
    card = _card("Sol Ring", oracle="{t}: add {c}{c}")
    assert scorer._archetype_score(card) > 0.5


def test_archetype_ramp_spell():
    scorer = CardScorer()
    card = _card("Cultivate", oracle="search your library for a basic land")
    score = scorer._archetype_score(card)
    assert score > 0.5


def test_archetype_default_spell():
    scorer = CardScorer()
    card = _card("Wrath of God", oracle="destroy all creatures")
    assert scorer._archetype_score(card) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full score with no KG → only archetype + WM (WM=0)
# ---------------------------------------------------------------------------


def test_score_no_kg():
    scorer = CardScorer()
    card = _card("Lightning Bolt", oracle="deals 3 damage")
    result = run(scorer.score(card, ["Sol Ring", "Swamp"]))
    # Only archetype contributes (beta * 1.0)
    expected = scorer.beta * scorer._archetype_score(card)
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Synergy score with stubbed KG
# ---------------------------------------------------------------------------


def _make_kg_with_synergies(synergies: dict[str, list[dict]]):
    """Create an async-capable KG stub."""
    kg = AsyncMock()

    async def get_synergies_for(card_name: str):
        return synergies.get(card_name, [])

    kg.get_synergies_for = get_synergies_for
    kg.detect_near_combos = AsyncMock(return_value=[])
    return kg


def test_synergy_score_with_kg():
    synergies = {
        "Goblin Chieftain": [
            {"card": "Goblin Guide", "strength": 0.9},
            {"card": "Krenko, Mob Boss", "strength": 1.2},
        ]
    }
    kg = _make_kg_with_synergies(synergies)
    scorer = CardScorer(kg=kg)

    score = run(
        scorer._synergy_score("Goblin Guide", ["Goblin Chieftain"])
    )
    assert score == pytest.approx(0.9)


def test_synergy_score_no_match():
    synergies = {
        "Goblin Chieftain": [{"card": "Krenko, Mob Boss", "strength": 1.0}]
    }
    kg = _make_kg_with_synergies(synergies)
    scorer = CardScorer(kg=kg)

    score = run(scorer._synergy_score("Lightning Bolt", ["Goblin Chieftain"]))
    assert score == pytest.approx(0.0)


def test_synergy_caching():
    calls = []

    async def get_synergies_for(card_name: str):
        calls.append(card_name)
        return [{"card": "X", "strength": 0.5}]

    kg = AsyncMock()
    kg.get_synergies_for = get_synergies_for
    kg.detect_near_combos = AsyncMock(return_value=[])

    scorer = CardScorer(kg=kg)
    run(scorer._synergy_score("X", ["A"]))
    run(scorer._synergy_score("X", ["A"]))  # should use cache
    assert calls.count("A") == 1


# ---------------------------------------------------------------------------
# Combo score with stubbed KG
# ---------------------------------------------------------------------------


def test_combo_score_missing_piece():
    near_combos = [
        {"missingPiece": "Walking Ballista", "comboId": "c1"},
        {"missingPiece": "Heliod, Sun-Crowned", "comboId": "c2"},
    ]
    kg = AsyncMock()
    kg.detect_near_combos = AsyncMock(return_value=near_combos)
    kg.get_synergies_for = AsyncMock(return_value=[])

    scorer = CardScorer(kg=kg)
    score = run(scorer._combo_score("Walking Ballista", ["Heliod, Sun-Crowned"]))
    assert score == pytest.approx(1.0)


def test_combo_score_not_missing():
    near_combos = [{"missingPiece": "Heliod, Sun-Crowned", "comboId": "c2"}]
    kg = AsyncMock()
    kg.detect_near_combos = AsyncMock(return_value=near_combos)
    kg.get_synergies_for = AsyncMock(return_value=[])

    scorer = CardScorer(kg=kg)
    score = run(scorer._combo_score("Lightning Bolt", ["Heliod, Sun-Crowned"]))
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------


def test_score_batch_ordering():
    """Highest-scoring card should come first when sorted."""
    scorer = CardScorer()
    cards = [
        _card("Wrath of God"),          # archetype 1.0
        _card("Forest", type_line="Basic Land — Forest"),  # archetype 0.0
        _card("Sol Ring", oracle="{t}: add {c}{c}"),       # archetype 0.8
    ]
    scored = run(scorer.score_batch(cards, []))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[0][0]["name"]
    # "Wrath of God" and "Sol Ring" both outscore "Forest"
    assert top != "Forest"


# ---------------------------------------------------------------------------
# reset_caches
# ---------------------------------------------------------------------------


def test_reset_caches_clears_state():
    scorer = CardScorer()
    scorer._synergy_cache[("A", "B")] = 0.5
    scorer._combo_cache[frozenset(["X"])] = []
    scorer.reset_caches()
    assert len(scorer._synergy_cache) == 0
    assert len(scorer._combo_cache) == 0
