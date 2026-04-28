"""Tests for OllamaAgent mulligan and bottom-card selection.

These tests verify that the hooks exist and have sensible fallback behavior.
End-to-end integration with Ollama requires an actual instance running.
"""

from __future__ import annotations

import pytest

from src.agents.llm_agent import OllamaAgent
from src.engine.game_state import CardInstance, Zone


def _card(name: str, type_line: str, cmc: float) -> CardInstance:
    return CardInstance(
        instance_id=f"id_{name}",
        card_data={"name": name, "type_line": type_line, "cmc": cmc},
        zone=Zone.HAND,
        owner_id="p1",
        controller_id="p1",
    )


def test_ollama_agent_decide_mulligan_exists():
    """OllamaAgent has a mulligan decision hook (even if Ollama unavailable)."""
    agent = OllamaAgent("p1", model="gemma4:e2b")
    hand = [_card("Mountain", "Basic Land - Mountain", 0) for _ in range(7)]
    # Should not raise; falls back to parent heuristic when Ollama unavailable
    result = agent.decide_mulligan(hand, mulligans_taken=0, max_mulligans=3)
    assert isinstance(result, bool)


def test_ollama_agent_decide_mulligan_forced_keep_at_cap():
    """Always keep when at max mulligans."""
    agent = OllamaAgent("p1", model="gemma4:e2b")
    bad_hand = [_card("Big Spell", "Sorcery", 7) for _ in range(7)]
    assert agent.decide_mulligan(bad_hand, mulligans_taken=3, max_mulligans=3)


def test_ollama_agent_select_bottom_cards_exists():
    """OllamaAgent has a bottom-cards selection hook."""
    agent = OllamaAgent("p1", model="gemma4:e2b")
    hand = [
        _card("Mountain", "Basic Land - Mountain", 0),
        _card("Mountain", "Basic Land - Mountain", 0),
        _card("Bolt", "Instant", 1),
        _card("Wrath", "Sorcery", 4),
        _card("Giant", "Creature - Giant", 6),
        _card("Goblin", "Creature - Goblin", 1),
        _card("Draw Spell", "Sorcery", 3),
    ]
    bottom = agent.select_bottom_cards(hand, 2)
    assert isinstance(bottom, list)
    assert len(bottom) <= 2


def test_ollama_agent_select_bottom_cards_empty_hand():
    """Gracefully handle empty hand."""
    agent = OllamaAgent("p1")
    assert agent.select_bottom_cards([], 1) == []


def test_ollama_agent_describe_hand():
    """_describe_hand returns readable format."""
    hand = [
        _card("Mountain", "Basic Land - Mountain", 0),
        _card("Bolt", "Instant", 1),
    ]
    desc = OllamaAgent._describe_hand(hand)
    assert "Mountain" in desc
    assert "Bolt" in desc
    assert "Land" in desc or "Instant" in desc
