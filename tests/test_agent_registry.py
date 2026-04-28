"""Smoke tests for the agent registry — guards against typos
in `src/agents/__init__.py` and missing factory imports."""

from __future__ import annotations

import pytest

from src.agents import AGENT_REGISTRY, list_agents, make_agent


def test_registry_has_core_agents():
    names = set(list_agents())
    expected = {
        "random", "heuristic", "kg_heuristic", "human",
        "ollama", "llm", "world_model", "active_inference",
        "llm_fusion", "fusion",
    }
    missing = expected - names
    assert not missing, f"registry missing: {missing}"


def test_make_random_returns_agent_with_player_id():
    a = make_agent("random", "player_1")
    assert a is not None
    assert a.player_id == "player_1"


def test_make_heuristic_accepts_seed_kwarg():
    a = make_agent("heuristic", "player_2", seed=42)
    assert a is not None
    assert a.player_id == "player_2"


def test_unknown_agent_raises_keyerror():
    with pytest.raises(KeyError):
        make_agent("not_a_real_agent", "player_1")


def test_llm_fusion_registered():
    # Building a fusion agent without checkpoints should still succeed
    # (world_model / kg are optional — the agent falls back gracefully).
    a = make_agent("llm_fusion", "player_1")
    assert a is not None
    assert a.player_id == "player_1"
