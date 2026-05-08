"""Tests for the goldfish simulator (NullAgent + GoldfishRunner).

All tests run without real card data or external services.  The
``GameRunner`` is mocked where full game execution is too slow.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.null_agent import NullAgent
from src.agents.goldfish_runner import (
    GoldfishRun,
    GoldfishRunner,
    GoldfishStats,
    TurnSnapshot,
    _make_null_deck,
)
from src.engine.game_state import Action, ActionType, GameState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_state(player_id: str = "p1") -> GameState:
    """Minimal GameState for action-choice tests."""
    from src.engine.game_state import PlayerState

    state = GameState()
    state.players = [PlayerState(player_id=player_id, life_total=20)]
    state.active_player_index = 0
    return state


def _pass_action(pid: str = "p1") -> Action:
    return Action(action_type=ActionType.PASS_PRIORITY, player_id=pid)


def _cast_action(pid: str = "p1") -> Action:
    return Action(action_type=ActionType.CAST_SPELL, player_id=pid)


# ---------------------------------------------------------------------------
# NullAgent
# ---------------------------------------------------------------------------

def test_null_agent_always_passes_when_pass_is_legal():
    agent = NullAgent(player_id="p1", name="Null")
    state = _minimal_state("p1")
    legal = [_cast_action("p1"), _pass_action("p1")]
    result = asyncio.run(agent.decide_action(state, legal))
    assert result.action_type == ActionType.PASS_PRIORITY


def test_null_agent_falls_back_to_first_action():
    agent = NullAgent(player_id="p1", name="Null")
    state = _minimal_state("p1")
    # Only non-pass action available
    legal = [_cast_action("p1")]
    result = asyncio.run(agent.decide_action(state, legal))
    assert result == legal[0]


def test_null_agent_empty_legal_actions():
    agent = NullAgent(player_id="p1", name="Null")
    state = _minimal_state("p1")
    result = asyncio.run(agent.decide_action(state, []))
    assert result.action_type == ActionType.PASS_PRIORITY


def test_null_agent_registered_in_registry():
    from src.agents import make_agent, AGENT_REGISTRY

    assert "null" in AGENT_REGISTRY
    agent = make_agent("null", "p1")
    assert isinstance(agent, NullAgent)


# ---------------------------------------------------------------------------
# Null deck helper
# ---------------------------------------------------------------------------

def test_make_null_deck_length():
    deck = _make_null_deck(40)
    assert len(deck) == 40


def test_make_null_deck_all_plains():
    deck = _make_null_deck(5)
    assert all(d["name"] == "Plains" for d in deck)


def test_make_null_deck_cards_are_independent_copies():
    deck = _make_null_deck(3)
    deck[0]["name"] = "Mutated"
    assert deck[1]["name"] == "Plains", "copies should be independent"


# ---------------------------------------------------------------------------
# GoldfishRun / GoldfishStats
# ---------------------------------------------------------------------------

def test_goldfish_run_won_property():
    assert GoldfishRun(kill_turn=4, turns_played=4).won is True
    assert GoldfishRun(kill_turn=None, turns_played=10).won is False


def test_goldfish_stats_win_rate():
    stats = GoldfishStats(
        runs=10,
        wins=7,
        kill_turns=[3, 4, 4, 5, 5, 5, 6],
        avg_spells_per_turn=[],
        avg_power_per_turn=[],
        avg_creatures_per_turn=[],
        curve_hit_rate=0.8,
    )
    assert stats.win_rate == pytest.approx(0.7)


def test_goldfish_stats_win_rate_zero_runs():
    stats = GoldfishStats(
        runs=0, wins=0, kill_turns=[],
        avg_spells_per_turn=[], avg_power_per_turn=[], avg_creatures_per_turn=[],
        curve_hit_rate=0.0,
    )
    assert stats.win_rate == 0.0


def test_goldfish_stats_avg_kill_turn():
    stats = GoldfishStats(
        runs=4,
        wins=3,
        kill_turns=[3, 4, 5],
        avg_spells_per_turn=[],
        avg_power_per_turn=[],
        avg_creatures_per_turn=[],
        curve_hit_rate=0.5,
    )
    assert stats.avg_kill_turn == pytest.approx(4.0)


def test_goldfish_stats_avg_kill_turn_none_when_no_wins():
    stats = GoldfishStats(
        runs=5, wins=0, kill_turns=[],
        avg_spells_per_turn=[], avg_power_per_turn=[], avg_creatures_per_turn=[],
        curve_hit_rate=0.0,
    )
    assert stats.avg_kill_turn is None


def test_goldfish_stats_kill_turn_distribution():
    stats = GoldfishStats(
        runs=10,
        wins=5,
        kill_turns=[3, 3, 4, 5, 6],  # 5 wins in 10 runs
        avg_spells_per_turn=[],
        avg_power_per_turn=[],
        avg_creatures_per_turn=[],
        curve_hit_rate=0.5,
    )
    cdf = stats.kill_turn_distribution
    assert cdf[3] == pytest.approx(0.2)   # 2/10 won by T3
    assert cdf[4] == pytest.approx(0.3)   # 3/10 by T4
    assert cdf[6] == pytest.approx(0.5)   # all 5 wins by T6


def test_goldfish_stats_summary_contains_key_info():
    stats = GoldfishStats(
        runs=50,
        wins=35,
        kill_turns=[4] * 35,
        avg_spells_per_turn=[0.5, 1.2, 1.8],
        avg_power_per_turn=[0.0, 1.0, 3.5],
        avg_creatures_per_turn=[0.0, 1.0, 2.5],
        curve_hit_rate=0.65,
    )
    summary = stats.summary(deck_name="test_deck")
    assert "test_deck" in summary
    assert "70.0%" in summary     # win rate
    assert "65.0%" in summary     # curve hit rate


# ---------------------------------------------------------------------------
# GoldfishRunner — unit test with mocked GameRunner
# ---------------------------------------------------------------------------

def _fake_game_result(winner: str, turns: int) -> MagicMock:
    result = MagicMock()
    result.winner = winner
    result.turns = turns
    result.log = []
    return result


def test_goldfish_runner_aggregates_wins():
    """GoldfishRunner.run() returns correct win stats when all games are won."""
    dummy_deck = [{"name": "Island", "type_line": "Basic Land — Island", "mana_cost": ""}]

    # Each game the active player wins on turn 4
    fake_result = _fake_game_result(GoldfishRunner.ACTIVE_PID, turns=4)

    with patch(
        "src.agents.goldfish_runner.GameRunner.run_game",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        runner = GoldfishRunner(runs=5, max_turns=8, seed=1)
        stats = asyncio.run(runner.run(dummy_deck))

    assert stats.runs == 5
    assert stats.wins == 5
    assert stats.win_rate == pytest.approx(1.0)


def test_goldfish_runner_aggregates_losses():
    """GoldfishRunner.run() shows 0% win rate when null opponent wins."""
    dummy_deck = [{"name": "Island", "type_line": "Basic Land — Island", "mana_cost": ""}]

    # Null opponent "wins" every game (timeout win)
    fake_result = _fake_game_result(GoldfishRunner.NULL_PID, turns=10)

    with patch(
        "src.agents.goldfish_runner.GameRunner.run_game",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        runner = GoldfishRunner(runs=4, max_turns=8, seed=2)
        stats = asyncio.run(runner.run(dummy_deck))

    assert stats.wins == 0
    assert stats.win_rate == pytest.approx(0.0)
    assert stats.avg_kill_turn is None


def test_goldfish_runner_uses_derived_seeds():
    """Each run gets a different seed derived from the base seed."""
    seeds_used: list[int | None] = []

    original_heuristic_init = None

    import src.agents.heuristic_agent as _ha_mod

    original_init = _ha_mod.HeuristicAgent.__init__

    def _capturing_init(self, player_id, name="", seed=None, **kw):
        seeds_used.append(seed)
        original_init(self, player_id=player_id, name=name, seed=seed, **kw)

    dummy_deck = [{"name": "Island", "type_line": "Basic Land — Island", "mana_cost": ""}]
    fake_result = _fake_game_result(GoldfishRunner.ACTIVE_PID, turns=3)

    with (
        patch.object(_ha_mod.HeuristicAgent, "__init__", _capturing_init),
        patch(
            "src.agents.goldfish_runner.GameRunner.run_game",
            new_callable=AsyncMock,
            return_value=fake_result,
        ),
    ):
        runner = GoldfishRunner(runs=3, max_turns=5, seed=100)
        asyncio.run(runner.run(dummy_deck))

    # 3 games → seeds 100, 101, 102
    assert seeds_used == [100, 101, 102]
