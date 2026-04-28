"""Reasoning-trace integration: agents populate ``last_reasoning`` and
the JSONL collector forwards it into the action record."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent
from src.agents.reasoning import ReasoningTrace
from src.engine.game_state import (
    Action,
    ActionType,
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.orchestrator.jsonl_trace import JsonlActionTrace


def _make_state() -> GameState:
    s = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    s.cards = [
        CardInstance(
            instance_id="A_mtn",
            card_data={
                "name": "Mountain",
                "type_line": "Basic Land — Mountain",
                "oracle_text": "",
            },
            zone=Zone.HAND,
            owner_id="A",
            controller_id="A",
        ),
    ]
    return s


def test_reasoning_trace_to_dict_truncates_scores() -> None:
    big = ReasoningTrace(agent_kind="x", scores=[float(i) for i in range(100)])
    d = big.to_dict()
    assert len(d["scores"]) == 32
    assert d.get("scores_truncated") is True


def test_random_agent_records_reasoning() -> None:
    state = _make_state()
    agent = RandomAgent(player_id="A")
    legal = [
        Action(action_type=ActionType.PLAY_LAND, player_id="A", card_instance_id="A_mtn"),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="A"),
    ]
    asyncio.run(agent.decide_action(state, legal))
    assert agent.last_reasoning is not None
    assert agent.last_reasoning["agent_kind"] == "random"
    assert agent.last_reasoning["legal_action_count"] == 2


def test_heuristic_agent_records_reasoning() -> None:
    state = _make_state()
    agent = HeuristicAgent(player_id="A", seed=1)
    legal = [
        Action(action_type=ActionType.PLAY_LAND, player_id="A", card_instance_id="A_mtn"),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="A"),
    ]
    chosen = asyncio.run(agent.decide_action(state, legal))
    assert chosen.action_type == ActionType.PLAY_LAND
    assert agent.last_reasoning is not None
    assert agent.last_reasoning["agent_kind"] == "heuristic"
    assert agent.last_reasoning["beliefs"]["chose_via"] in {
        "top_priority",
        "aggressive_override",
        "chump_block",
    }


def test_jsonl_trace_includes_reasoning(tmp_path: Path) -> None:
    state = _make_state()
    out = tmp_path / "trace.jsonl"
    trace = JsonlActionTrace(out, game_id="g1")
    trace.on_state(state, "A")
    action = Action(
        action_type=ActionType.PLAY_LAND, player_id="A", card_instance_id="A_mtn"
    )
    reasoning = ReasoningTrace(
        agent_kind="heuristic",
        rationale="play the only land",
        legal_action_count=2,
        chosen_index=0,
        beliefs={"chose_via": "top_priority"},
    ).to_dict()
    trace.on_action(action, reasoning=reasoning, agent_name="HeuristicTest")
    trace.close()

    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["agent"] == "HeuristicTest"
    assert rec["reasoning"]["rationale"] == "play the only land"
    assert rec["reasoning"]["agent_kind"] == "heuristic"
    assert rec["reasoning"]["beliefs"]["chose_via"] == "top_priority"


def test_priority_loop_passes_reasoning_through(tmp_path: Path) -> None:
    """Smoke: build a 2-player game, run a few ticks via GameRunner,
    and confirm the JSONL records carry a ``reasoning`` field."""
    pytest.importorskip("torch", reason="GameRunner imports world model utilities lazily")
    from main import build_simple_deck
    from src.orchestrator.game_runner import GameConfig, GameRunner

    out = tmp_path / "trace.jsonl"
    trace = JsonlActionTrace(out, game_id="smoke")
    runner = GameRunner(config=GameConfig(max_turns=3), self_play_collector=trace)
    agents = {
        "player_0": RandomAgent("player_0"),
        "player_1": HeuristicAgent("player_1", seed=2),
    }
    decks = {pid: build_simple_deck() for pid in agents}
    asyncio.run(runner.run_game(agents, decks))
    trace.close()

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    decision_records = [r for r in lines if r.get("action") and "reasoning" in r]
    assert decision_records, "no reasoning records were produced"
    kinds = {r["reasoning"]["agent_kind"] for r in decision_records}
    assert kinds & {"random", "heuristic"}
