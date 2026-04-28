"""JSONL action trace collector."""

from __future__ import annotations

import json
from pathlib import Path

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


def _state(tmp_cards):
    s = GameState(
        format="commander",
        turn_number=2,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    s.cards = list(tmp_cards)
    return s


def test_trace_writes_one_record_per_action(tmp_path: Path):
    state = _state([
        CardInstance(
            instance_id="A_mountain",
            card_data={"name": "Mountain", "type_line": "Basic Land — Mountain", "oracle_text": ""},
            zone=Zone.HAND,
            owner_id="A",
            controller_id="A",
        ),
    ])
    out = tmp_path / "trace.jsonl"
    trace = JsonlActionTrace(out, game_id="g1")
    trace.on_state(state, "A")
    action = Action(
        action_type=ActionType.PLAY_LAND,
        player_id="A",
        card_instance_id="A_mountain",
    )
    trace.on_action(action)
    trace.finish_game(winner=0, num_turns=2)
    trace.close()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["game_id"] == "g1"
    assert rec["action"]["type"] == "PLAY_LAND"
    assert rec["action"]["card_instance_id"] == "A_mountain"
    assert rec["state"]["priority_player"] == "A"
    end = json.loads(lines[1])
    assert end["event"] == "game_end"
    assert end["winner"] == 0


def test_trace_accepts_int_player_id_from_priority_loop(tmp_path: Path):
    state = _state([])
    out = tmp_path / "trace.jsonl"
    with JsonlActionTrace(out) as trace:
        # priority_loop calls on_state with priority_player_index (an int).
        trace.on_state(state, 1)
        trace.on_action(Action(action_type=ActionType.PASS_PRIORITY, player_id="B"))
    line = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert line["state"]["priority_player"] == "B"
