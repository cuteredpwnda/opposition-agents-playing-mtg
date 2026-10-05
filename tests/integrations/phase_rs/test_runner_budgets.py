from __future__ import annotations

import asyncio

import pytest

from src.integrations.phase_rs import runner
from src.integrations.phase_rs.agent_bridge import RandomActionPicker
from src.integrations.phase_rs.client import (
    PROTOCOL_VERSION,
    GameCreated,
    PhaseServerConfig,
    ServerHello,
    parse_server_message,
)


@pytest.mark.parametrize("cap", ["turn", "time"])
async def test_budget_is_incomplete_and_does_not_reconnect(monkeypatch, cap):
    started = parse_server_message({
        "type": "GameStarted",
        "data": {"state": {"turn_number": 7}, "your_player": 0, "legal_actions": []},
    })
    connections = []

    class FakeClient:
        def __init__(self, config):
            connections.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def handshake(self):
            return ServerHello("test", "test", PROTOCOL_VERSION, "Full")

        async def create_game_with_ai(self, **kwargs):
            return GameCreated("pod", "token")

        async def expect(self, message_type):
            return started

        async def recv(self, *, timeout):
            await asyncio.sleep(timeout)
            raise TimeoutError

    monkeypatch.setattr(runner, "PhaseServerClient", FakeClient)
    result = await runner._play(
        cfg=PhaseServerConfig(stream_timeout_s=30), deck={},
        picker=RandomActionPicker(seed=7), ai_difficulty="VeryEasy",
        ai_deck_name="Blue Control", display_name="Test", max_actions=100, log=[],
        max_turns=6 if cap == "turn" else None,
        max_game_seconds=0.01 if cap == "time" else None,
    )
    assert result.reason == ("turn_cap" if cap == "turn" else "game_timeout")
    assert result.winner_seat is None
    assert result.actions_sent == 0
    assert not any(e["event"] == "game_over" for e in result.trace)
    assert len(connections) == 1


@pytest.mark.parametrize("kwargs", [
    {"max_turns": 0}, {"max_game_seconds": 0},
    {"max_game_seconds": float("inf")}, {"max_game_seconds": float("nan")},
])
async def test_invalid_budget_rejected_before_connecting(kwargs):
    with pytest.raises(ValueError):
        await runner.run_game(deck={}, **kwargs)


@pytest.mark.parametrize("size", [0, -1, 65 * 1024 * 1024])
def test_unsafe_message_limit_is_rejected(size):
    with pytest.raises(ValueError):
        PhaseServerConfig(max_message_bytes=size)
