from __future__ import annotations

import pytest

from src.integrations.phase_rs import runner
from src.integrations.phase_rs.agent_bridge import RandomActionPicker
from src.integrations.phase_rs.client import (
    PROTOCOL_VERSION,
    FullSessionKey,
    GameCreated,
    PhaseServerClient,
    PhaseServerConfig,
    PhaseServerError,
    ServerHello,
    parse_server_message,
)


def started(*, revision=1, seat=0, terminal=False, key=None):
    return parse_server_message({
        "type": "GameStarted",
        "data": {
            "state_revision": revision, "your_player": seat,
            "full_key": key or {"game_code": "TEST", "generation": 17},
            "state": {"turn_number": 7, "waiting_for": (
                {"type": "GameOver", "data": {"winner": 0}} if terminal
                else {"type": "Priority", "data": {"player": 1}}
            )},
        },
    })


@pytest.mark.parametrize("generation", [0, -1, True, 1.5])
def test_invalid_full_session_key(generation):
    with pytest.raises(ValueError):
        FullSessionKey("TEST", generation)


async def test_reconnect_sends_exact_issued_generation(monkeypatch):
    client = PhaseServerClient()
    calls = []

    async def send(kind, data):
        calls.append((kind, data))

    async def expect(kind):
        assert kind == "GameStarted"
        return started()

    monkeypatch.setattr(client, "_send", send)
    monkeypatch.setattr(client, "expect", expect)
    await client.reconnect("TEST", "private-token", FullSessionKey("TEST", 17))
    assert calls == [("Reconnect", {
        "game_code": "TEST", "player_token": "private-token",
        "full_key": {"game_code": "TEST", "generation": 17},
    })]


async def test_client_close_is_idempotent_and_used_by_context_exit():
    class Socket:
        closes = 0

        async def close(self):
            self.closes += 1

    client = PhaseServerClient()
    socket = Socket()
    client._ws = socket
    await client.close()
    await client.close()
    await client.__aexit__(None, None, None)
    assert socket.closes == 1
    assert client._ws is None


async def test_reconnect_does_not_accept_a_different_generation(monkeypatch):
    client = PhaseServerClient()

    async def send(*args):
        pass

    async def expect(*args):
        return started(key={"game_code": "TEST", "generation": 18})

    monkeypatch.setattr(client, "_send", send)
    monkeypatch.setattr(client, "expect", expect)
    with pytest.raises(PhaseServerError, match="requested Full session"):
        await client.reconnect("TEST", "private-token", FullSessionKey("TEST", 17))


@pytest.mark.parametrize("failure", [None, "wrong_seat", "stale", "rejected", "no_key"])
async def test_runner_requires_session_restoration_not_just_a_socket(monkeypatch, failure):
    clients = []
    reconnects = []
    opened = []
    closed = []

    class Client:
        def __init__(self, config):
            self.config = config
            clients.append(self)

        async def __aenter__(self):
            opened.append(self)
            return self

        async def __aexit__(self, *args):
            closed.append(self)

        async def close(self):
            closed.append(self)

        async def handshake(self):
            return ServerHello("test", "test", PROTOCOL_VERSION, "Full")

        async def create_game_with_ai(self, **kwargs):
            return GameCreated("TEST", "private-token",
                               None if failure == "no_key" else FullSessionKey("TEST", 17))

        async def expect(self, kind):
            result = started()
            if failure == "no_key":
                result.full_key = None
            return result

        async def recv(self, **kwargs):
            raise TimeoutError("simulated stream silence")

        async def reconnect(self, game_code, token, key):
            reconnects.append((game_code, token, key))
            if failure == "rejected":
                raise PhaseServerError("session expired")
            return started(
                revision=0 if failure == "stale" else 2,
                seat=1 if failure == "wrong_seat" else 0, terminal=True,
            )

    monkeypatch.setattr(runner, "PhaseServerClient", Client)
    result = await runner._play(
        cfg=PhaseServerConfig(), deck={}, picker=RandomActionPicker(seed=7),
        ai_difficulty="VeryEasy", ai_deck_name="Blue Control",
        display_name="Test", max_actions=10, log=[], reconnect_attempts=1,
    )
    assert len(clients) == 1
    assert closed[-1] is clients[0]
    assert "private-token" not in str(result.trace) + str(result.log)
    if failure is None:
        assert result.reason == "game_rules"
        assert result.winner_seat == 0
        assert any(e["event"] == "reconnect_success" for e in result.trace)
        assert result.trace[-1]["event"] == "game_over"
        assert reconnects == [("TEST", "private-token", FullSessionKey("TEST", 17))]
        assert len(opened) == 2
    else:
        assert result.reason == "stream_timeout"
        assert not any(e["event"] == "reconnect_success" for e in result.trace)
        if failure == "no_key":
            assert reconnects == []
