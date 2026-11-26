from __future__ import annotations

import pytest

from src.integrations.phase_rs import runner
from src.integrations.phase_rs.agent_bridge import PickerError, RandomActionPicker
from src.integrations.phase_rs.client import (
    PROTOCOL_VERSION,
    ActionFailed,
    GameCreated,
    PhaseServerClient,
    PhaseServerConfig,
    PhaseServerError,
    ServerHello,
    parse_server_message,
)


@pytest.mark.parametrize("drain", [False, True])
@pytest.mark.parametrize("reason", [
    "action_failed", "request_rejected", "ai_driver_fault", "picker_invalid_response",
    "state_withheld", "action_noop",
])
async def test_runner_records_operational_failures(monkeypatch, drain, reason):
    failure = ActionFailed(message="Explicit failure", reason=reason)
    started = parse_server_message({
        "type": "GameStarted",
        "data": {
            "state": {
                "turn_number": 1, "phase": "PreCombatMain",
                "waiting_for": {"type": "Priority", "data": {"player": 0}},
            },
            "your_player": 0,
            "legal_actions": (
                [{"type": "PassPriority"}] if drain or reason == "picker_invalid_response" else []
            ),
        },
    })

    class FakeClient:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def handshake(self):
            return ServerHello("test", "test", PROTOCOL_VERSION, "Full")

        async def create_game_with_ai(self, **kwargs):
            return GameCreated("test", "token")

        async def expect(self, message_type):
            return started

        async def recv(self, **kwargs):
            if reason == "picker_invalid_response":
                raise TimeoutError
            return failure

    class FailingPicker:
        name = "failing"
        last_reasoning = {"candidate_coverage": 0.5}

        def pick(self, *args):
            raise PickerError("Explicit failure", reason="picker_invalid_response")

    monkeypatch.setattr(runner, "PhaseServerClient", FakeClient)
    result = await runner._play(
        cfg=PhaseServerConfig(), deck={},
        picker=(
            FailingPicker() if reason == "picker_invalid_response" else RandomActionPicker(seed=7)
        ),
        ai_difficulty="Medium", ai_deck_name="Red Deck Wins",
        display_name="Test", max_actions=10, log=[],
    )
    assert result.reason == reason
    assert result.winner_seat is None
    assert result.trace[-1]["event"] == reason
    assert result.trace[-1]["message"] == "Explicit failure"
    if reason == "picker_invalid_response":
        assert result.trace[-1]["picker_reasoning"] == {"candidate_coverage": 0.5}
        assert result.trace[-1]["decision_time_sec"] >= 0
    assert any("Explicit failure" in line for line in result.log)


async def test_control_plane_does_not_skip_failure(monkeypatch):
    client = PhaseServerClient(PhaseServerConfig())

    async def receive(**kwargs):
        return ActionFailed("Explicit failure", reason="request_rejected")

    monkeypatch.setattr(client, "recv", receive)
    with pytest.raises(PhaseServerError, match="request_rejected: Explicit failure"):
        await client.expect("GameCreated")


async def test_handshake_rejects_stale_server_before_sending_client_hello(monkeypatch):
    client = PhaseServerClient(PhaseServerConfig())
    sent = []

    async def expect(message_type):
        return ServerHello("old", "old", 106, "Full")

    async def send(*args):
        sent.append(args)

    monkeypatch.setattr(client, "expect", expect)
    monkeypatch.setattr(client, "_send", send)
    with pytest.raises(PhaseServerError, match="protocol mismatch: server=106 client=374"):
        await client.handshake()
    assert not sent


@pytest.mark.parametrize(("actual", "matches"), [
    ("b42447bc3", True),
    ("1191bba65", False),
    ("b", False),
    ("dev", False),
])
async def test_handshake_can_require_the_frozen_checkout(monkeypatch, actual, matches):
    client = PhaseServerClient(PhaseServerConfig(
        expected_server_commit="b42447bc3611273fb2413fa95d22a9e5ff623c01",
    ))
    sent = []

    async def expect(message_type):
        return ServerHello("test", actual, PROTOCOL_VERSION, "Full")

    async def send(*args):
        sent.append(args)

    monkeypatch.setattr(client, "expect", expect)
    monkeypatch.setattr(client, "_send", send)
    if matches:
        await client.handshake()
        assert len(sent) == 1
    else:
        with pytest.raises(PhaseServerError, match="server build mismatch"):
            await client.handshake()
        assert not sent


@pytest.mark.parametrize("drain", [False, True])
@pytest.mark.parametrize("winner", [None, 0])
@pytest.mark.parametrize("kind", ["StateUpdate", "GameOver"])
async def test_terminal_evidence_is_retained_in_both_receive_paths(
    monkeypatch, drain, winner, kind,
):
    started = parse_server_message({
        "type": "GameStarted",
        "data": {
            "state": {"turn_number": 1, "waiting_for": {"type": "Priority"}},
            "your_player": 0,
            "legal_actions": [{"type": "PassPriority"}] if drain else [],
        },
    })
    terminal = parse_server_message({
        "type": kind,
        "data": (
            {"state": {"turn_number": 1,
                       "waiting_for": {"type": "GameOver", "data": {"winner": winner}}}}
            if kind == "StateUpdate" else {"winner": winner, "reason": "engine_reason"}
        ),
    })

    class FakeClient:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def handshake(self):
            return ServerHello("test", "test", PROTOCOL_VERSION, "Full")

        async def create_game_with_ai(self, **kwargs):
            return GameCreated("test", "token")

        async def expect(self, message_type):
            return started

        async def recv(self, **kwargs):
            return terminal

    monkeypatch.setattr(runner, "PhaseServerClient", FakeClient)
    result = await runner._play(
        cfg=PhaseServerConfig(), deck={}, picker=RandomActionPicker(seed=7),
        ai_difficulty="Medium", ai_deck_name="Blue Control",
        display_name="Test", max_actions=10, log=[],
    )
    assert result.winner_seat == winner
    assert result.reason == ("game_rules" if kind == "StateUpdate" else "engine_reason")
    assert result.trace[-1]["event"] == "game_over"
    assert result.trace[-1]["winner"] == winner
    assert result.actions_sent == 0
