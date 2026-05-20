"""Validate our parser against phase-rs's adapter-contract fixtures.

The phase-rs repo (vendored as a submodule under ``external/phase-rs``) ships
canonical JSON envelopes for the messages we depend on. We parse each one
through ``parse_server_message`` and check that the structural fields land
in the expected dataclass fields.

These tests are pure-parser; they do not touch the network. They are the
guard that catches upstream protocol drift the moment we ``git submodule
update`` — if a field is renamed, the test fails with a clear message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.integrations.phase_rs.client import (
    GameStarted,
    StateUpdate,
    parse_envelope,
    parse_server_message,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "external" / "phase-rs" / "fixtures" / "adapter-contract"

# Skip the whole module if the submodule isn't checked out — keeps offline
# CI green when external/ is empty.
pytestmark = pytest.mark.skipif(
    not FIXTURES.exists(),
    reason="phase-rs submodule not checked out (run: git submodule update --init)",
)


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_envelope_shape_game_started() -> None:
    payload = _load("game_started.json")
    msg_type, data = parse_envelope(payload)
    assert msg_type == "GameStarted"
    assert isinstance(data, dict)
    assert "state" in data and "your_player" in data


def test_parse_game_started() -> None:
    payload = _load("game_started.json")
    msg = parse_server_message(payload)
    assert isinstance(msg, GameStarted)
    assert msg.your_player == 0
    assert msg.legal_actions == [{"type": "PassPriority"}]
    assert msg.state["turn_number"] == 1
    assert msg.state["phase"] == "PreCombatMain"


def test_parse_state_update() -> None:
    payload = _load("state_update.json")
    msg = parse_server_message(payload)
    assert isinstance(msg, StateUpdate)
    assert msg.state["active_player"] == 0
    # events / legal_actions default to [] when absent
    assert isinstance(msg.events, list)
    assert isinstance(msg.legal_actions, list)


def test_parse_waiting_for_priority_is_passthrough() -> None:
    # waiting_for_priority.json is a *field* fixture, not a ServerMessage —
    # validates the inner shape. parse_envelope must still accept it as a
    # tagged union and return the raw inner data unchanged.
    payload = _load("waiting_for_priority.json")
    msg_type, data = parse_envelope(payload)
    assert msg_type == "Priority"
    assert "player" in data


def test_parse_game_action_envelope() -> None:
    # game_action.json is a sample GameAction (ChooseLegend). We don't model
    # GameAction; we just confirm the envelope is the same tag/content shape
    # so our send_action(action) call can echo dicts of this form verbatim.
    payload = _load("game_action.json")
    msg_type, data = parse_envelope(payload)
    assert msg_type == "ChooseLegend"
    assert "keep" in data
