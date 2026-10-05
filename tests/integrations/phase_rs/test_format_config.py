from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from src.integrations.phase_rs.client import (
    PROTOCOL_VERSION,
    GameCreated,
    PhaseServerClient,
    PhaseServerConfig,
)
from src.integrations.phase_rs.decks import STARTER_DECK_NAMES
from src.integrations.phase_rs.format_config import format_config
from src.integrations.phase_rs.server_process import DEFAULT_SUBMODULE


def test_engine_snapshot_matches_pinned_revision():
    snapshot = json.loads(
        (Path(__file__).resolve().parents[3] / "src" / "integrations"
         / "phase_rs" / "format_defaults.json").read_text("utf-8")
    )
    assert snapshot["protocol_version"] == PROTOCOL_VERSION
    if not (DEFAULT_SUBMODULE / "Cargo.toml").exists():
        pytest.skip("Engine source not checked out")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=DEFAULT_SUBMODULE,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert snapshot["phase_rs_commit"] == revision


def test_starter_deck_names_match_engine_registry():
    source = DEFAULT_SUBMODULE / "crates" / "engine" / "src" / "starter_decks.rs"
    if not source.exists():
        pytest.skip("Engine source not checked out")
    assert STARTER_DECK_NAMES == tuple(re.findall(r'name: "([^"]+)"', source.read_text("utf-8")))


def test_constructed_and_commander_defaults_are_distinct():
    modern = format_config("Modern")
    commander = format_config("Commander")
    assert modern["starting_life"] == 20
    assert modern["deck_size"] == {"type": "Minimum", "data": 60}
    assert modern["uses_commander"] is False
    assert commander["starting_life"] == 40
    assert commander["deck_size"] == {"type": "Exactly", "data": 100}
    assert commander["commander_damage_threshold"] == 21
    assert commander["uses_commander"] is True


def test_configs_are_independent_copies():
    modified = format_config("Modern")
    modified["deck_size"]["data"] = 1
    assert format_config("Modern")["deck_size"]["data"] == 60


def test_unknown_format_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported engine format"):
        format_config("ImaginaryFormat")


@pytest.mark.parametrize("name", [None, "Modern", "Commander", "Pioneer"])
async def test_create_game_sends_complete_engine_defaults(monkeypatch, name):
    client = PhaseServerClient(PhaseServerConfig())
    sent = []

    async def send(message_type, payload):
        sent.append((message_type, payload))

    async def expect(message_type):
        return GameCreated("game", "token")

    monkeypatch.setattr(client, "_send", send)
    monkeypatch.setattr(client, "expect", expect)
    await client.create_game_with_ai(deck={"main_deck": ["Mountain"] * 60}, format_name=name)
    assert sent[0][0] == "CreateGameWithSettings"
    if name is None:
        assert "format_config" not in sent[0][1]
    else:
        assert sent[0][1]["format_config"] == format_config(name)
        assert "starting_life" in sent[0][1]["format_config"]
        assert "uses_commander" in sent[0][1]["format_config"]


async def test_commander_pod_sends_custom_ai_decks(monkeypatch):
    client = PhaseServerClient(PhaseServerConfig())
    sent = []
    decks = [
        {"main_deck": ["Mountain"] * 99, "commander": [f"Commander {seat}"]}
        for seat in range(3)
    ]

    async def send(message_type, payload):
        sent.append(payload)

    async def expect(message_type):
        return GameCreated("pod", "token")

    monkeypatch.setattr(client, "_send", send)
    monkeypatch.setattr(client, "expect", expect)
    await client.create_game_with_ai(deck=decks[0], ai_decks=decks, format_name="Commander")
    payload = sent[0]
    assert payload["player_count"] == 4
    assert payload["format_config"]["starting_life"] == 40
    assert [seat["seatIndex"] for seat in payload["ai_seats"]] == [1, 2, 3]
    assert [seat["deck"]["data"] for seat in payload["ai_seats"]] == decks
    assert all(seat["deck"]["type"] == "DeckList" for seat in payload["ai_seats"])


@pytest.mark.parametrize(("decks", "format_name"), [
    ([], "Commander"), ([{}], "Commander"),
    ([{"main_deck": ["Mountain"]}] * 3, "Modern"),
    ([{"main_deck": ["Mountain"]}] * 3, None),
])
async def test_invalid_ai_seat_configuration_fails_before_sending(decks, format_name):
    client = PhaseServerClient(PhaseServerConfig())
    with pytest.raises(ValueError):
        await client.create_game_with_ai(deck={}, ai_decks=decks, format_name=format_name)
