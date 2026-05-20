"""Offline tests for the :class:`PhaseRsStateView` projection.

These read the committed phase-rs adapter-contract fixtures and verify
that the typed view exposes the fields heuristics actually want, without
needing a running ``phase-server``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.integrations.phase_rs.state_view import PhaseRsStateView

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "external"
    / "phase-rs"
    / "fixtures"
    / "adapter-contract"
)


def _load(name: str) -> dict:
    path = FIXTURE_ROOT / name
    if not path.exists():
        pytest.skip(
            f"phase-rs submodule fixture missing at {path!s}; "
            "run: git submodule update --init external/phase-rs"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def test_state_view_from_game_started_snapshot():
    msg = _load("game_started.json")
    data = msg["data"]
    view = PhaseRsStateView.from_snapshot(
        state=data["state"],
        legal_actions=data.get("legal_actions", []),
        our_seat=0,
    )

    assert view.our_seat == 0
    assert view.turn_number == 1
    assert view.phase == "PreCombatMain"
    assert view.active_player == 0
    assert view.priority_player == 0
    assert view.waiting_for_type == "Priority"
    assert view.me.seat == 0
    assert view.me.life == 20
    assert view.me.hand_size == 0  # fixture starts with empty libraries / hands
    assert len(view.opponents) == 1
    assert view.opponents[0].seat == 1
    assert view.opponents[0].life == 20
    assert view.is_my_turn
    assert view.i_have_priority
    assert view.is_main_phase
    assert not view.is_combat_phase
    assert view.stack_is_empty
    assert view.raw is data["state"]  # zero-copy passthrough


def test_state_view_orients_around_opponent_seat():
    msg = _load("game_started.json")
    data = msg["data"]
    view = PhaseRsStateView.from_snapshot(
        state=data["state"],
        legal_actions=[],
        our_seat=1,
    )
    assert view.me.seat == 1
    assert {opp.seat for opp in view.opponents} == {0}
    assert not view.is_my_turn  # active_player=0, our_seat=1
    assert not view.i_have_priority


def test_state_view_summarises_legal_action_types():
    view = PhaseRsStateView.from_snapshot(
        state={"players": [{"id": 0}], "turn_number": 3, "phase": "DeclareAttackers"},
        legal_actions=[
            {"type": "PlayLand", "data": {"card_id": 42}},
            {"type": "PassPriority"},
            {"type": "CastSpell", "data": {}},
        ],
        our_seat=0,
    )
    assert view.legal_action_types == ("PlayLand", "PassPriority", "CastSpell")
    assert view.can_play_land
    assert view.is_combat_phase
    assert not view.is_main_phase
