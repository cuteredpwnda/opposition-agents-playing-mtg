from __future__ import annotations

import pytest

from src.engine.game_state import ActionType
from src.integrations.phase_rs.adapter import (
    engine_action_to_phase_action,
    legal_actions_to_engine_actions,
    phase_state_to_game_state,
)


def test_legal_actions_to_engine_actions_maps_common_types():
    legal = [
        {"type": "PlayLand", "data": {"card_id": 11}},
        {"type": "PassPriority"},
        {"type": "DeclareAttackers", "data": {"attacker_ids": [21, 22]}},
    ]
    out = legal_actions_to_engine_actions(legal, seat=0)
    assert [a.action_type for a in out] == [
        ActionType.PLAY_LAND,
        ActionType.PASS_PRIORITY,
        ActionType.DECLARE_ATTACKERS,
    ]
    assert out[0].card_instance_id == "11"
    assert out[2].targets == ["21", "22"]


def test_engine_action_to_phase_action_roundtrip_uses_index_metadata():
    legal = [
        {"type": "PassPriority"},
        {"type": "Concede"},
    ]
    actions = legal_actions_to_engine_actions(legal, seat=0)
    chosen = actions[0]
    raw = engine_action_to_phase_action(chosen, legal)
    assert raw == {"type": "PassPriority"}


def test_phase_state_to_game_state_builds_minimal_projection():
    state = {
        "turn_number": 3,
        "phase": "PreCombatMain",
        "active_player": 0,
        "priority_player": 1,
        "players": [
            {"id": 0, "life": 20},
            {"id": 1, "life": 17},
        ],
        "objects": {
            "42": {
                "name": "Goblin Guide",
                "type_line": "Creature — Goblin Scout",
                "zone": "battlefield",
                "controller": 0,
                "owner": 0,
                "power": 2,
                "toughness": 2,
            }
        },
        "stack": [],
    }
    gs = phase_state_to_game_state(state)
    assert gs.turn_number == 3
    assert gs.active_player.player_id == "seat0"
    assert gs.priority_player.player_id == "seat1"
    assert len(gs.cards) == 1
    assert gs.cards[0].name == "Goblin Guide"


@pytest.mark.parametrize("kind", ["Spell", "ActivatedAbility", "TriggeredAbility"])
def test_native_stack_entries_preserve_source_controller_and_kind(kind):
    state = {
        "players": [{"id": 0}, {"id": 1}],
        "objects": {"42": {"name": "Lightning Bolt", "owner": 0, "controller": 0}},
        "stack": [{"id": 99, "source_id": 42, "controller": 1,
                   "kind": {"type": kind, "data": {}}}],
    }
    stack = phase_state_to_game_state(state).stack
    assert len(stack) == 1
    assert stack[0].source_card_id == "42"
    assert stack[0].controller_id == "seat1"
    assert stack[0].is_spell is (kind == "Spell")
    assert stack[0].card_data["name"] == "Lightning Bolt"


@pytest.mark.parametrize("stack", [[{}], [{"source_id": 42}], [True], [[]]])
def test_malformed_stack_is_explicit(stack):
    with pytest.raises(ValueError, match="stack entry"):
        phase_state_to_game_state({"players": [], "objects": {}, "stack": stack})


@pytest.mark.parametrize("source", [42, "42"])
def test_legacy_scalar_stack_remains_supported(source):
    state = {
        "players": [{"id": 0}],
        "objects": {"42": {"name": "Lightning Bolt", "controller": 0}},
        "stack": [source],
    }
    item = phase_state_to_game_state(state).stack[0]
    assert item.source_card_id == "42"
    assert item.controller_id == "seat0"
    assert item.is_spell
