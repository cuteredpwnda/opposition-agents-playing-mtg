from __future__ import annotations

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
