from __future__ import annotations

import copy
import json

import pytest

from src.integrations.phase_rs.decision_context import decision_context


def test_context_preserves_card_identity_and_board_without_internal_bulk():
    state = {
        "turn_number": 3,
        "waiting_for": {"type": "Priority", "data": {"player": 0}},
        "players": [
            {"id": 0, "life": 20, "hand": [1], "library": [3, 4]},
            {"id": 1, "life": 12, "hand": [2], "library": [5]},
        ],
        "battlefield": [6],
        "stack": [7],
        "objects": {
            "1": {"id": 1, "name": "Lightning Bolt", "zone": "Hand",
                  "oracle_text": "Deal 3 damage.", "mana_cost": {"generic": 0}},
            "2": {"id": 2, "name": "Hidden Card", "zone": "Hand"},
            "6": {"id": 6, "name": "Goblin Guide", "zone": "Battlefield",
                  "tapped": True, "power": 2, "toughness": 2, "controller": 0,
                  "counters": {"PlusOnePlusOne": 1}, "keywords": ["Haste"],
                  "trigger_definitions": ["x" * 100_000]},
            "7": {"id": 7, "name": "Shock", "zone": "Stack"},
        },
        "resolved_rules_journal": ["x" * 100_000],
        "deck_pools": {"private": "x" * 100_000},
    }
    original = copy.deepcopy(state)
    context = decision_context(state, 0)
    assert context["players"][0]["hand"] == [1]
    assert context["players"][1]["hand_size"] == 1
    assert "hand" not in context["players"][1]
    assert context["players"][0]["library_size"] == 2
    assert "library" not in context["players"][0]
    assert "2" not in context["objects"]
    bolt = dict(zip(context["object_fields"], context["objects"]["1"]))
    guide = dict(zip(context["object_fields"], context["objects"]["6"]))
    assert bolt["oracle_text"] == "Deal 3 damage."
    assert guide["tapped"] is True
    assert guide["keywords"] == ["Haste"]
    assert context["battlefield"] == [6] and context["stack"] == [7]
    assert "trigger_definitions" not in context["object_fields"]
    assert len(json.dumps(context).encode()) < 2048
    assert state == original
    guide["counters"].clear()
    assert state == original


@pytest.mark.parametrize("state", [
    {"players": {}}, {"players": [None]}, {"players": [{"hand": "invalid"}]},
    {"objects": []}, {"objects": {"1": None}}, {"objects": {"1,2": {}}},
])
def test_invalid_native_shapes_fail_explicitly(state):
    with pytest.raises(ValueError, match="Native"):
        decision_context(state, 0)


def test_identical_instances_are_grouped_without_losing_ids_or_characteristics():
    card = {"name": "Goblin", "zone": "Battlefield", "controller": 0, "power": 0,
            "toughness": 1, "tapped": False, "counters": {"PlusOnePlusOne": 2}}
    state = {"objects": {
        "1": {"id": 1, **card}, "2": {"id": 2, **card},
        "3": {"id": 3, **card, "tapped": True},
    }}
    context = decision_context(state, 0)
    assert set(context["objects"]) == {"1,2", "3"}
    shared = dict(zip(context["object_fields"], context["objects"]["1,2"]))
    tapped = dict(zip(context["object_fields"], context["objects"]["3"]))
    assert shared["power"] == 0
    assert shared["counters"] == {"PlusOnePlusOne": 2}
    assert shared["tapped"] is None and tapped["tapped"] is True
    assert {oid for key in context["objects"] for oid in key.split(",")} == {"1", "2", "3"}
