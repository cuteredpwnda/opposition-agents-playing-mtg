from __future__ import annotations

import copy

import pytest

from src.integrations.phase_rs.runner import _our_simultaneous_round_key


def test_native_mulligan_phase_advances_without_count_change():
    waiting = {
        "type": "MulliganDecision",
        "data": {
            "pending": [
                {"player": 0, "mulligan_count": 1, "phase": {"type": "Decision"}},
                {"player": 1, "mulligan_count": 0, "phase": {"type": "Decision"}},
            ],
        },
    }
    decision_key = _our_simultaneous_round_key(waiting, 0)
    bottoming = copy.deepcopy(waiting)
    bottoming["data"]["pending"][0]["phase"] = {
        "type": "BottomCards", "count": 1, "then": {"type": "Keep"},
    }
    assert decision_key != _our_simultaneous_round_key(bottoming, 0)

    other_seat_acted = copy.deepcopy(bottoming)
    other_seat_acted["data"]["pending"].pop()
    assert (
        _our_simultaneous_round_key(bottoming, 0)
        == _our_simultaneous_round_key(other_seat_acted, 0)
    )


@pytest.mark.parametrize("variant", ["MulliganDecision", "MulliganBottomCards"])
def test_legacy_simultaneous_key_ignores_other_seat_choices(variant):
    waiting = {
        "type": variant,
        "data": {"pending": [
            {"player": 0, "mulligan_count": 0},
            {"player": 1, "mulligan_count": 0, "chosen": False},
        ]},
    }
    key = _our_simultaneous_round_key(waiting, 0)
    waiting["data"]["pending"][1]["chosen"] = True
    assert key == _our_simultaneous_round_key(waiting, 0)
    waiting["data"]["pending"][0]["mulligan_count"] += 1
    assert key != _our_simultaneous_round_key(waiting, 0)


@pytest.mark.parametrize("waiting", [
    None,
    {"type": "Priority"},
    {"type": "MulliganDecision", "data": {"pending": [{"player": 1}]}},
])
def test_no_simultaneous_key_without_our_pending_entry(waiting):
    assert _our_simultaneous_round_key(waiting, 0) is None
