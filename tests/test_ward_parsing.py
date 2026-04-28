"""Ward cost parsing (CR 702.21)."""

from __future__ import annotations

from src.engine.game_state import CardInstance, Zone
from src.engine.keywords import ward_cost


def _card(oracle: str) -> CardInstance:
    return CardInstance(
        instance_id="x",
        card_data={
            "name": "Test",
            "type_line": "Creature",
            "oracle_text": oracle,
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )


def test_ward_numeric():
    assert ward_cost(_card("Ward {2}")) == 2
    assert ward_cost(_card("Flying\nWard {3}")) == 3


def test_ward_coloured_hybrid():
    assert ward_cost(_card("Ward {1}{U}")) == 2
    assert ward_cost(_card("Ward {U}{U}")) == 2


def test_ward_bare():
    assert ward_cost(_card("Ward")) == 1


def test_ward_absent_no_false_positive():
    assert ward_cost(_card("Wardrobe of contents.")) == 0
    assert ward_cost(_card("Forward march!")) == 0
    assert ward_cost(_card("")) == 0
    assert ward_cost(None) == 0


def test_ward_with_other_text():
    assert (
        ward_cost(
            _card("Flying\nWard {2}\nWhen this enters, draw a card.")
        )
        == 2
    )
