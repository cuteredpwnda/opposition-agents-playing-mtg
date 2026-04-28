"""Companion (CR 702.139) — reveal at game start, tax {3} once to grab it."""

from __future__ import annotations

from src.engine.command_zone import command_zone_objects_for
from src.engine.companion import (
    COMPANION_TAX,
    get_companion,
    pay_companion_tax,
    reveal_companion,
)
from src.engine.game_state import GameState, Phase, PlayerState, Zone


def _state(*pids: str) -> GameState:
    return GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id=p) for p in pids],
    )


def _lurrus_card_data() -> dict:
    return {
        "name": "Lurrus of the Dream-Den",
        "mana_cost": "{W}{B}",
        "cmc": 3.0,
        "type_line": "Legendary Creature — Cat Nightmare",
        "oracle_text": "Companion — Each permanent card in your starting deck has mana value 2 or less.",
        "colors": ["W", "B"],
    }


def test_companion_tax_constant():
    assert COMPANION_TAX == 3


def test_reveal_companion_creates_outside_game_marker():
    s = _state("A", "B")
    card = reveal_companion(s, "A", _lurrus_card_data())

    # Card lives in the EXILE proxy zone with the companion flag.
    assert card.zone == Zone.EXILE
    assert card.card_data["companion"] is True
    assert card.controller_id == "A"

    # Tracking object exists for player A only.
    objs = command_zone_objects_for(s, "A", kind="companion")
    assert len(objs) == 1
    assert objs[0].name == "Lurrus of the Dream-Den"
    assert objs[0].card_data["tax_paid"] is False
    assert command_zone_objects_for(s, "B", kind="companion") == []

    assert get_companion(s, "A") is card


def test_pay_companion_tax_moves_card_to_hand_once():
    s = _state("A", "B")
    card = reveal_companion(s, "A", _lurrus_card_data())

    moved = pay_companion_tax(s, "A")
    assert moved is card
    assert card.zone == Zone.HAND

    objs = command_zone_objects_for(s, "A", kind="companion")
    assert objs[0].card_data["tax_paid"] is True

    # Second attempt is a no-op.
    assert pay_companion_tax(s, "A") is None


def test_pay_companion_tax_with_no_companion_returns_none():
    s = _state("A", "B")
    assert pay_companion_tax(s, "A") is None


def test_two_players_can_each_have_their_own_companion():
    s = _state("A", "B")
    a_card = reveal_companion(s, "A", _lurrus_card_data())
    b_card = reveal_companion(
        s,
        "B",
        dict(_lurrus_card_data(), name="Yorion, Sky Nomad"),
    )

    assert get_companion(s, "A") is a_card
    assert get_companion(s, "B") is b_card
    assert a_card is not b_card
