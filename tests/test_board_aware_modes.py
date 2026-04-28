"""Board-aware modal mode picking."""

from __future__ import annotations

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.engine.spell_effects import _pick_modes


def _state(*cards):
    s = GameState(
        format="commander",
        turn_number=3,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[
            PlayerState(player_id="A", life_total=20),
            PlayerState(player_id="B", life_total=20),
        ],
    )
    s.cards = list(cards)
    return s


def _bear(owner="B"):
    return CardInstance(
        instance_id=f"{owner}_bear",
        card_data={
            "name": "Grizzly Bears",
            "type_line": "Creature — Bear",
            "power": "2",
            "toughness": "2",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id=owner,
        controller_id=owner,
    )


def test_low_life_prefers_lifegain_over_draw():
    state = _state()
    state.players[0].life_total = 3  # panic mode
    modes = ["Draw two cards.", "You gain 5 life."]
    picks = _pick_modes(state, None, "A", modes, 1)
    assert picks == ["You gain 5 life."]


def test_high_threat_boosts_destroy_mode():
    fattie = CardInstance(
        instance_id="B_dragon",
        card_data={
            "name": "Big Dragon",
            "type_line": "Creature — Dragon",
            "power": "6",
            "toughness": "6",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="B",
        controller_id="B",
    )
    state = _state(fattie)
    modes = ["Destroy target creature.", "You gain 2 life."]
    picks = _pick_modes(state, None, "A", modes, 1)
    assert picks == ["Destroy target creature."]


def test_no_targets_demotes_destroy_mode():
    state = _state()
    modes = ["Destroy target creature.", "Draw a card."]
    picks = _pick_modes(state, None, "A", modes, 1)
    assert picks == ["Draw a card."]


def test_small_damage_against_big_creatures_demoted():
    state = _state(_bear())  # toughness 2
    modes = [
        "Sample deals 1 damage to any target.",
        "Draw a card.",
    ]
    picks = _pick_modes(state, None, "A", modes, 1)
    # 1 damage doesn't kill the bear and opp has 20 life → draw should win.
    assert picks == ["Draw a card."]
