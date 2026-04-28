"""Smarter proliferate target selection."""

from __future__ import annotations

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    StackItem,
    Zone,
)
from src.engine.spell_effects import apply_spell_effect


def _state_with(*cards: CardInstance) -> GameState:
    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    state.cards = list(cards)
    return state


def _proliferate_item(controller: str = "A") -> StackItem:
    return StackItem(
        source_card_id="A_inexorable_tide",
        controller_id=controller,
        is_spell=True,
        card_data={
            "name": "Tezzeret's Gambit",
            "type_line": "Sorcery",
            "oracle_text": "Draw two cards. Proliferate.",
        },
    )


def test_proliferate_skips_minus_counter_on_own_creature():
    """Adding -1/-1 to our own dork would kill it; skip."""
    dork = CardInstance(
        instance_id="A_dork",
        card_data={
            "name": "Phyrexian Hulk",
            "type_line": "Creature",
            "power": "5",
            "toughness": "5",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    dork.counters["-1/-1"] = 1
    dork.counters["+1/+1"] = 2
    state = _state_with(dork)
    apply_spell_effect(state, _proliferate_item())
    assert dork.counters["-1/-1"] == 1  # not bumped
    assert dork.counters["+1/+1"] == 3  # bumped


def test_proliferate_bumps_opponent_minus_counter():
    foe = CardInstance(
        instance_id="B_thug",
        card_data={
            "name": "Foe",
            "type_line": "Creature",
            "power": "3",
            "toughness": "3",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="B",
        controller_id="B",
    )
    foe.counters["-1/-1"] = 1
    state = _state_with(foe)
    apply_spell_effect(state, _proliferate_item())
    assert foe.counters["-1/-1"] == 2


def test_proliferate_does_not_pump_opponent_plus_counter():
    foe = CardInstance(
        instance_id="B_walker",
        card_data={
            "name": "Foe Walker",
            "type_line": "Creature",
            "power": "1",
            "toughness": "1",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="B",
        controller_id="B",
    )
    foe.counters["+1/+1"] = 1
    state = _state_with(foe)
    apply_spell_effect(state, _proliferate_item())
    assert foe.counters["+1/+1"] == 1  # untouched


def test_proliferate_bumps_opponent_poison():
    state = _state_with()
    state.players[1].poison_counters = 2
    apply_spell_effect(state, _proliferate_item())
    assert state.players[1].poison_counters == 3
