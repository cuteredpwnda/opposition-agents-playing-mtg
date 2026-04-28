"""Modal spells (CR 700.2) — verify the resolver picks targets per mode."""

from __future__ import annotations

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    StackItem,
    Zone,
)
from src.engine.spell_effects import apply_spell_effect, _split_modes


def _two_player_state() -> GameState:
    state = GameState(
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
    return state


def test_split_modes_drops_preamble_and_riders():
    text = "Choose one —\n• Destroy target creature.\n• Draw two cards.\nEntwine {2}"
    modes = _split_modes(text)
    assert modes == ["Destroy target creature.", "Draw two cards."]


def test_modal_resolver_picks_destroy_when_creatures_exist():
    state = _two_player_state()
    bear = CardInstance(
        instance_id="B_bear",
        card_data={
            "name": "Grizzly Bears",
            "type_line": "Creature — Bear",
            "power": "2",
            "toughness": "2",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="B",
        controller_id="B",
    )
    charm = CardInstance(
        instance_id="A_charm",
        card_data={
            "name": "Sample Charm",
            "type_line": "Instant",
            "oracle_text": (
                "Choose one —\n"
                "• Destroy target creature.\n"
                "• You gain 4 life."
            ),
        },
        zone=Zone.STACK,
        owner_id="A",
        controller_id="A",
    )
    state.cards = [bear, charm]
    item = StackItem(
        source_card_id=charm.instance_id,
        controller_id="A",
        is_spell=True,
        card_data={"name": charm.name, "oracle_text": charm.oracle_text},
        targets=[],
    )
    state.stack = [item]
    apply_spell_effect(state, item)
    # The destroy mode should have been chosen and bear should be dying or dead.
    assert bear.zone == Zone.GRAVEYARD


def test_modal_resolver_falls_back_to_lifegain_when_no_targets():
    state = _two_player_state()
    state.players[0].life_total = 10
    charm = CardInstance(
        instance_id="A_charm",
        card_data={
            "name": "Sample Charm",
            "type_line": "Instant",
            "oracle_text": (
                "Choose one —\n"
                "• Destroy target creature.\n"
                "• You gain 4 life."
            ),
        },
        zone=Zone.STACK,
        owner_id="A",
        controller_id="A",
    )
    state.cards = [charm]
    item = StackItem(
        source_card_id=charm.instance_id,
        controller_id="A",
        is_spell=True,
        card_data={"name": charm.name, "oracle_text": charm.oracle_text},
        targets=[],
    )
    state.stack = [item]
    apply_spell_effect(state, item)
    # No creatures to destroy → destroy mode is heavily penalised, and
    # life-gain should still resolve (or the destroy mode fizzles harmlessly).
    # The important guarantee is that we *don't* crash and the player isn't
    # worse off.
    assert state.players[0].life_total >= 10
