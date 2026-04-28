"""Equip activation: bonus applies to current target, clears on re-equip."""

from __future__ import annotations

from src.engine.equip import execute_equip
from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.engine.keywords import effective_power, effective_toughness
from src.engine.mana import add_mana


def _state() -> GameState:
    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    return state


def _bonesplitter() -> CardInstance:
    return CardInstance(
        instance_id="A_bonesplitter",
        card_data={
            "name": "Bonesplitter",
            "type_line": "Artifact — Equipment",
            "oracle_text": "Equipped creature gets +3/+0.\nEquip {1}",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )


def _bear(iid: str) -> CardInstance:
    return CardInstance(
        instance_id=iid,
        card_data={
            "name": "Grizzly Bears",
            "type_line": "Creature — Bear",
            "power": "2",
            "toughness": "2",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )


def test_equip_grants_bonus_to_target():
    state = _state()
    eq = _bonesplitter()
    bear = _bear("A_bear")
    state.cards.extend([eq, bear])
    add_mana(state.players[0], "C", 1)

    assert execute_equip(state, eq, bear) is True
    assert eq.attached_to == bear.instance_id
    assert effective_power(bear) == 5
    assert effective_toughness(bear) == 2


def test_re_equip_clears_old_target_bonus():
    state = _state()
    eq = _bonesplitter()
    bear1 = _bear("A_bear1")
    bear2 = _bear("A_bear2")
    state.cards.extend([eq, bear1, bear2])
    add_mana(state.players[0], "C", 2)

    assert execute_equip(state, eq, bear1) is True
    assert effective_power(bear1) == 5

    assert execute_equip(state, eq, bear2) is True
    assert effective_power(bear2) == 5
    # Old target no longer carries the bonus.
    assert effective_power(bear1) == 2
