"""Echo cost decision (CR 702.50)."""

from __future__ import annotations

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.engine.triggers import resolve_trigger
from src.engine.game_state import Trigger, TriggerType


def _state_with_creature(card: CardInstance, mana: dict[str, int]) -> GameState:
    s = GameState(
        format="commander",
        turn_number=4,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.UPKEEP,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    s.players[0].mana_pool = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0,
                              **mana}
    s.cards = [card]
    return s


def _make_echo_creature(name, power, toughness, oracle, cost_text="{2}{R}"):
    return CardInstance(
        instance_id=f"A_{name.lower().replace(' ', '_')}",
        card_data={
            "name": name,
            "type_line": "Creature — Beast",
            "power": str(power),
            "toughness": str(toughness),
            "oracle_text": oracle,
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )


def _echo_trigger(card: CardInstance) -> Trigger:
    return Trigger(
        source_card_id=card.instance_id,
        controller_id=card.controller_id,
        trigger_type=TriggerType.UPKEEP,
        description="At the beginning of your upkeep, sacrifice it unless you pay its echo cost.",
    )


def test_pays_echo_for_high_power_creature():
    card = _make_echo_creature(
        "Big Beast", 5, 5, "Echo {2}{R}\nFlying\nTrample", cost_text="{2}{R}",
    )
    state = _state_with_creature(card, mana={"R": 1, "C": 5})
    resolve_trigger(state, _echo_trigger(card))
    assert card.zone == Zone.BATTLEFIELD


def test_skips_echo_for_vanilla_low_power():
    card = _make_echo_creature("Vanilla 1/1", 1, 1, "Echo {1}{R}")
    state = _state_with_creature(card, mana={"R": 1, "C": 5})
    resolve_trigger(state, _echo_trigger(card))
    # 1 power, no keywords, no abilities → worth=1 < cmc=2 → sacrificed.
    assert card.zone == Zone.GRAVEYARD


def test_pays_echo_for_card_with_activated_ability_text():
    card = _make_echo_creature(
        "Utility 2/2", 2, 2,
        "Echo {1}{R}\n{T}: Add {R}.",
    )
    state = _state_with_creature(card, mana={"R": 1, "C": 5})
    resolve_trigger(state, _echo_trigger(card))
    # power=2 + non_trivial_text bonus 3 → 5 ≥ cmc=2 → pay.
    assert card.zone == Zone.BATTLEFIELD
