"""Additional state-based actions: planeswalker loyalty, legendary rule,
aura unattached."""

from __future__ import annotations

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.engine.rules_engine import RulesEngine


def _state(*cards: CardInstance) -> GameState:
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


def test_planeswalker_with_zero_loyalty_dies():
    pw = CardInstance(
        instance_id="A_pw",
        card_data={
            "name": "Liliana, Heretical Healer",
            "type_line": "Legendary Planeswalker — Liliana",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    pw.counters["loyalty"] = 0
    state = _state(pw)
    RulesEngine().check_state_based_actions(state)
    assert pw.zone == Zone.GRAVEYARD


def test_legendary_rule_sacrifices_duplicates():
    a = CardInstance(
        instance_id="A_atraxa1",
        card_data={
            "name": "Atraxa, Praetors' Voice",
            "type_line": "Legendary Creature — Phyrexian Angel Horror",
            "power": "4",
            "toughness": "4",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    b = CardInstance(
        instance_id="A_atraxa2",
        card_data={
            "name": "Atraxa, Praetors' Voice",
            "type_line": "Legendary Creature — Phyrexian Angel Horror",
            "power": "4",
            "toughness": "4",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    a.turn_entered = 1
    b.turn_entered = 5  # newer
    state = _state(a, b)
    RulesEngine().check_state_based_actions(state)
    # Newer one is kept.
    assert b.zone == Zone.BATTLEFIELD
    assert a.zone == Zone.GRAVEYARD


def test_legendary_rule_does_not_kill_across_controllers():
    a = CardInstance(
        instance_id="A_jace",
        card_data={
            "name": "Jace, the Mind Sculptor",
            "type_line": "Legendary Planeswalker — Jace",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    a.counters["loyalty"] = 3
    b = CardInstance(
        instance_id="B_jace",
        card_data={
            "name": "Jace, the Mind Sculptor",
            "type_line": "Legendary Planeswalker — Jace",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="B",
        controller_id="B",
    )
    b.counters["loyalty"] = 3
    state = _state(a, b)
    RulesEngine().check_state_based_actions(state)
    assert a.zone == Zone.BATTLEFIELD
    assert b.zone == Zone.BATTLEFIELD


def test_aura_with_no_target_goes_to_graveyard():
    aura = CardInstance(
        instance_id="A_pacifism",
        card_data={
            "name": "Pacifism",
            "type_line": "Enchantment — Aura",
            "oracle_text": "Enchant creature\nEnchanted creature can't attack or block.",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    aura.attached_to = None
    state = _state(aura)
    RulesEngine().check_state_based_actions(state)
    assert aura.zone == Zone.GRAVEYARD


def test_aura_with_dead_target_falls_off():
    bear = CardInstance(
        instance_id="B_bear",
        card_data={
            "name": "Grizzly Bears",
            "type_line": "Creature — Bear",
            "power": "2",
            "toughness": "2",
            "oracle_text": "",
        },
        zone=Zone.GRAVEYARD,  # already gone
        owner_id="B",
        controller_id="B",
    )
    aura = CardInstance(
        instance_id="A_pacifism",
        card_data={
            "name": "Pacifism",
            "type_line": "Enchantment — Aura",
            "oracle_text": "Enchant creature\nEnchanted creature can't attack or block.",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    aura.attached_to = bear.instance_id
    state = _state(aura, bear)
    RulesEngine().check_state_based_actions(state)
    assert aura.zone == Zone.GRAVEYARD
