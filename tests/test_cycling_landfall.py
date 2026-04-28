"""Tests for cycling + landfall + enters-tapped."""

import pytest

from src.engine.game_state import (
    ActionType, CardInstance, GameState, PlayerState, Zone,
)
from src.engine.cycling import parse_cycling_cost, execute_cycle
from src.engine.rules_engine import RulesEngine
from src.engine.triggers import check_landfall_triggers


def make_player(pid: str) -> PlayerState:
    p = PlayerState(player_id=pid, name=pid, life_total=20)
    p.land_plays_remaining = 1
    return p


def make_card(pid: str, name: str, oracle: str, type_line: str, zone: Zone) -> CardInstance:
    return CardInstance(
        instance_id=f"{pid}_{name}",
        card_data={
            "name": name, "type_line": type_line, "oracle_text": oracle,
            "mana_cost": "", "set": "TST",
        },
        zone=zone, owner_id=pid, controller_id=pid,
    )


# ---------------------------------------------------------------------------
# Cycling
# ---------------------------------------------------------------------------


def test_parse_cycling_cost():
    card = make_card("p1", "Cyc", "Cycling {1}{U}", "Creature - Bird", Zone.HAND)
    assert parse_cycling_cost(card) == "{1}{U}"


def test_parse_cycling_cost_none():
    card = make_card("p1", "Plain", "Flying", "Creature - Bird", Zone.HAND)
    assert parse_cycling_cost(card) is None


def test_execute_cycle_discards_and_draws():
    p = make_player("p1")
    cyc = make_card("p1", "Cyc", "Cycling {2}", "Creature - Bird", Zone.HAND)
    next_card = make_card("p1", "NextCard", "", "Sorcery", Zone.LIBRARY)

    state = GameState(players=[p, make_player("p2")], cards=[cyc, next_card])
    # Ensure 2 generic mana available
    p.mana_pool["C"] = 2

    state = execute_cycle(state, cyc, p)

    assert cyc.zone == Zone.GRAVEYARD, "cycled card goes to graveyard"
    assert next_card.zone == Zone.HAND, "draws next card"


def test_cycling_appears_in_legal_actions():
    p1 = make_player("p1")
    p2 = make_player("p2")
    p1.mana_pool["C"] = 3
    cyc = make_card("p1", "Cyc", "Cycling {1}", "Creature - Bird", Zone.HAND)
    state = GameState(players=[p1, p2], cards=[cyc])

    engine = RulesEngine()
    actions = engine.get_legal_actions(state, "p1")
    cycling_actions = [
        a for a in actions
        if a.action_type == ActionType.SPECIAL_ACTION
        and (a.metadata or {}).get("special") == "cycle"
    ]
    assert len(cycling_actions) == 1
    assert cycling_actions[0].card_instance_id == cyc.instance_id


# ---------------------------------------------------------------------------
# Landfall
# ---------------------------------------------------------------------------


def test_landfall_trigger_detected():
    p = make_player("p1")
    landfall_perm = make_card(
        "p1", "Lotus Cobra",
        "Landfall — Whenever a land enters the battlefield under your control, add one mana of any color.",
        "Creature - Snake",
        Zone.BATTLEFIELD,
    )
    new_land = make_card("p1", "Forest", "", "Basic Land - Forest", Zone.BATTLEFIELD)
    state = GameState(players=[p, make_player("p2")], cards=[landfall_perm, new_land])

    triggers = check_landfall_triggers(state, "p1", new_land)
    assert len(triggers) == 1
    assert "Lotus Cobra" in triggers[0].description


def test_landfall_only_for_controller():
    p1 = make_player("p1")
    p2 = make_player("p2")
    cobra = make_card(
        "p2", "Lotus Cobra",
        "Landfall — Whenever a land enters the battlefield under your control, add one mana.",
        "Creature - Snake",
        Zone.BATTLEFIELD,
    )
    new_land = make_card("p1", "Forest", "", "Basic Land - Forest", Zone.BATTLEFIELD)
    state = GameState(players=[p1, p2], cards=[cobra, new_land])

    # p1 plays the land — p2's cobra must NOT trigger (controller mismatch)
    triggers = check_landfall_triggers(state, "p1", new_land)
    assert triggers == []


# ---------------------------------------------------------------------------
# Enters tapped
# ---------------------------------------------------------------------------


def test_play_land_with_enters_tapped():
    from src.engine.game_state import Action, Phase
    p = make_player("p1")
    p.land_plays_remaining = 1
    tapped_land = make_card(
        "p1", "Temple of Mystery",
        "Temple of Mystery enters the battlefield tapped. When Temple of Mystery enters, scry 1.",
        "Land",
        Zone.HAND,
    )
    state = GameState(players=[p, make_player("p2")], cards=[tapped_land])
    state.phase = Phase.MAIN_1
    state.active_player_index = 0

    engine = RulesEngine()
    action = Action(
        action_type=ActionType.PLAY_LAND,
        player_id="p1",
        card_instance_id=tapped_land.instance_id,
    )
    state = engine.execute_action(state, action)

    assert tapped_land.zone == Zone.BATTLEFIELD
    assert tapped_land.tapped, "Temple should enter tapped"
