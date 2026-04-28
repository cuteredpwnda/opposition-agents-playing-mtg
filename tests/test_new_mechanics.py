"""Tests for the new mechanic implementations:

- Token factory (Treasure / Food / Clue / generic creature)
- Counter SBAs (+1/+1 vs -1/-1, poison, stun)
- Equip / Crew / Flashback / Kicker
- Upkeep-step trigger scanning
"""

from __future__ import annotations

import pytest

from src.engine.game_state import (
    ActionType, CardInstance, GameState, Phase, PlayerState, Zone,
)
from src.engine import tokens as tok_mod
from src.engine import counters as cnt_mod
from src.engine import equip as eq_mod
from src.engine.phases import advance_phase, _scan_phase_triggers
from src.engine.game_state import TriggerType
from src.engine.rules_engine import RulesEngine


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_player(pid: str) -> PlayerState:
    return PlayerState(player_id=pid, name=pid, life_total=20)


def make_card(
    pid: str, name: str, oracle: str, type_line: str, zone: Zone,
    *, power: str | None = None, toughness: str | None = None, mana_cost: str = "",
) -> CardInstance:
    data = {
        "name": name, "type_line": type_line, "oracle_text": oracle,
        "mana_cost": mana_cost, "set": "TST",
    }
    if power is not None:
        data["power"] = power
    if toughness is not None:
        data["toughness"] = toughness
    return CardInstance(
        instance_id=f"{pid}_{name}",
        card_data=data,
        zone=zone, owner_id=pid, controller_id=pid,
    )


def make_state(*cards: CardInstance) -> GameState:
    p1, p2 = make_player("p1"), make_player("p2")
    return GameState(players=[p1, p2], cards=list(cards))


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_create_treasure_token():
    state = make_state()
    t = tok_mod.create_treasure_token(state, "p1")
    assert t.zone == Zone.BATTLEFIELD
    assert t.controller_id == "p1"
    assert "Treasure" in t.type_line
    assert t.card_data.get("is_token") is True
    assert t in state.cards


def test_sacrifice_treasure_for_mana():
    state = make_state()
    t = tok_mod.create_treasure_token(state, "p1")
    p1 = state.players[0]
    assert tok_mod.sacrifice_for_mana(state, t, "U") is True
    assert t.zone == Zone.GRAVEYARD
    assert p1.mana_pool["U"] == 1


def test_create_creature_token():
    state = make_state()
    t = tok_mod.create_creature_token(
        state, "p1", power=2, toughness=2, subtypes=["Soldier"], colors=["W"],
    )
    assert t.is_creature()
    assert t.power == "2" and t.toughness == "2"
    assert "Soldier" in t.type_line


# ---------------------------------------------------------------------------
# Counter SBAs
# ---------------------------------------------------------------------------


def test_plus_minus_counters_cancel():
    c = make_card("p1", "Bear", "", "Creature - Bear", Zone.BATTLEFIELD,
                  power="2", toughness="2")
    c.counters["+1/+1"] = 2
    c.counters["-1/-1"] = 1
    state = make_state(c)
    events = cnt_mod.apply_counter_sbas(state)
    assert c.counters.get("+1/+1", 0) == 1
    assert c.counters.get("-1/-1", 0) == 0
    assert any("cancel" in e for e in events)


def test_poison_loses_at_ten():
    state = make_state()
    state.players[0].poison_counters = 10
    events = cnt_mod.apply_counter_sbas(state)
    assert len(state.players) == 1
    assert state.players[0].player_id == "p2"
    assert any("poison" in e.lower() for e in events)


def test_stun_skips_untap():
    c = make_card("p1", "Bear", "", "Creature - Bear", Zone.BATTLEFIELD,
                  power="2", toughness="2")
    c.tapped = True
    c.counters["stun"] = 1
    skipped = cnt_mod.apply_stun_on_untap(c)
    assert skipped is True
    assert c.counters.get("stun", 0) == 0
    # Still tapped — caller is responsible for not untapping.
    assert c.tapped is True


def test_effective_toughness_with_counters():
    c = make_card("p1", "Bear", "", "Creature - Bear", Zone.BATTLEFIELD,
                  power="2", toughness="2")
    c.counters["+1/+1"] = 1
    assert cnt_mod.effective_toughness(c) == 3
    c.counters["-1/-1"] = 2
    # SBA cancels first
    cnt_mod.apply_counter_sbas(make_state(c))
    assert cnt_mod.effective_toughness(c) == 1  # 2 + 0 - 1


# ---------------------------------------------------------------------------
# Equip
# ---------------------------------------------------------------------------


def test_parse_equip_cost_numeric():
    e = make_card("p1", "Sword", "Equipped creature gets +2/+2.\nEquip 2",
                  "Artifact - Equipment", Zone.BATTLEFIELD)
    assert eq_mod.parse_equip_cost(e) == "{2}"


def test_execute_equip_attaches_and_buffs():
    e = make_card("p1", "Sword", "Equipped creature gets +2/+1.\nEquip {2}",
                  "Artifact - Equipment", Zone.BATTLEFIELD)
    bear = make_card("p1", "Bear", "", "Creature - Bear", Zone.BATTLEFIELD,
                     power="2", toughness="2")
    state = make_state(e, bear)
    state.players[0].mana_pool["C"] = 2
    ok = eq_mod.execute_equip(state, e, bear)
    assert ok is True
    assert e.attached_to == bear.instance_id
    assert bear.counters.get("equip_pwr") == 2
    assert bear.counters.get("equip_tou") == 1


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------


def test_can_crew_meets_threshold():
    veh = make_card("p1", "Truck", "Crew 3", "Artifact - Vehicle",
                    Zone.BATTLEFIELD, power="4", toughness="4")
    a = make_card("p1", "A", "", "Creature - Soldier", Zone.BATTLEFIELD,
                  power="2", toughness="2")
    b = make_card("p1", "B", "", "Creature - Soldier", Zone.BATTLEFIELD,
                  power="2", toughness="2")
    a.summoning_sick = False
    b.summoning_sick = False
    state = make_state(veh, a, b)
    assert eq_mod.can_crew(state, veh, [a, b]) is True


def test_execute_crew_taps_and_grants_creature_type():
    veh = make_card("p1", "Truck", "Crew 3", "Artifact - Vehicle",
                    Zone.BATTLEFIELD, power="4", toughness="4")
    a = make_card("p1", "A", "", "Creature - Soldier", Zone.BATTLEFIELD,
                  power="2", toughness="2")
    b = make_card("p1", "B", "", "Creature - Soldier", Zone.BATTLEFIELD,
                  power="2", toughness="2")
    a.summoning_sick = b.summoning_sick = False
    state = make_state(veh, a, b)
    assert eq_mod.execute_crew(state, veh, [a, b]) is True
    assert a.tapped and b.tapped
    assert "Creature" in veh.type_line
    assert veh.counters.get("crewed_eot") == 1
    eq_mod.clear_crew_eot(state)
    assert "Creature" not in veh.type_line


# ---------------------------------------------------------------------------
# Flashback
# ---------------------------------------------------------------------------


def test_flashback_legal_action_from_graveyard():
    p1 = make_player("p1")
    p1.mana_pool["R"] = 1
    p1.mana_pool["C"] = 2
    fb = make_card(
        "p1", "Pyro", "Deal 3 damage to any target.\nFlashback {2}{R}",
        "Sorcery", Zone.GRAVEYARD, mana_cost="{R}",
    )
    state = GameState(players=[p1, make_player("p2")], cards=[fb])
    state.phase = Phase.MAIN_1
    state.active_player_index = 0
    state.priority_player_index = 0
    engine = RulesEngine()
    actions = engine.get_legal_actions(state, "p1")
    fb_actions = [
        a for a in actions
        if a.action_type == ActionType.SPECIAL_ACTION
        and (a.metadata or {}).get("special") == "flashback"
    ]
    assert len(fb_actions) == 1


# ---------------------------------------------------------------------------
# Kicker
# ---------------------------------------------------------------------------


def test_kicker_surfaces_extra_cast_action():
    p1 = make_player("p1")
    # Enough mana for both base {R} and kicker {2}{R}.
    p1.mana_pool["R"] = 2
    p1.mana_pool["C"] = 2
    spell = make_card(
        "p1", "Bolt", "Deal 3 damage to any target.\nKicker {2}{R}",
        "Sorcery", Zone.HAND, mana_cost="{R}",
    )
    state = GameState(players=[p1, make_player("p2")], cards=[spell])
    state.phase = Phase.MAIN_1
    state.active_player_index = 0
    state.priority_player_index = 0
    engine = RulesEngine()
    actions = engine.get_legal_actions(state, "p1")
    kicker_actions = [
        a for a in actions
        if a.action_type == ActionType.CAST_SPELL
        and (a.metadata or {}).get("kicked") is True
    ]
    assert len(kicker_actions) == 1


# ---------------------------------------------------------------------------
# Upkeep trigger scanning
# ---------------------------------------------------------------------------


def test_upkeep_trigger_pushes_stack_item():
    src = make_card(
        "p1", "Howler", "At the beginning of your upkeep, draw a card.",
        "Enchantment", Zone.BATTLEFIELD,
    )
    state = make_state(src)
    state.active_player_index = 0
    initial = len(state.stack)
    _scan_phase_triggers(state, "upkeep", TriggerType.UPKEEP)
    assert len(state.stack) == initial + 1
    assert "Howler" in state.stack[-1].card_data["name"]
