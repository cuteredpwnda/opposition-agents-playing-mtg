"""Replacement effects framework (CR 614)."""

from __future__ import annotations

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.engine.replacement_effects import (
    apply_damage_to_player,
    apply_lifegain,
    apply_replacements,
    install_replacements_for,
    remove_replacements_for,
)
from src.engine.zones import move_card


def _state():
    return GameState(
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


def _make(name, oracle, controller="A", typeline="Enchantment"):
    return CardInstance(
        instance_id=f"{controller}_{name.lower().replace(' ', '_')}",
        card_data={"name": name, "type_line": typeline, "oracle_text": oracle},
        zone=Zone.BATTLEFIELD,
        owner_id=controller,
        controller_id=controller,
    )


def test_lifegain_doubling():
    state = _state()
    boon = _make(
        "Boon Reflection",
        "If you would gain life, you gain twice that much life instead.",
    )
    state.cards.append(boon)
    install_replacements_for(state, boon)
    gained = apply_lifegain(state, "A", 3)
    assert gained == 6
    assert state.players[0].life_total == 26


def test_combat_damage_prevention():
    state = _state()
    holy = _make(
        "Personal Sanctuary",
        "Prevent all combat damage that would be dealt to you.",
    )
    state.cards.append(holy)
    install_replacements_for(state, holy)
    dealt = apply_damage_to_player(state, "A", 5, combat=True)
    assert dealt == 0
    assert state.players[0].life_total == 20
    # Non-combat still goes through.
    dealt2 = apply_damage_to_player(state, "A", 5, combat=False)
    assert dealt2 == 5
    assert state.players[0].life_total == 15


def test_death_to_exile_replacement():
    state = _state()
    anafenza = _make(
        "Anafenza-like",
        "If a creature would die, exile it instead.",
    )
    state.cards.append(anafenza)
    install_replacements_for(state, anafenza)
    event = apply_replacements(
        state,
        {"type": "creature_dies", "card_id": "x", "destination_zone": Zone.GRAVEYARD},
    )
    assert event["destination_zone"] == Zone.EXILE


def test_replacement_uninstalls_when_card_leaves_battlefield():
    state = _state()
    boon = _make(
        "Boon Reflection",
        "If you would gain life, you gain twice that much life instead.",
    )
    state.cards.append(boon)
    install_replacements_for(state, boon)
    assert apply_lifegain(state, "A", 2) == 4
    state.players[0].life_total = 20
    # Now move it to graveyard and re-check.
    move_card(state, boon.instance_id, Zone.BATTLEFIELD, Zone.GRAVEYARD, "A")
    assert apply_lifegain(state, "A", 2) == 2  # no longer doubled


def test_install_via_move_card_etb():
    state = _state()
    boon = CardInstance(
        instance_id="A_boon",
        card_data={
            "name": "Boon Reflection",
            "type_line": "Enchantment",
            "oracle_text": "If you would gain life, you gain twice that much life instead.",
        },
        zone=Zone.HAND,
        owner_id="A",
        controller_id="A",
    )
    state.cards.append(boon)
    move_card(state, boon.instance_id, Zone.HAND, Zone.BATTLEFIELD, "A")
    # Should be auto-installed by the ETB hook.
    assert apply_lifegain(state, "A", 4) == 8
