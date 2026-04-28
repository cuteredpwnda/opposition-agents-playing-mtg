"""Cascade keyword (CR 702.85)."""

from __future__ import annotations

from src.engine.cascade import execute_cascade, has_cascade
from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)


def _state():
    return GameState(
        format="commander",
        turn_number=4,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )


def _card(name, *, mana_cost="", oracle="", typeline="Sorcery", zone=Zone.HAND, owner="A"):
    return CardInstance(
        instance_id=f"{owner}_{name.lower().replace(' ', '_').replace(',', '')}",
        card_data={
            "name": name,
            "mana_cost": mana_cost,
            "type_line": typeline,
            "oracle_text": oracle,
        },
        zone=zone,
        owner_id=owner,
        controller_id=owner,
    )


def test_has_cascade_detects_keyword():
    c = _card("Bloodbraid Elf", oracle="Haste\nCascade")
    assert has_cascade(c) is True
    nc = _card("Vanilla", oracle="Flying")
    assert has_cascade(nc) is False


def test_cascade_finds_lower_mv_card_and_casts_it():
    state = _state()
    # Source: MV 4 cascade spell on the battlefield going to stack.
    source = _card(
        "Bloodbraid Elf", mana_cost="{2}{R}{G}", oracle="Haste\nCascade",
        typeline="Creature — Elf Berserker",
    )
    source.zone = Zone.STACK
    state.cards.append(source)

    # Library top to bottom: a 5-MV (skipped, too big), a land (skipped),
    # then a 1-MV creature (chosen).
    big = _card("Big", mana_cost="{4}{R}", typeline="Sorcery", zone=Zone.LIBRARY)
    land = _card("Forest", mana_cost="", typeline="Basic Land — Forest", zone=Zone.LIBRARY)
    small = _card("Small", mana_cost="{R}", typeline="Sorcery", zone=Zone.LIBRARY)
    state.cards.extend([big, land, small])

    chosen = execute_cascade(state, source)
    assert chosen is not None
    assert chosen.name == "Small"
    assert chosen.zone == Zone.STACK
    assert state.stack[-1].source_card_id == chosen.instance_id
    # The big and land go to the bottom of the library.
    assert big.zone == Zone.LIBRARY
    assert land.zone == Zone.LIBRARY


def test_cascade_no_match_returns_none_but_still_exiles_to_bottom():
    state = _state()
    source = _card(
        "Cheap Cascade", mana_cost="{R}", oracle="Cascade", typeline="Sorcery",
    )
    source.zone = Zone.STACK
    state.cards.append(source)

    # Library top: a 3-MV card (too big to cascade into).
    big = _card("Big", mana_cost="{2}{R}", typeline="Sorcery", zone=Zone.LIBRARY)
    state.cards.append(big)

    chosen = execute_cascade(state, source)
    assert chosen is None
    assert big.zone == Zone.LIBRARY  # back on bottom
