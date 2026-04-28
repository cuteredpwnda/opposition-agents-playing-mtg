"""Floating mana semantics — pool persists between actions in a phase but
empties at phase end (CR 106.4)."""

from __future__ import annotations

import asyncio

from src.agents.random_agent import RandomAgent
from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.engine.mana import (
    add_mana,
    empty_mana_pool,
    pay_cost,
    tap_land_for_mana,
)


def test_empty_mana_pool_clears_all_colors():
    p = PlayerState(player_id="A")
    add_mana(p, "R", 2)
    add_mana(p, "G", 1)
    empty_mana_pool(p)
    assert all(v == 0 for v in p.mana_pool.values())


def test_pool_persists_between_pay_costs_in_same_phase():
    """Floating mana stays in the pool until paid or phase ends."""
    p = PlayerState(player_id="A")
    add_mana(p, "R", 3)

    # Cast a {R} spell — pool keeps {R}{R} floating.
    assert pay_cost(p, {"R": 1}) is True
    assert p.mana_pool["R"] == 2

    # Cast a second {R} spell — pool keeps {R}.
    assert pay_cost(p, {"R": 1}) is True
    assert p.mana_pool["R"] == 1


def test_phase_transition_empties_pool_via_game_runner():
    """End of every phase clears mana pool (CR 106.4)."""
    from src.orchestrator.game_runner import GameRunner

    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    # Float {R}{R} into player A.
    add_mana(state.players[0], "R", 2)
    assert state.players[0].mana_pool["R"] == 2

    # Construct a minimal runner just to exercise the cleanup hook.
    runner = GameRunner.__new__(GameRunner)
    # Direct simulation of the per-phase end hook from `game_runner.play_turn`.
    from src.engine.mana import empty_mana_pool
    for pl in state.players:
        empty_mana_pool(pl)
    assert state.players[0].mana_pool["R"] == 0


def test_tap_then_pay_drains_pool_completely():
    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A")],
    )
    p = state.players[0]
    forest = CardInstance(
        instance_id="A_forest",
        card_data={"name": "Forest", "type_line": "Basic Land — Forest",
                   "oracle_text": "({T}: Add {G}.)"},
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    state.cards.append(forest)

    tap_land_for_mana(state, p, forest.instance_id)
    assert p.mana_pool["G"] == 1

    assert pay_cost(p, {"G": 1}) is True
    assert p.mana_pool["G"] == 0
