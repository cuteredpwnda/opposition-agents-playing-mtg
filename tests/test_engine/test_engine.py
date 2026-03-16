"""Tests for the game engine."""

import pytest

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
)
from src.engine.mana import can_pay, parse_mana_cost, pay_cost
from src.engine.phases import PHASE_ORDER, advance_phase, is_main_phase


# ---------------------------------------------------------------------------
# Phase tests
# ---------------------------------------------------------------------------

class TestPhases:
    def test_phase_order_length(self):
        assert len(PHASE_ORDER) == 12

    def test_advance_phase_wraps(self):
        gs = GameState(
            players={"p1": PlayerState(), "p2": PlayerState()},
            active_player="p1",
            priority_player="p1",
            phase=Phase.CLEANUP,
            turn_number=1,
        )
        gs = advance_phase(gs)
        assert gs.phase == Phase.UNTAP
        assert gs.turn_number == 2

    def test_is_main_phase(self):
        assert is_main_phase(Phase.MAIN_1)
        assert is_main_phase(Phase.MAIN_2)
        assert not is_main_phase(Phase.COMBAT_DAMAGE)


# ---------------------------------------------------------------------------
# Mana tests
# ---------------------------------------------------------------------------

class TestMana:
    def test_parse_mana_cost_simple(self):
        cost = parse_mana_cost("{2}{U}{U}")
        assert cost == {"generic": 2, "U": 2}

    def test_parse_mana_cost_empty(self):
        cost = parse_mana_cost("")
        assert cost == {}

    def test_can_pay_sufficient(self):
        pool = {"U": 3, "W": 1}
        cost = {"generic": 1, "U": 2}
        assert can_pay(pool, cost)

    def test_can_pay_insufficient(self):
        pool = {"U": 1}
        cost = {"generic": 1, "U": 2}
        assert not can_pay(pool, cost)

    def test_pay_cost_deducts(self):
        pool = {"U": 3, "W": 1}
        cost = {"generic": 1, "U": 2}
        remaining = pay_cost(dict(pool), cost)
        assert remaining["U"] == 1
        assert remaining["W"] == 0  # used for generic


# ---------------------------------------------------------------------------
# Game state tests
# ---------------------------------------------------------------------------

class TestGameState:
    def _make_gs(self) -> GameState:
        return GameState(
            players={
                "p1": PlayerState(life=40),
                "p2": PlayerState(life=40),
            },
            active_player="p1",
            priority_player="p1",
            phase=Phase.MAIN_1,
            turn_number=1,
            cards_in_zone={
                ("p1", "hand"): [
                    CardInstance(
                        instance_id="c1", owner="p1", controller="p1",
                        name="Lightning Bolt", oracle_text="Deal 3 damage.",
                        type_line="Instant", mana_cost="{R}", cmc=1,
                    )
                ],
                ("p1", "battlefield"): [],
                ("p2", "hand"): [],
                ("p2", "battlefield"): [],
            },
        )

    def test_initial_life(self):
        gs = self._make_gs()
        assert gs.players["p1"].life == 40
        assert gs.players["p2"].life == 40

    def test_hand_contents(self):
        gs = self._make_gs()
        hand = gs.cards_in_zone[("p1", "hand")]
        assert len(hand) == 1
        assert hand[0].name == "Lightning Bolt"
