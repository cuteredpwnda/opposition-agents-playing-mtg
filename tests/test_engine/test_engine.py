"""Tests for the game engine."""

import pytest

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
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
        advance_phase(gs)
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
        assert cost.get("generic", 0) == 0 and not any(v > 0 for v in cost.values())

    def test_can_pay_sufficient(self):
        player = PlayerState(mana_pool={"W": 1, "U": 3, "B": 0, "R": 0, "G": 0, "C": 0})
        cost = {"generic": 1, "U": 2}
        assert can_pay(player, cost)

    def test_can_pay_insufficient(self):
        player = PlayerState(mana_pool={"W": 0, "U": 1, "B": 0, "R": 0, "G": 0, "C": 0})
        cost = {"generic": 1, "U": 2}
        assert not can_pay(player, cost)

    def test_pay_cost_deducts(self):
        player = PlayerState(mana_pool={"W": 1, "U": 3, "B": 0, "R": 0, "G": 0, "C": 0})
        cost = {"generic": 1, "U": 2}
        pay_cost(player, cost)
        # pay colored U:2 → U=1; generic:1 consumed from U (cheapest available per CUBWRG order) → U=0
        assert player.mana_pool["U"] == 0
        assert player.mana_pool["W"] == 1  # W was not consumed


# ---------------------------------------------------------------------------
# Game state tests
# ---------------------------------------------------------------------------

class TestGameState:
    def _make_gs(self) -> GameState:
        return GameState(
            players=[
                PlayerState(player_id="p1", life_total=40),
                PlayerState(player_id="p2", life_total=40),
            ],
            active_player="p1",
            priority_player="p1",
            phase=Phase.MAIN_1,
            turn_number=1,
            cards=[
                CardInstance(
                    instance_id="c1",
                    card_data={
                        "name": "Lightning Bolt",
                        "oracle_text": "Deal 3 damage.",
                        "type_line": "Instant",
                        "mana_cost": "{R}",
                        "cmc": 1,
                    },
                    zone=Zone.HAND,
                    owner_id="p1",
                    controller_id="p1",
                )
            ],
        )

    def test_initial_life(self):
        gs = self._make_gs()
        p1 = next(p for p in gs.players if p.player_id == "p1")
        p2 = next(p for p in gs.players if p.player_id == "p2")
        assert p1.life_total == 40
        assert p2.life_total == 40

    def test_hand_contents(self):
        gs = self._make_gs()
        hand = gs.cards_in_zone("p1", Zone.HAND)
        assert len(hand) == 1
        assert hand[0].name == "Lightning Bolt"
