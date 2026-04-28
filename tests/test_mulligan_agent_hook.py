"""Tests for the strategy-aware mulligan hook on ``MTGAgent``."""

from __future__ import annotations

import asyncio

import pytest

from src.agents.base_agent import MTGAgent
from src.agents.mulligan import select_bottom_cards, should_keep
from src.engine.agent_strategies import Strategy
from src.engine.game_state import Action, ActionType, CardInstance, GameState, Zone


def _card(name: str, type_line: str, cmc: float) -> CardInstance:
    return CardInstance(
        instance_id=f"id_{name}",
        card_data={"name": name, "type_line": type_line, "cmc": cmc},
        zone=Zone.HAND,
        owner_id="p1",
        controller_id="p1",
    )


def _land(name: str = "Mountain") -> CardInstance:
    return _card(name, "Basic Land - Mountain", 0)


def _spell(name: str, cmc: float, kind: str = "Instant") -> CardInstance:
    return _card(name, kind, cmc)


# ---------------------------------------------------------------------------
# Pure-function policy
# ---------------------------------------------------------------------------


def test_should_keep_fallback_classic_keep():
    hand = [_land(), _land(), _land(), _spell("Bolt", 1), _spell("Bolt", 1),
            _spell("Goblin", 2, "Creature - Goblin"), _spell("Big", 5)]
    assert should_keep(hand, strategy=None, mulligans_taken=0,
                       max_mulligans=3)


def test_should_keep_aggressive_prefers_low_curve():
    hand = [_land(), _land(),
            _spell("Bolt", 1), _spell("Bolt", 1),
            _spell("Goblin", 2, "Creature - Goblin"),
            _spell("Big", 6), _spell("Big", 7)]
    assert should_keep(hand, strategy=Strategy.AGGRESSIVE,
                       mulligans_taken=0, max_mulligans=3)


def test_should_keep_aggressive_rejects_no_cheap_plays():
    hand = [_land(), _land(),
            _spell("Big", 5), _spell("Big", 6), _spell("Big", 7),
            _land(), _land()]
    assert not should_keep(hand, strategy=Strategy.AGGRESSIVE,
                           mulligans_taken=0, max_mulligans=3)


def test_should_keep_control_requires_more_lands():
    two_lands = [_land(), _land()] + [_spell("X", 4) for _ in range(5)]
    assert not should_keep(two_lands, strategy=Strategy.CONTROL,
                           mulligans_taken=0, max_mulligans=3)
    four_lands = [_land()] * 4 + [_spell("X", 4) for _ in range(3)]
    assert should_keep(four_lands, strategy=Strategy.CONTROL,
                       mulligans_taken=0, max_mulligans=3)


def test_should_keep_forces_keep_at_max_mulligans():
    bad_hand = [_spell("Big", 7) for _ in range(7)]
    assert should_keep(bad_hand, strategy=Strategy.AGGRESSIVE,
                       mulligans_taken=3, max_mulligans=3)


def test_select_bottom_cards_returns_n_cards():
    hand = [_land(), _land(), _spell("Bolt", 1),
            _spell("Big", 6), _spell("Big", 7)]
    bottom = select_bottom_cards(hand, 2, strategy=Strategy.AGGRESSIVE)
    assert len(bottom) == 2
    # Aggressive bottoms expensive non-lands first
    names = [c.card_data["name"] for c in bottom]
    assert "Big" in names[0] or "Big" in names[1]


def test_select_bottom_cards_handles_empty_and_zero():
    assert select_bottom_cards([], 3) == []
    assert select_bottom_cards([_land()], 0) == []


# ---------------------------------------------------------------------------
# MTGAgent base-class hook
# ---------------------------------------------------------------------------


class _FixedStrategyAgent(MTGAgent):
    def __init__(self, player_id: str, strategy: Strategy):
        super().__init__(player_id)
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    async def decide_action(self, game_state: GameState,
                            legal_actions: list[Action]) -> Action:
        return legal_actions[0] if legal_actions else Action(
            action_type=ActionType.PASS_PRIORITY,
            player_id=self.player_id,
        )


def test_mtgagent_default_decide_mulligan_uses_strategy():
    agent = _FixedStrategyAgent("p1", Strategy.AGGRESSIVE)
    bad_hand = [_spell("Big", 7) for _ in range(7)]
    assert not agent.decide_mulligan(bad_hand, mulligans_taken=0,
                                     max_mulligans=3)
    # Forced keep at the cap.
    assert agent.decide_mulligan(bad_hand, mulligans_taken=3,
                                 max_mulligans=3)


def test_mtgagent_select_bottom_cards_default():
    agent = _FixedStrategyAgent("p1", Strategy.CONTROL)
    hand = [_land(), _land(),
            _spell("Wrath", 4), _spell("Bolt", 1),
            _spell("Counter", 2), _spell("Big", 6),
            _spell("Goblin", 2, "Creature - Goblin")]
    bottom = agent.select_bottom_cards(hand, 2)
    assert len(bottom) == 2


# ---------------------------------------------------------------------------
# End-to-end through GameSimulator
# ---------------------------------------------------------------------------


def test_simulator_uses_agent_mulligan_hook(monkeypatch):
    """When an agent's hook always says KEEP, no mulligans are taken."""
    from src.engine.game_simulator import GameSimulator
    from src.engine.game_execution import AgentGamePlayer

    a1 = AgentGamePlayer("p1", strategy=Strategy.AGGRESSIVE)
    a2 = AgentGamePlayer("p2", strategy=Strategy.CONTROL)
    sim = GameSimulator(a1, a2, max_turns=1)

    keep_calls = {"n": 0}

    class AlwaysKeep:
        @property
        def strategy(self):
            return Strategy.AGGRESSIVE

        def decide_mulligan(self, hand, mulligans_taken, max_mulligans):
            keep_calls["n"] += 1
            return True

        def select_bottom_cards(self, hand, n):
            return hand[:n]

    # Inject a fake "wrapped agent" with the hook the simulator looks for.
    a1.llm_agent = AlwaysKeep()  # type: ignore[attr-defined]

    deck = [{"name": f"Spell{i}", "type_line": "Instant", "cmc": 5}
            for i in range(40)]
    sim.setup_game(deck1=deck, deck2=deck, shuffle=False, max_mulligans=2)

    p1 = sim.game.players[0]
    assert p1.mulligans_taken == 0
    assert keep_calls["n"] >= 1
