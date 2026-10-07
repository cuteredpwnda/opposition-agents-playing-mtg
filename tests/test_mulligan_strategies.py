"""Tests for strategy-aware mulligan policies.

The heuristics in ``src/agents/mulligan.py`` use a ``HandStats`` summary
to decide keep/mulligan based on agent strategy (aggressive, control, combo, reactive).
"""

import pytest
from unittest.mock import MagicMock

from src.agents.base_agent import AgentStrategy
from src.agents.mulligan import HandStats, should_keep, summarise_hand


def _make_mock_card(name: str, is_land: bool = False, cmc: float = 0.0, is_creature: bool = False):
    """Create a mock CardInstance for testing."""
    card = MagicMock()
    card.name = name
    card.is_land = MagicMock(return_value=is_land)
    card.cmc = cmc
    card.is_creature = MagicMock(return_value=is_creature)
    return card


class TestHandStats:
    """Unit tests for the HandStats summary."""

    def test_summarise_hand_with_only_lands(self):
        cards = [_make_mock_card(f"Forest_{i}", is_land=True) for i in range(7)]
        stats = summarise_hand(cards)
        assert stats.total == 7
        assert stats.lands == 7
        assert stats.creatures == 0
        assert stats.cheap_spells == 0
        assert stats.expensive_spells == 0

    def test_summarise_hand_mixed_creatures_and_lands(self):
        hand = [
            _make_mock_card("Forest_1", is_land=True),
            _make_mock_card("Forest_2", is_land=True),
            _make_mock_card("Goblin Guide", is_creature=True, cmc=2.0),
            _make_mock_card("Lightning Bolt", cmc=1.0),
        ]
        stats = summarise_hand(hand)
        assert stats.total == 4
        assert stats.lands == 2
        assert stats.creatures == 1
        assert stats.cheap_spells == 2  # Goblin Guide (2) + Lightning Bolt (1)
        assert stats.expensive_spells == 0

    def test_summarise_hand_empty(self):
        stats = summarise_hand([])
        assert stats.total == 0
        assert stats.lands == 0
        assert stats.creatures == 0


class TestAggressive:
    """Aggressive mulligan: 1-3 lands, 2+ cheap spells."""

    def test_keep_perfect_aggressive_hand(self):
        hand = [
            _make_mock_card("Forest", is_land=True),
            _make_mock_card("Goblin Guide", is_creature=True, cmc=2.0),
            _make_mock_card("Goblin Guide", is_creature=True, cmc=2.0),
            _make_mock_card("Lava Spike", cmc=1.0),
        ]
        # 1 land, 3 cheap spells
        result = should_keep(hand, strategy=AgentStrategy.AGGRESSIVE, mulligans_taken=0, max_mulligans=3)
        assert result is True

    def test_mulligan_too_many_lands_aggressive(self):
        hand = [_make_mock_card(f"Forest_{i}", is_land=True) for i in range(7)]
        # 7 lands, 0 cheap spells
        result = should_keep(hand, strategy=AgentStrategy.AGGRESSIVE, mulligans_taken=0, max_mulligans=3)
        assert result is False

    def test_mulligan_too_few_cheap_spells_aggressive(self):
        hand = [
            _make_mock_card("Forest", is_land=True),
            _make_mock_card("Island", is_land=True),
            _make_mock_card("Cancel", cmc=3.0),  # CMC 3, not cheap
            _make_mock_card("Cancel", cmc=3.0),
        ]
        # 2 lands, 0 cheap spells (both Cancel is 3-drop)
        result = should_keep(hand, strategy=AgentStrategy.AGGRESSIVE, mulligans_taken=0, max_mulligans=3)
        assert result is False

    def test_keep_on_final_mulligan_aggressive(self):
        hand = [_make_mock_card(f"Forest_{i}", is_land=True) for i in range(7)]
        # Even bad hand is kept at max mulligans
        result = should_keep(hand, strategy=AgentStrategy.AGGRESSIVE, mulligans_taken=3, max_mulligans=3)
        assert result is True


class TestControl:
    """Control mulligan: 3-5 lands, 1+ non-land."""

    def test_keep_ideal_control_hand(self):
        hand = [_make_mock_card(f"Forest_{i}", is_land=True) for i in range(5)] + [
            _make_mock_card("Cancel", cmc=3.0),
        ]
        # 5 lands, 1 non-land
        result = should_keep(hand, strategy=AgentStrategy.CONTROL, mulligans_taken=0, max_mulligans=3)
        assert result is True

    def test_mulligan_too_few_lands_control(self):
        hand = [
            _make_mock_card("Forest", is_land=True),
            _make_mock_card("Island", is_land=True),
        ] + [_make_mock_card(f"Goblin Guide_{i}", is_creature=True, cmc=2.0) for i in range(5)]
        # 2 lands, 5 non-lands — too few lands for control
        result = should_keep(hand, strategy=AgentStrategy.CONTROL, mulligans_taken=0, max_mulligans=3)
        assert result is False

    def test_mulligan_no_spells_control(self):
        hand = [_make_mock_card(f"Forest_{i}", is_land=True) for i in range(7)]
        # 7 lands, 0 spells — fails "1+ non-land" requirement
        result = should_keep(hand, strategy=AgentStrategy.CONTROL, mulligans_taken=0, max_mulligans=3)
        assert result is False


class TestCombo:
    """Combo mulligan: 2-5 lands, 4+ non-lands (cards to chain)."""

    def test_keep_ideal_combo_hand(self):
        hand = [
            _make_mock_card("Forest", is_land=True),
            _make_mock_card("Island", is_land=True),
        ] + [_make_mock_card(f"Goblin Guide_{i}", is_creature=True, cmc=2.0) for i in range(5)]
        # 2 lands, 5 non-lands
        result = should_keep(hand, strategy=AgentStrategy.COMBO, mulligans_taken=0, max_mulligans=3)
        assert result is True

    def test_mulligan_too_few_non_lands_combo(self):
        hand = [_make_mock_card(f"Forest_{i}", is_land=True) for i in range(5)] + [
            _make_mock_card("Goblin Guide", is_creature=True, cmc=2.0),
            _make_mock_card("Cancel", cmc=3.0),
        ]
        # 5 lands, 2 non-lands — need 4+
        result = should_keep(hand, strategy=AgentStrategy.COMBO, mulligans_taken=0, max_mulligans=3)
        assert result is False

    def test_mulligan_too_many_lands_combo(self):
        hand = [_make_mock_card(f"Forest_{i}", is_land=True) for i in range(6)] + [
            _make_mock_card("Goblin Guide", is_creature=True, cmc=2.0),
        ]
        # 6 lands, 1 non-land — too many lands
        result = should_keep(hand, strategy=AgentStrategy.COMBO, mulligans_taken=0, max_mulligans=3)
        assert result is False


class TestReactive:
    """Reactive mulligan: 2-5 lands, 1+ cheap interaction (cheap non-land)."""

    def test_keep_ideal_reactive_hand(self):
        hand = [
            _make_mock_card("Forest", is_land=True),
            _make_mock_card("Island", is_land=True),
            _make_mock_card("Lightning Bolt", cmc=1.0),
        ]
        # 2 lands, 1 cheap spell (Bolt is CMC 1)
        result = should_keep(hand, strategy=AgentStrategy.REACTIVE, mulligans_taken=0, max_mulligans=3)
        assert result is True

    def test_mulligan_expensive_interaction_reactive(self):
        hand = [_make_mock_card(f"Forest_{i}", is_land=True) for i in range(5)] + [
            _make_mock_card("Cancel", cmc=3.0),
        ]
        # 5 lands, 1 non-land but Cancel (CMC 3) is not cheap
        result = should_keep(hand, strategy=AgentStrategy.REACTIVE, mulligans_taken=0, max_mulligans=3)
        assert result is False


class TestDefaultStrategy:
    """Default (None) strategy matches aggressive."""

    def test_default_strategy_is_aggressive(self):
        hand = [
            _make_mock_card("Forest", is_land=True),
            _make_mock_card("Goblin Guide", is_creature=True, cmc=2.0),
            _make_mock_card("Goblin Guide", is_creature=True, cmc=2.0),
            _make_mock_card("Lava Spike", cmc=1.0),
        ]
        # Should keep with default strategy (same as aggressive)
        result = should_keep(hand, strategy=None, mulligans_taken=0, max_mulligans=3)
        assert result is True


class TestHeuristicAgentStrategy:
    """Test that HeuristicAgent exposes its strategy correctly."""

    @pytest.mark.asyncio
    async def test_heuristic_agent_aggressive_strategy(self):
        from src.agents.heuristic_agent import HeuristicAgent
        agent = HeuristicAgent(player_id="p1", prefer_aggressive=True)
        assert agent.strategy == AgentStrategy.AGGRESSIVE

    @pytest.mark.asyncio
    async def test_heuristic_agent_control_strategy(self):
        from src.agents.heuristic_agent import HeuristicAgent
        agent = HeuristicAgent(player_id="p1", prefer_aggressive=False)
        assert agent.strategy == AgentStrategy.CONTROL
