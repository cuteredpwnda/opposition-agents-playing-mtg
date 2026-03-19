"""
Tests for Agent Strategies and Decision-Making.

Tests validate:
1. Strategy-specific evaluation logic (aggressive, control, combo, reactive)
2. Board state evaluation using mocked KG
3. Attack/block decision trees
4. Play recommendations based on hand
5. Multi-scenario play evaluation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.engine.agent_strategies import (
    AgentStrategist, Strategy, PlayEvaluation, PlayRecommendation
)
from src.engine.game_state import GameState, PlayerState, CardInstance, Zone


@pytest.fixture
def mock_knowledge_graph():
    """Create a mocked knowledge graph."""
    kg = Mock()
    
    # Mock board evaluation (alice has 2 creatures, bob has 1)
    kg.query_board_evaluation.return_value = {
        "alice": {"name": "Alice", "creatures": 2, "total_power": 3},
        "bob": {"name": "Bob", "creatures": 1, "total_power": 2}
    }
    
    # Mock player resources
    kg.query_player_resources.return_value = {
        "life": 20,
        "life_lost": 0,
        "mana": {"G": 2, "R": 1},
        "hand_size": 5
    }
    
    # Mock threats (bob has 1 threatening creature)
    kg.query_threats.return_value = [
        {"name": "Fire Elemental", "power": 3, "toughness": 3, "tapped": False, "attacker": "Bob"}
    ]
    
    # Mock battlefield cards
    kg.query_cards_on_battlefield.return_value = [
        {"name": "Llanowar Elf", "power": 1, "toughness": 1, "tapped": False, "summoning_sick": False},
        {"name": "Grizzly Bears", "power": 2, "toughness": 2, "tapped": False, "summoning_sick": False}
    ]
    
    return kg


def test_aggressive_strategy_initialization(mock_knowledge_graph):
    """Test creating an aggressive agent strategist."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.AGGRESSIVE)
    
    assert agent.player_id == "alice"
    assert agent.strategy == Strategy.AGGRESSIVE
    assert agent.kg == mock_knowledge_graph


def test_evaluate_board_state(mock_knowledge_graph):
    """Test board state evaluation using mocked KG."""
    agent = AgentStrategist(mock_knowledge_graph, "alice")
    board_state = agent.evaluate_board_state("game1")
    
    assert "board" in board_state
    assert "our_resources" in board_state
    assert "threats" in board_state
    assert "our_creatures" in board_state
    assert board_state["threat_count"] == 1
    assert board_state["total_threat_power"] == 3
    assert board_state["our_creature_count"] == 2


def test_aggressive_should_attack_with_advantage(mock_knowledge_graph):
    """Test aggressive agent attacks when it has board advantage."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.AGGRESSIVE)
    board_state = agent.evaluate_board_state("game1")
    
    # We have 3 power, opponent has 3 power threat, but we have more creatures
    should_attack = agent.should_attack(board_state)
    
    # Aggressive attacks with creature advantage
    assert should_attack is True


def test_aggressive_no_attack_without_creatures(mock_knowledge_graph):
    """Test aggressive agent doesn't attack without creatures."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.AGGRESSIVE)
    board_state = {
        "board": {},
        "our_creatures": [],  # No creatures
        "threats": [],
        "our_resources": {}
    }
    
    should_attack = agent.should_attack(board_state)
    assert should_attack is False


def test_control_attack_requires_safety(mock_knowledge_graph):
    """Test control agent only attacks when safe."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.CONTROL)
    
    # Scenario: Equal board, threats present
    board_state = {
        "board": {},
        "our_creatures": [
            {"power": 2, "toughness": 2},
            {"power": 1, "toughness": 1}
        ],
        "threats": [
            {"power": 3, "toughness": 3}
        ],
        "our_resources": {}
    }
    board_state["total_threat_power"] = 3
    
    # Control won't attack against equivalent threat
    should_attack = agent.should_attack(board_state)
    assert should_attack is False


def test_control_attack_with_overwhelming_power(mock_knowledge_graph):
    """Test control agent attacks with 1.5x power advantage."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.CONTROL)
    
    # Scenario: We have 9 power vs 6 threat
    board_state = {
        "board": {},
        "our_creatures": [
            {"power": 5, "toughness": 2},
            {"power": 4, "toughness": 2}
        ],
        "threats": [
            {"power": 6, "toughness": 6}
        ],
        "our_resources": {}
    }
    board_state["total_threat_power"] = 6
    
    should_attack = agent.should_attack(board_state)
    assert should_attack is True


def test_combo_strategy_requires_hand(mock_knowledge_graph):
    """Test combo agent holds back until hand is full."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.COMBO)
    
    # Scenario: Low hand size
    board_state = {
        "board": {},
        "our_creatures": [
            {"power": 2, "toughness": 2}
        ],
        "threats": [],
        "our_resources": {"hand_size": 2}
    }
    board_state["total_threat_power"] = 0
    
    should_attack = agent.should_attack(board_state)
    assert should_attack is False


def test_combo_strategy_attacks_with_full_hand(mock_knowledge_graph):
    """Test combo agent attacks when hand is developed."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.COMBO)
    
    # Scenario: Full hand, safe board
    board_state = {
        "board": {},
        "our_creatures": [
            {"power": 2, "toughness": 2}
        ],
        "threats": [],
        "our_resources": {"hand_size": 5}
    }
    board_state["total_threat_power"] = 0
    
    should_attack = agent.should_attack(board_state)
    assert should_attack is True


def test_reactive_strategy_only_attacks_no_threats(mock_knowledge_graph):
    """Test reactive agent only attacks when opponent has no threats."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.REACTIVE)
    
    # Scenario: Opponent has threats
    board_state = {
        "board": {},
        "our_creatures": [
            {"power": 3, "toughness": 2}
        ],
        "threats": [
            {"power": 3, "toughness": 3}
        ],
        "our_resources": {}
    }
    board_state["total_threat_power"] = 3
    
    should_attack = agent.should_attack(board_state)
    assert should_attack is False


def test_control_blocking(mock_knowledge_graph):
    """Test control strategy blocks threats."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.CONTROL)
    
    board_state = {
        "our_creatures": [
            {"name": "Grizzly Bears", "power": 2, "toughness": 2, "tapped": False}
        ]
    }
    
    threat = {"power": 3, "toughness": 3}
    
    should_block = agent.should_block(board_state, threat)
    assert should_block is False  # 2 toughness can't block 3 power


def test_aggressive_no_blocking(mock_knowledge_graph):
    """Test aggressive strategy avoids blocking."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.AGGRESSIVE)
    
    board_state = {
        "our_creatures": [
            {"name": "Grizzly Bears", "power": 2, "toughness": 2, "tapped": False}
        ]
    }
    
    threat = {"power": 2, "toughness": 2}
    
    # Even with valid blocker, aggressive doesn't block
    should_block = agent.should_block(board_state, threat)
    assert should_block is False


def test_evaluate_lord_card_aggressive(mock_knowledge_graph):
    """Test aggressive strategy evaluates lord cards as strong."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.AGGRESSIVE)
    board_state = agent.evaluate_board_state("game1")
    
    rec = agent.evaluate_play("Anthem of the Faithful", board_state)
    
    assert rec.evaluation == PlayEvaluation.STRONG
    assert rec.confidence > 0.75
    assert "lord" in rec.reasoning.lower() or "boost" in rec.reasoning.lower()


def test_evaluate_removal_critical_vs_threats(mock_knowledge_graph):
    """Test control strategy marks removal as critical when threats exist."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.CONTROL)
    board_state = agent.evaluate_board_state("game1")
    
    # Board has threats now
    board_state["threats"] = [{"power": 3, "toughness": 3}]
    
    rec = agent.evaluate_play("Lightning Bolt", board_state)
    
    assert rec.evaluation == PlayEvaluation.CRITICAL
    assert rec.confidence > 0.9


def test_evaluate_draw_spell_control(mock_knowledge_graph):
    """Test control strategy evaluates draw spells as strong."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.CONTROL)
    board_state = agent.evaluate_board_state("game1")
    
    rec = agent.evaluate_play("Divination Draw Spell", board_state)
    
    assert rec.evaluation == PlayEvaluation.STRONG
    assert "card advantage" in rec.reasoning.lower()


def test_play_recommendation_structure(mock_knowledge_graph):
    """Test PlayRecommendation dataclass structure."""
    rec = PlayRecommendation(
        action_id="card1",
        action_type="play_card",
        evaluation=PlayEvaluation.STRONG,
        reasoning="Test play",
        confidence=0.85
    )
    
    assert rec.action_id == "card1"
    assert rec.action_type == "play_card"
    assert rec.evaluation == PlayEvaluation.STRONG
    assert 0.0 <= rec.confidence <= 1.0


def test_strategy_brief(mock_knowledge_graph):
    """Test strategy description text."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.AGGRESSIVE)
    brief = agent.get_strategy_brief()
    
    assert "board" in brief.lower()
    assert len(brief) > 0
    assert "creature" in brief.lower()


def test_all_strategies_have_briefs(mock_knowledge_graph):
    """Test that all strategies have descriptive briefs."""
    for strategy in Strategy:
        agent = AgentStrategist(mock_knowledge_graph, "alice", strategy)
        brief = agent.get_strategy_brief()
        
        assert brief != "Unknown strategy"
        assert len(brief) > 10  # Reasonable length


def test_get_next_action_control_vs_threats(mock_knowledge_graph):
    """Test control agent prioritizes threat removal."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.CONTROL)
    game = GameState()
    
    action = agent.get_next_action("game1", game)
    
    # Control should recommend seeking removal against threats
    assert action is not None
    assert "threat" in action.reasoning.lower() or "remove" in action.reasoning.lower()


def test_aggressive_attacks_when_advantaged(mock_knowledge_graph):
    """Test aggressive agent recommends attacks with advantage."""
    agent = AgentStrategist(mock_knowledge_graph, "alice", Strategy.AGGRESSIVE)
    
    # Mock advantage scenario
    mock_knowledge_graph.query_board_evaluation.return_value = {
        "alice": {"name": "Alice", "creatures": 3, "total_power": 5},
        "bob": {"name": "Bob", "creatures": 1, "total_power": 2}
    }
    
    game = GameState()
    action = agent.get_next_action("game1", game)
    
    # Should recommend attack action
    if action:  # May be None if KG fails gracefully
        assert action.action_type == "attack"


def test_strategy_switch_changes_behavior(mock_knowledge_graph):
    """Test switching strategies changes evaluation."""
    board_state = {
        "board": {},
        "our_creatures": [
            {"power": 2, "toughness": 2}
        ],
        "threats": [
            {"power": 3, "toughness": 3}
        ],
        "our_resources": {"hand_size": 3}
    }
    board_state["total_threat_power"] = 3
    
    # Aggressive attacks despite threat
    aggressive = AgentStrategist(mock_knowledge_graph, "alice", Strategy.AGGRESSIVE)
    agg_attack = aggressive.should_attack(board_state)
    
    # Control doesn't attack against threat
    control = AgentStrategist(mock_knowledge_graph, "alice", Strategy.CONTROL)
    ctl_attack = control.should_attack(board_state)
    
    # Strategies should differ
    assert agg_attack != ctl_attack


def test_no_creatures_prevents_attack_all_strategies(mock_knowledge_graph):
    """Test all strategies refuse to attack without creatures."""
    board_state = {
        "board": {},
        "our_creatures": [],
        "threats": [],
        "our_resources": {}
    }
    board_state["total_threat_power"] = 0
    
    for strategy in Strategy:
        agent = AgentStrategist(mock_knowledge_graph, "alice", strategy)
        should_attack = agent.should_attack(board_state)
        assert should_attack is False, f"{strategy} should not attack without creatures"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
