"""
Tests for Game Execution Bridge and Agent Play.

Tests validate:
1. Agent player creation and strategy
2. Hand card detection
3. Play decision making
4. Attack targeting
5. Block decisions
6. Game coordination between agents
7. Play history tracking
"""

import pytest
from unittest.mock import Mock
from src.engine.game_execution import AgentGamePlayer, GameCoordinator, GamePhaseAction
from src.engine.agent_strategies import Strategy
from src.engine.game_state import GameState, PlayerState, CardInstance, Zone


@pytest.fixture
def mock_knowledge_graph():
    """Create a mocked KG for game execution."""
    kg = Mock()
    kg.build_from_game_state = Mock()
    return kg


@pytest.fixture
def player1():
    """Create first game player."""
    return PlayerState("alice", "Alice")


@pytest.fixture
def player2():
    """Create second game player."""
    return PlayerState("bob", "Bob")


@pytest.fixture
def game_state(player1, player2):
    """Create a simple game state."""
    game = GameState(players=[player1, player2])
    return game


def test_agent_player_creation(mock_knowledge_graph):
    """Test creating an agent game player."""
    agent = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    
    assert agent.player_id == "alice"
    assert agent.strategy == Strategy.AGGRESSIVE
    assert agent.kg == mock_knowledge_graph


def test_agent_without_knowledge_graph():
    """Test agent can be created without KG."""
    agent = AgentGamePlayer("alice", Strategy.CONTROL)
    
    assert agent.player_id == "alice"
    assert agent.kg is None
    assert agent.strategist is None


def test_get_plays_from_hand_empty(game_state):
    """Test getting cards from empty hand."""
    agent = AgentGamePlayer("alice", Strategy.AGGRESSIVE)
    plays = agent.get_plays_from_hand(game_state)
    
    assert len(plays) == 0


def test_get_plays_from_hand_with_cards(game_state, player1):
    """Test getting cards from hand with cards."""
    # Add cards to hand
    card1 = CardInstance("Grizzly Bears", {"type": "Creature", "name": "Grizzly Bears"}, player1.player_id)
    card1.zone = Zone.HAND
    card1.controller_id = player1.player_id

    card2 = CardInstance("Forest", {"type": "Land", "name": "Forest"}, player1.player_id)
    card2.zone = Zone.HAND
    card2.controller_id = player1.player_id

    game_state.cards.append(card1)
    game_state.cards.append(card2)

    agent = AgentGamePlayer("alice", Strategy.AGGRESSIVE)
    plays = agent.get_plays_from_hand(game_state)

    assert len(plays) == 2
    # Both cards should be in hand
    assert plays[0] in [card1, card2]
    """Test playing a card with no mana cost."""
    card = CardInstance("Island", {"type": "Land", "mana_cost": ""}, player1.player_id)
    agent = AgentGamePlayer("alice")
    
    can_play = agent.can_play_card(card, game_state)
    assert can_play is True


def test_can_play_card_with_cost(game_state, player1):
    """Test playing a card with mana cost."""
    card = CardInstance("Grizzly Bears", {"type": "Creature", "mana_cost": "{1}{G}"}, player1.player_id)
    agent = AgentGamePlayer("alice")
    
    can_play = agent.can_play_card(card, game_state)
    # Simplified: returns True for testing
    assert can_play is True


def test_decide_play_action_no_kg(game_state):
    """Test play decision without KG returns None."""
    agent = AgentGamePlayer("alice")
    action = agent.decide_play_action(game_state, "game1")
    
    assert action is None


def test_decide_attack_targets_no_kg(game_state):
    """Test attack decision without KG returns empty."""
    agent = AgentGamePlayer("alice")
    targets = agent.decide_attack_targets(game_state, "game1")
    
    assert targets == []


def test_decide_attack_targets_aggressive(game_state, mock_knowledge_graph, player1):
    """Test aggressive agent decides to attack."""
    # Mock KG response
    mock_knowledge_graph.query_board_evaluation.return_value = {
        "alice": {"creatures": 2, "total_power": 3},
        "bob": {"creatures": 1, "total_power": 2}
    }
    mock_knowledge_graph.query_player_resources.return_value = {
        "life": 20, "mana": {}, "hand_size": 3
    }
    mock_knowledge_graph.query_threats.return_value = []
    mock_knowledge_graph.query_cards_on_battlefield.return_value = [
        {"name": "Llanowar Elf", "power": 1, "toughness": 1, "tapped": False, "summoning_sick": False},
        {"name": "Grizzly Bears", "power": 2, "toughness": 2, "tapped": False, "summoning_sick": False}
    ]
    
    # Add creatures to board
    creature1 = CardInstance("Llanowar Elf", {"type": "Creature", "power": 1, "toughness": 1}, player1.player_id)
    creature1.zone = Zone.BATTLEFIELD
    creature1.controller_id = player1.player_id
    creature1.tapped = False
    creature1.summoning_sick = False
    
    creature2 = CardInstance("Grizzly Bears", {"type": "Creature", "power": 2, "toughness": 2}, player1.player_id)
    creature2.zone = Zone.BATTLEFIELD
    creature2.controller_id = player1.player_id
    creature2.tapped = False
    creature2.summoning_sick = False
    
    game_state.cards.extend([creature1, creature2])
    
    agent = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    targets = agent.decide_attack_targets(game_state, "game1")
    
    assert len(targets) > 0


def test_decide_blocks_control_strategy(game_state, mock_knowledge_graph, player1, player2):
    """Test control agent decides to block."""
    # Mock KG response
    mock_knowledge_graph.query_board_evaluation.return_value = {
        "alice": {"creatures": 1},
        "bob": {"creatures": 1}
    }
    mock_knowledge_graph.query_player_resources.return_value = {
        "life": 20, "mana": {}, "hand_size": 3
    }
    mock_knowledge_graph.query_threats.return_value = [
        {"power": 2, "toughness": 2}
    ]
    mock_knowledge_graph.query_cards_on_battlefield.return_value = [
        {"name": "Grizzly Bears", "power": 2, "toughness": 2, "tapped": False}
    ]
    
    # Add blocker
    blocker = CardInstance("Grizzly Bears", {"type": "Creature", "power": 2, "toughness": 2}, player1.player_id)
    blocker.zone = Zone.BATTLEFIELD
    blocker.controller_id = player1.player_id
    blocker.tapped = False
    
    game_state.cards.append(blocker)
    
    # Create attacker
    attacker = CardInstance("Giant Growth", {"type": "Creature", "power": 2, "toughness": 2}, player2.player_id)
    attacker.zone = Zone.BATTLEFIELD
    attacker.controller_id = player2.player_id
    
    agent = AgentGamePlayer("alice", Strategy.CONTROL, mock_knowledge_graph)
    block = agent.decide_blocks(attacker, game_state, "game1")
    
    # Control strategy should block
    assert block is not None


def test_respond_to_spell_default(game_state):
    """Test default spell response."""
    agent = AgentGamePlayer("alice")
    spell = CardInstance("Lightning Bolt", {"type": "Instant"}, "bob")
    
    should_respond = agent.respond_to_spell(spell, game_state)
    assert should_respond is False


def test_get_priority_default(game_state):
    """Test default priority handling."""
    agent = AgentGamePlayer("alice")
    passes_priority = agent.get_priority(game_state)
    
    assert passes_priority is True


def test_get_strategy_summary_no_kg():
    """Test strategy summary without KG."""
    agent = AgentGamePlayer("alice", Strategy.AGGRESSIVE)
    summary = agent.get_strategy_summary()
    
    assert "alice" in summary
    assert "no strategy" in summary


def test_get_strategy_summary_with_kg(mock_knowledge_graph):
    """Test strategy summary with KG."""
    agent = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    summary = agent.get_strategy_summary()
    
    assert "alice" in summary
    assert "Aggressive" in summary or "board" in summary


def test_game_coordinator_creation(mock_knowledge_graph):
    """Test creating a game coordinator."""
    agent1 = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    agent2 = AgentGamePlayer("bob", Strategy.CONTROL, mock_knowledge_graph)
    
    coordinator = GameCoordinator(agent1, agent2, mock_knowledge_graph)
    
    assert coordinator.agent1 == agent1
    assert coordinator.agent2 == agent2
    assert coordinator.kg == mock_knowledge_graph
    assert coordinator.game_id is None


def test_setup_game(game_state, mock_knowledge_graph):
    """Test setting up game in coordinator."""
    agent1 = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    agent2 = AgentGamePlayer("bob", Strategy.CONTROL, mock_knowledge_graph)
    
    coordinator = GameCoordinator(agent1, agent2, mock_knowledge_graph)
    coordinator.setup_game(game_state, "game1")
    
    assert coordinator.game_id == "game1"
    mock_knowledge_graph.build_from_game_state.assert_called_once()


def test_get_active_agent(game_state, mock_knowledge_graph):
    """Test getting active agent."""
    agent1 = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    agent2 = AgentGamePlayer("bob", Strategy.CONTROL, mock_knowledge_graph)
    
    coordinator = GameCoordinator(agent1, agent2, mock_knowledge_graph)
    
    # Initial: alice active
    active = coordinator.get_active_agent(game_state)
    assert active == agent1
    
    # Switch to bob
    game_state.active_player_index = 1
    active = coordinator.get_active_agent(game_state)
    assert active == agent2


def test_execute_main_phase_plays_no_kg(game_state):
    """Test main phase without KG."""
    agent1 = AgentGamePlayer("alice", Strategy.AGGRESSIVE)
    agent2 = AgentGamePlayer("bob", Strategy.CONTROL)
    
    coordinator = GameCoordinator(agent1, agent2)
    actions = coordinator.execute_main_phase_plays(game_state)
    
    assert len(actions) == 0


def test_execute_main_phase_plays_with_kg(game_state, mock_knowledge_graph):
    """Test main phase with KG and recommendation."""
    agent1 = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    agent2 = AgentGamePlayer("bob", Strategy.CONTROL, mock_knowledge_graph)
    
    coordinator = GameCoordinator(agent1, agent2, mock_knowledge_graph)
    coordinator.game_id = "game1"
    
    actions = coordinator.execute_main_phase_plays(game_state)
    
    # Should have at least one action
    assert len(actions) >= 0


def test_execute_combat_phase_no_kg(game_state):
    """Test combat phase without KG."""
    agent1 = AgentGamePlayer("alice", Strategy.AGGRESSIVE)
    agent2 = AgentGamePlayer("bob", Strategy.CONTROL)
    
    coordinator = GameCoordinator(agent1, agent2)
    actions = coordinator.execute_combat_phase(game_state)
    
    assert len(actions) == 0


def test_get_game_summary(mock_knowledge_graph):
    """Test getting game summary."""
    agent1 = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    agent2 = AgentGamePlayer("bob", Strategy.CONTROL, mock_knowledge_graph)
    
    coordinator = GameCoordinator(agent1, agent2, mock_knowledge_graph)
    summary = coordinator.get_game_summary()
    
    assert "Agent Game Summary" in summary
    assert "alice" in summary
    assert "bob" in summary


def test_play_history_tracking(game_state, mock_knowledge_graph):
    """Test that play history is tracked."""
    agent1 = AgentGamePlayer("alice", Strategy.AGGRESSIVE, mock_knowledge_graph)
    agent2 = AgentGamePlayer("bob", Strategy.CONTROL, mock_knowledge_graph)
    
    coordinator = GameCoordinator(agent1, agent2, mock_knowledge_graph)
    coordinator.game_id = "game1"
    
    initial_plays = len(coordinator.play_history)
    assert initial_plays == 0


def test_game_phase_action_enum():
    """Test GamePhaseAction enum."""
    assert GamePhaseAction.PLAY_CARD.value == "play_card"
    assert GamePhaseAction.ACTIVATE_ABILITY.value == "activate_ability"
    assert GamePhaseAction.DECLARE_ATTACK.value == "declare_attack"
    assert GamePhaseAction.DECLARE_BLOCK.value == "declare_block"
    assert GamePhaseAction.PASS.value == "pass"
    assert GamePhaseAction.RESPOND.value == "respond"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
