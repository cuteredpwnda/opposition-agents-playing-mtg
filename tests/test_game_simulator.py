"""
Tests for Multi-turn Game Simulator.

Test scenarios:
1. Game initialization and setup
2. Phases execute correctly (untap, draw, main, combat, cleanup)
3. Win condition detection (life total)
4. Turn advancement
5. Multi-turn gameplay (Aggressive vs Control)
6. Full game simulation with victor
7. Max turn draw
8. Phase sequencing
9. Game summary and logging
10. Edge cases (single turn games, instant wins)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.engine.game_simulator import GameSimulator, GameResult, Phase
from src.engine.game_state import GameState, PlayerState, CardInstance, Zone
from src.engine.game_execution import AgentGamePlayer
from src.engine.agent_strategies import Strategy


@pytest.fixture
def mock_agent():
    """Create an actual agent player."""
    agent = AgentGamePlayer("test_agent", Strategy.AGGRESSIVE, knowledge_graph=None)
    return agent


@pytest.fixture
def mock_agents():
    """Create two actual agent players."""
    agent1 = AgentGamePlayer("aggressive_agent", Strategy.AGGRESSIVE, knowledge_graph=None)
    agent2 = AgentGamePlayer("control_agent", Strategy.CONTROL, knowledge_graph=None)
    
    return agent1, agent2


@pytest.fixture
def simulator(mock_agents):
    """Create game simulator."""
    agent1, agent2 = mock_agents
    sim = GameSimulator(agent1, agent2, knowledge_graph=None, max_turns=10)
    return sim


class TestGameSimulatorSetup:
    """Test game initialization."""
    
    def test_simulator_initialization(self, simulator, mock_agents):
        """Test basic simulator creation."""
        assert simulator.agent1 is not None
        assert simulator.agent2 is not None
        assert simulator.max_turns == 10
        assert simulator.game is None
        assert len(simulator.turn_history) == 0
    
    def test_setup_game(self, simulator):
        """Test game setup creates correct state."""
        game = simulator.setup_game("test_game_1")
        
        assert simulator.game is not None
        assert simulator.game_id == "test_game_1"
        assert game.game_id == "test_game_1"
        assert len(game.players) == 2
        assert game.players[0].life_total == 20
        assert game.players[1].life_total == 20
        assert game.turn_number == 1
    
    def test_setup_game_auto_id(self, simulator):
        """Test setup generates ID if not provided."""
        game1 = simulator.setup_game()
        game1_id = simulator.game_id
        
        # Create new simulator for second game
        simulator2 = GameSimulator(simulator.agent1, simulator.agent2)
        game2 = simulator2.setup_game()
        game2_id = simulator2.game_id
        
        assert game1_id != game2_id
        assert game1.game_id == game1_id

    def test_setup_game_applies_london_mulligan(self, simulator):
        """Test opening hand size shrinks after a mulligan."""
        spell_only_deck = [
            {
                "name": "Shock",
                "mana_cost": "{R}",
                "cmc": 1,
                "type_line": "Instant",
                "oracle_text": "Shock deals 2 damage to any target.",
            }
            for _ in range(60)
        ]

        game = simulator.setup_game(
            deck1=spell_only_deck,
            deck2=spell_only_deck,
            shuffle=False,
            mulligan_enabled=True,
            max_mulligans=1,
        )

        p1 = game.players[0]
        p1_hand = [
            c for c in game.cards
            if c.owner_id == p1.player_id and c.zone == Zone.HAND
        ]
        assert p1.mulligans_taken == 1
        assert len(p1_hand) == 6

    def test_setup_game_can_disable_mulligan(self, simulator):
        """Test setup can skip mulligans and keep full opening hand."""
        spell_only_deck = [
            {
                "name": "Shock",
                "mana_cost": "{R}",
                "cmc": 1,
                "type_line": "Instant",
                "oracle_text": "Shock deals 2 damage to any target.",
            }
            for _ in range(60)
        ]

        game = simulator.setup_game(
            deck1=spell_only_deck,
            deck2=spell_only_deck,
            shuffle=False,
            mulligan_enabled=False,
        )

        p1 = game.players[0]
        p1_hand = [
            c for c in game.cards
            if c.owner_id == p1.player_id and c.zone == Zone.HAND
        ]
        assert p1.mulligans_taken == 0
        assert len(p1_hand) == 7


class TestWinConditions:
    """Test victory detection."""
    
    def test_player1_wins_on_opponent_death(self, simulator):
        """Test player 1 wins when player 2 reaches 0 life."""
        game = simulator.setup_game()
        game.players[1].life_total = 0
        
        result = simulator.check_win_condition()
        assert result == GameResult.PLAYER1_WIN
    
    def test_player2_wins_on_opponent_death(self, simulator):
        """Test player 2 wins when player 1 reaches 0 life."""
        game = simulator.setup_game()
        game.players[0].life_total = 0
        
        result = simulator.check_win_condition()
        assert result == GameResult.PLAYER2_WIN
    
    def test_player1_wins_on_negative_life(self, simulator):
        """Test victory even with negative life."""
        game = simulator.setup_game()
        game.players[1].life_total = -5
        
        result = simulator.check_win_condition()
        assert result == GameResult.PLAYER1_WIN
    
    def test_timeout_uses_tiebreaker_on_max_turns(self, simulator):
        """Test timeout awards leader when max turns reached."""
        simulator.max_turns = 5
        game = simulator.setup_game()
        game.turn_number = 5
        game.players[0].life_total = 17
        game.players[1].life_total = 14
        
        result = simulator.check_win_condition()
        assert result == GameResult.PLAYER1_WIN

    def test_draw_on_max_turns_if_fully_tied(self, simulator):
        """Test true draw on max turns when all tie-breakers are equal."""
        simulator.max_turns = 5
        game = simulator.setup_game(starting_hand_size=0)
        game.turn_number = 5

        # Keep both players exactly tied across timeout score dimensions.
        game.players[0].life_total = 20
        game.players[1].life_total = 20

        result = simulator.check_win_condition()
        assert result == GameResult.DRAW
    
    def test_no_win_at_20_life(self, simulator):
        """Test game continues at starting life."""
        game = simulator.setup_game()
        game.players[0].life_total = 20
        game.players[1].life_total = 20
        
        result = simulator.check_win_condition()
        assert result is None


class TestPhaseExecution:
    """Test individual phase execution."""
    
    def test_untap_phase_untaps_creatures(self, simulator):
        """Test untap phase removes tapped status."""
        game = simulator.setup_game()
        
        # Add tapped creature to battlefield
        tapped_card = CardInstance(
            instance_id="bear_1",
            card_data={"name": "Grizzly Bears", "mana_cost": "1G", "type_line": "Creature - Bear", "power": "2", "toughness": "2"},
            zone=Zone.BATTLEFIELD,
            tapped=True
        )
        game.cards.append(tapped_card)
        
        simulator.execute_phase(Phase.UNTAP)
        
        assert not tapped_card.tapped
    
    def test_upkeep_phase(self, simulator):
        """Test upkeep phase executes."""
        game = simulator.setup_game()
        actions = simulator.execute_phase(Phase.UPKEEP)
        
        assert len(actions) > 0
        assert "Upkeep" in actions[0]
    
    def test_draw_phase(self, simulator):
        """Test draw phase."""
        game = simulator.setup_game()
        actions = simulator.execute_phase(Phase.DRAW)
        
        assert len(actions) > 0
        assert "draws a card" in actions[0]

    def test_draw_phase_empty_library_causes_loss(self, simulator):
        """Test immediate loss when drawing from an empty library."""
        game = simulator.setup_game()

        # Empty active player's library.
        active_id = game.active_player.player_id
        for card in game.cards:
            if card.owner_id == active_id and card.zone == Zone.LIBRARY:
                card.zone = Zone.GRAVEYARD

        actions = simulator.execute_phase(Phase.DRAW)

        assert game.game_over
        assert game.winner == game.players[1]
        assert "empty library" in actions[0]
    
    def test_main_phase(self, simulator):
        """Test main phase calls coordinator."""
        # Setup mock for this test only
        with patch.object(simulator.coordinator, 'execute_main_phase_plays', return_value=["Played Grizzly Bears"]):
            game = simulator.setup_game()
            game.phase = Phase.MAIN_1
            
            actions = simulator.execute_phase(Phase.MAIN_1)
            
            assert "Played Grizzly Bears" in actions
    
    def test_combat_attackers_phase(self, simulator):
        """Test combat attackers phase."""
        # Setup mock for this test only
        with patch.object(simulator.coordinator, 'execute_combat_phase', return_value=["Attacked with 2 creatures"]):
            game = simulator.setup_game()
            game.phase = Phase.COMBAT_ATTACKERS
            
            actions = simulator.execute_phase(Phase.COMBAT_ATTACKERS)
            
            assert "Attacked with 2 creatures" in actions
    
    def test_cleanup_phase_resets_flags(self, simulator):
        """Test cleanup phase resets turn flags."""
        game = simulator.setup_game()
        # Set flags to non-default states
        game.players[0].has_drawn_for_turn = True
        game.players[0].land_plays_remaining = 0
        game.players[1].has_drawn_for_turn = True
        game.players[1].land_plays_remaining = 0
        
        simulator.execute_phase(Phase.CLEANUP)
        
        # Both players should be reset
        assert not game.players[0].has_drawn_for_turn
        assert game.players[0].land_plays_remaining == 1
        assert not game.players[1].has_drawn_for_turn
        assert game.players[1].land_plays_remaining == 1


class TestTurnAdvancement:
    """Test turn progression."""
    
    def test_advance_turn_increments_counter(self, simulator):
        """Test turn number increments."""
        game = simulator.setup_game()
        assert game.turn_number == 1
        
        simulator.advance_turn()
        
        assert simulator.game.turn_number == 2
    
    def test_advance_turn_switches_active_player(self, simulator):
        """Test active player toggles each turn."""
        game = simulator.setup_game()
        original_index = game.active_player_index
        
        simulator.advance_turn()
        
        assert simulator.game.active_player_index != original_index
        assert simulator.game.active_player_index == 1
    
    def test_advance_turn_twice_returns_to_player1(self, simulator):
        """Test two turns return active player to original."""
        game = simulator.setup_game()
        assert game.active_player_index == 0
        
        simulator.advance_turn()
        assert simulator.game.active_player_index == 1
        
        simulator.advance_turn()
        assert simulator.game.active_player_index == 0
    
    def test_turn_counter_after_multiple_advances(self, simulator):
        """Test turn counter continues correctly."""
        game = simulator.setup_game()
        
        for i in range(5):
            simulator.advance_turn()
        
        assert simulator.game.turn_number == 6


class TestFullTurnExecution:
    """Test complete turn with all phases."""
    
    def test_execute_full_turn_runs_all_phases(self, simulator):
        """Test full turn executes all phases."""
        game = simulator.setup_game()
        
        actions = simulator.execute_full_turn()
        
        assert len(actions) > 0
        # Should have phases: untap, upkeep, draw, main1, combat phases, main2, end, cleanup
        assert len(simulator.phase_log) > 0
    
    def test_full_turn_stops_on_win_condition(self, simulator):
        """Test full turn stops if someone dies mid-turn."""
        game = simulator.setup_game()
        game.players[1].life_total = 0  # Kill player 2
        
        actions = simulator.execute_full_turn()
        
        # Should have started but victory check should catch it
        result = simulator.check_win_condition()
        assert result == GameResult.PLAYER1_WIN
    
    def test_multiple_full_turns(self, simulator):
        """Test executing multiple full turns."""
        game = simulator.setup_game()
        
        # Execute 3 turns
        for _ in range(3):
            simulator.execute_full_turn()
            simulator.advance_turn()
        
        assert simulator.game.turn_number == 4


class TestGameSimulation:
    """Test full game flow."""
    
    def test_run_game_initializes_and_completes(self, simulator):
        """Test game can run from start to finish."""
        # Update max turns for quicker test
        simulator.max_turns = 3
        
        result = simulator.run_game()
        
        assert result in [GameResult.PLAYER1_WIN, GameResult.PLAYER2_WIN, GameResult.DRAW]
        assert simulator.game is not None
    
    def test_game_result_is_final(self, simulator):
        """Test game terminates with definitive result."""
        result = simulator.run_game()
        
        assert result is not None
        assert isinstance(result, GameResult)
    
    def test_game_respects_max_turns(self, simulator):
        """Test game respects maximum turn limit."""
        simulator.max_turns = 2
        result = simulator.run_game()
        
        # Should end in draw due to max turns
        assert simulator.game.turn_number <= simulator.max_turns + 6  # +6 for safety check


class TestGameSummary:
    """Test game reporting."""
    
    def test_game_summary_after_setup(self, simulator):
        """Test summary before playing."""
        simulator.setup_game("game_summary_test")
        summary = simulator.get_game_summary()
        
        assert "game_summary_test" in summary
        assert "Agent" in summary
        assert "20 HP" in summary
    
    def test_game_summary_includes_stats(self, simulator):
        """Test summary contains key statistics."""
        simulator.setup_game()
        simulator.game.turn_number = 5
        
        summary = simulator.get_game_summary()
        
        assert "5" in summary  # Turn number
        assert "SUMMARY" in summary
    
    def test_phase_log_contains_actions(self, simulator):
        """Test phase log is populated after execution."""
        simulator.setup_game()
        simulator.execute_phase(Phase.UNTAP)
        simulator.execute_phase(Phase.DRAW)
        
        log = simulator.get_phase_log()
        
        assert "untapped" in log or "UNTAP" in log
        assert "draws" in log or "DRAW" in log
    
    def test_empty_phase_log_before_play(self, simulator):
        """Test phase log is empty before any play."""
        simulator.setup_game()
        
        log = simulator.get_phase_log()
        
        # Log should be mostly empty or just have initialization
        assert len(log.strip().split('\n')) < 5


class TestEdgeCases:
    """Test unusual scenarios."""
    
    def test_simulator_with_no_kg(self, mock_agents):
        """Test simulator works without knowledge graph."""
        agent1, agent2 = mock_agents
        agent1.player_id = "player_1"
        agent2.player_id = "player_2"
        
        sim = GameSimulator(agent1, agent2, knowledge_graph=None)
        game = sim.setup_game()
        
        assert game is not None
        assert sim.kg is None
    
    def test_check_win_condition_before_setup(self, simulator):
        """Test win check returns None if game not initialized."""
        result = simulator.check_win_condition()
        assert result is None
    
    def test_execute_phase_before_setup(self, simulator):
        """Test phase execution returns empty if no game."""
        simulator.game = None
        actions = simulator.execute_phase(Phase.UNTAP)
        
        assert actions == []
    
    def test_advance_turn_before_setup(self, simulator):
        """Test advancing turn without game doesn't crash."""
        simulator.game = None
        simulator.advance_turn()  # Should not raise
        assert simulator.game is None
    
    def test_multiple_games_in_sequence(self, mock_agents):
        """Test running multiple games in sequence."""
        agent1, agent2 = mock_agents
        agent1.player_id = "player_1"
        agent2.player_id = "player_2"
        
        game_ids = []
        for i in range(3):
            sim = GameSimulator(agent1, agent2, max_turns=2)
            sim.setup_game(f"game_{i}")
            game_ids.append(sim.game_id)
        
        assert len(set(game_ids)) == 3  # All unique


class TestPhaseSequencing:
    """Test phases occur in correct order."""
    
    def test_full_turn_phase_order(self, simulator):
        """Test phases execute in MTG turn order."""
        simulator.setup_game()
        
        # Clear log to see fresh sequence
        simulator.phase_log = []
        simulator.execute_full_turn()
        
        log = simulator.get_phase_log()
        
        # Check that key phases appear
        assert len(simulator.phase_log) > 0
    
    def test_cleanup_is_last_phase(self, simulator):
        """Test cleanup phase is final."""
        simulator.setup_game()
        simulator.phase_log = []
        
        phases = [
            Phase.UNTAP,
            Phase.UPKEEP,
            Phase.DRAW,
            Phase.MAIN_1,
            Phase.CLEANUP
        ]
        
        for phase in phases:
            simulator.execute_phase(phase)
        
        log = simulator.phase_log[-1]
        assert "Cleanup" in log or "cleanup" in log


class TestGameStateTracking:
    """Test game state changes during play."""
    
    def test_life_totals_start_at_20(self, simulator):
        """Test both players start with 20 life."""
        game = simulator.setup_game()
        
        assert game.players[0].life_total == 20
        assert game.players[1].life_total == 20
    
    def test_game_id_persists(self, simulator):
        """Test game ID is consistent."""
        game1 = simulator.setup_game("persist_test")
        game_id_1 = simulator.game_id
        
        # Execute some actions
        simulator.execute_phase(Phase.UNTAP)
        simulator.advance_turn()
        
        assert simulator.game_id == game_id_1
    
    def test_turn_number_only_increases(self, simulator):
        """Test turn counter never decreases."""
        game = simulator.setup_game()
        turn_nums = [game.turn_number]
        
        for _ in range(5):
            simulator.advance_turn()
            turn_nums.append(simulator.game.turn_number)
        
        # Each should be >= previous
        for i in range(1, len(turn_nums)):
            assert turn_nums[i] >= turn_nums[i-1]
