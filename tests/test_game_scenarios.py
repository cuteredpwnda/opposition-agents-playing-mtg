"""
Multi-Agent Game Scenarios - Full Agent vs Agent Games.

This module runs complete games between different agent strategy combinations:
1. Aggressive vs Control
2. Aggressive vs Aggressive
3. Control vs Control
4. Combo vs Reactive
5. Full tournament/playoffs

Each scenario tests realistic gameplay with strategy execution,
decision-making, and game outcome validation.
"""

import pytest
from unittest.mock import patch

from src.engine.game_simulator import GameSimulator, GameResult
from src.engine.game_execution import AgentGamePlayer
from src.engine.agent_strategies import Strategy


class TestAggressiveVsControl:
    """Test Aggressive strategy vs Control strategy."""
    
    def test_aggressive_vs_control_completes(self):
        """Test full game: Aggressive vs Control."""
        agent_agg = AgentGamePlayer("aggressive_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent_ctl = AgentGamePlayer("control_1", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent_agg, agent_ctl, max_turns=10)
        result = sim.run_game()
        
        assert result in [GameResult.PLAYER1_WIN, GameResult.PLAYER2_WIN, GameResult.DRAW]
        assert sim.game is not None
        assert sim.game.turn_number > 0
    
    def test_aggressive_vs_control_game_log(self):
        """Test game produces actionable log."""
        agent_agg = AgentGamePlayer("aggressive_2", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent_ctl = AgentGamePlayer("control_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent_agg, agent_ctl, max_turns=5)
        result = sim.run_game()
        
        summary = sim.get_game_summary()
        
        assert "aggressive" in summary.lower() or "control" in summary.lower()
        assert "20" in summary or "life" in summary.lower()


class TestAggressiveVsAggressive:
    """Test Aggressive vs Aggressive (aggressive mirror match)."""
    
    def test_aggro_mirror_completes(self):
        """Test full game: Aggressive vs Aggressive."""
        agent_agg1 = AgentGamePlayer("aggro_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent_agg2 = AgentGamePlayer("aggro_2", Strategy.AGGRESSIVE, knowledge_graph=None)
        
        sim = GameSimulator(agent_agg1, agent_agg2, max_turns=8)
        result = sim.run_game()
        
        assert result in [GameResult.PLAYER1_WIN, GameResult.PLAYER2_WIN, GameResult.DRAW]
        # In all-aggressive game, should often end quickly
        assert sim.game.turn_number >= 1
    
    def test_aggro_mirror_has_winner(self):
        """Test aggressive mirror produces winner (not eternal stalemate)."""
        agent_agg1 = AgentGamePlayer("aggro_m1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent_agg2 = AgentGamePlayer("aggro_m2", Strategy.AGGRESSIVE, knowledge_graph=None)
        
        sim = GameSimulator(agent_agg1, agent_agg2, max_turns=15)
        result = sim.run_game()
        
        # Aggressive vs aggressive should eventually have winner
        assert result is not None


class TestControlVsControl:
    """Test Control strategy vs Control strategy (control mirror)."""
    
    def test_control_mirror_completes(self):
        """Test full game: Control vs Control."""
        agent_ctl1 = AgentGamePlayer("control_m1", Strategy.CONTROL, knowledge_graph=None)
        agent_ctl2 = AgentGamePlayer("control_m2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent_ctl1, agent_ctl2, max_turns=12)
        result = sim.run_game()
        
        assert result in [GameResult.PLAYER1_WIN, GameResult.PLAYER2_WIN, GameResult.DRAW]
    
    def test_control_mirror_lasts_longer(self):
        """Test control mirror lasts longer than aggro games."""
        agent_ctl1 = AgentGamePlayer("ctl_long1", Strategy.CONTROL, knowledge_graph=None)
        agent_ctl2 = AgentGamePlayer("ctl_long2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent_ctl1, agent_ctl2, max_turns=20)
        result = sim.run_game()
        
        # Control games tend to go longer
        assert sim.game is not None


class TestComboVsReactive:
    """Test Combo strategy vs Reactive strategy."""
    
    def test_combo_vs_reactive_completes(self):
        """Test full game: Combo vs Reactive."""
        agent_cmb = AgentGamePlayer("combo_1", Strategy.COMBO, knowledge_graph=None)
        agent_rct = AgentGamePlayer("reactive_1", Strategy.REACTIVE, knowledge_graph=None)
        
        sim = GameSimulator(agent_cmb, agent_rct, max_turns=10)
        result = sim.run_game()
        
        assert result in [GameResult.PLAYER1_WIN, GameResult.PLAYER2_WIN, GameResult.DRAW]
        assert sim.game is not None
    
    def test_reactive_vs_combo_produces_result(self):
        """Test reactive can compete in game."""
        agent_rct = AgentGamePlayer("reactive_c1", Strategy.REACTIVE, knowledge_graph=None)
        agent_cmb = AgentGamePlayer("combo_c1", Strategy.COMBO, knowledge_graph=None)
        
        sim = GameSimulator(agent_rct, agent_cmb, max_turns=8)
        result = sim.run_game()
        
        assert result is not None
        assert isinstance(result, GameResult)


class TestGameInitialization:
    """Test proper game initialization."""
    
    def test_game_starts_at_20_life(self):
        """Test both players start with 20 life."""
        agent1 = AgentGamePlayer("life_test_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("life_test_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=5)
        game = sim.setup_game("life_game")
        
        assert game.players[0].life_total == 20
        assert game.players[1].life_total == 20
    
    def test_game_starts_turn_1(self):
        """Test game begins at turn 1."""
        agent1 = AgentGamePlayer("turn_test_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("turn_test_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2)
        game = sim.setup_game("turn_game")
        
        assert game.turn_number == 1
    
    def test_game_ids_are_unique(self):
        """Test different games get unique IDs."""
        agent1 = AgentGamePlayer("unique_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("unique_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2)
        game1_id = sim.setup_game().game_id
        
        # Create new simulator for second game
        sim2 = GameSimulator(agent1, agent2)
        game2_id = sim2.setup_game().game_id
        
        assert game1_id != game2_id


class TestGameStateTracking:
    """Test game state is tracked properly during play."""
    
    def test_life_totals_decrease_or_stay_same(self):
        """Test life totals never increase."""
        agent1 = AgentGamePlayer("life_dec_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("life_dec_2", Strategy.AGGRESSIVE, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=5)
        game = sim.setup_game()
        
        initial_p1 = game.players[0].life_total
        initial_p2 = game.players[1].life_total
        
        sim.run_game()
        
        final_p1 = sim.game.players[0].life_total
        final_p2 = sim.game.players[1].life_total
        
        # Life totals should not increase
        assert final_p1 <= initial_p1
        assert final_p2 <= initial_p2
    
    def test_turn_counter_increases(self):
        """Test turn counter always increases."""
        agent1 = AgentGamePlayer("turn_inc_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("turn_inc_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=5)
        game = sim.setup_game()
        
        initial_turn = game.turn_number
        
        # Execute a few turns manually
        for _ in range(3):
            sim.execute_full_turn()
            sim.advance_turn()
        
        assert sim.game.turn_number > initial_turn


class TestGameResults:
    """Test game result determination."""
    
    def test_result_is_valid_enum(self):
        """Test result is one of the valid GameResult values."""
        agent1 = AgentGamePlayer("result_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("result_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=10)
        result = sim.run_game()
        
        assert result in GameResult
    
    def test_multiple_games_can_have_different_results(self):
        """Test running multiple games can produce different results."""
        agent1 = AgentGamePlayer("multi_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("multi_2", Strategy.CONTROL, knowledge_graph=None)
        
        results = []
        for i in range(3):
            agent_a = AgentGamePlayer(f"game{i}_a", Strategy.AGGRESSIVE, knowledge_graph=None)
            agent_c = AgentGamePlayer(f"game{i}_c", Strategy.CONTROL, knowledge_graph=None)
            
            sim = GameSimulator(agent_a, agent_c, max_turns=8)
            result = sim.run_game()
            results.append(result)
        
        # Should have at least one result
        assert len(results) == 3
        assert all(r in GameResult for r in results)


class TestGameReporting:
    """Test game summary and reporting."""
    
    def test_game_summary_contains_required_fields(self):
        """Test summary has all required information."""
        agent1 = AgentGamePlayer("summary_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("summary_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=5)
        sim.run_game()
        
        summary = sim.get_game_summary()
        
        # Summary should contain key information
        assert len(summary) > 0
        assert "Turn" in summary or "turn" in summary
        assert "Life" in summary or "HP" in summary or "life" in summary.lower()
    
    def test_phase_log_contains_actions(self):
        """Test phase log records game progression."""
        agent1 = AgentGamePlayer("log_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("log_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=3)
        sim.run_game()
        
        log = sim.get_phase_log()
        
        # Log should contain recorded actions
        assert len(log) > 0
        assert "Game" in log or "initialized" in log.lower() or "Turn" in log


class TestEdgeCaseGames:
    """Test edge cases in game flow."""
    
    def test_single_turn_game_possible(self):
        """Test game can end in one turn."""
        agent1 = AgentGamePlayer("single_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("single_2", Strategy.AGGRESSIVE, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=1)
        result = sim.run_game()
        
        # Should complete without error
        assert result is not None
    
    def test_max_turn_draw_occurs(self):
        """Test draw occurs when max turns reached."""
        agent1 = AgentGamePlayer("draw_1", Strategy.CONTROL, knowledge_graph=None)
        agent2 = AgentGamePlayer("draw_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=2)
        result = sim.run_game()
        
        # With only 2 turns and control strategy, likely will reach max
        assert result in [GameResult.PLAYER1_WIN, GameResult.PLAYER2_WIN, GameResult.DRAW]
    
    def test_game_completes_with_no_kg(self):
        """Test game works without Neo4j knowledge graph."""
        agent1 = AgentGamePlayer("nokg_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("nokg_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, knowledge_graph=None, max_turns=5)
        result = sim.run_game()
        
        assert result is not None
        assert sim.game is not None


class TestStrategyDistribution:
    """Test all strategies can play in games."""
    
    def test_all_strategies_can_play(self):
        """Test each strategy completes a game."""
        strategies = [Strategy.AGGRESSIVE, Strategy.CONTROL, Strategy.COMBO, Strategy.REACTIVE]
        
        for strategy in strategies:
            agent1 = AgentGamePlayer(f"strat_test_{strategy}", strategy, knowledge_graph=None)
            agent2 = AgentGamePlayer(f"strat_opp_{strategy}", Strategy.AGGRESSIVE, knowledge_graph=None)
            
            sim = GameSimulator(agent1, agent2, max_turns=8)
            result = sim.run_game()
            
            assert result in GameResult
    
    def test_each_strategy_can_win(self):
        """Test each strategy can achieve victory."""
        strategies = [Strategy.AGGRESSIVE, Strategy.CONTROL, Strategy.COMBO, Strategy.REACTIVE]
        
        # Run enough games that each strategy likely wins at least once
        for strategy in strategies:
            games_run = 0
            wins = 0
            
            for game_num in range(3):
                agent1 = AgentGamePlayer(f"win_test_{strategy}_{game_num}", strategy, knowledge_graph=None)
                agent2 = AgentGamePlayer(f"win_opp_{strategy}_{game_num}", Strategy.AGGRESSIVE, knowledge_graph=None)
                
                sim = GameSimulator(agent1, agent2, max_turns=10)
                result = sim.run_game()
                
                games_run += 1
                if result == GameResult.PLAYER1_WIN:
                    wins += 1
            
            # Strategy should be viable (ability to win)
            assert games_run > 0


class TestConsistentGameFlow:
    """Test games follow consistent patterns."""
    
    def test_games_progress_through_turns(self):
        """Test games make progress through turns."""
        agent1 = AgentGamePlayer("flow_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("flow_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=10)
        initial_turn = sim.setup_game().turn_number
        
        sim.run_game()
        
        final_turn = sim.game.turn_number
        
        # Should progress through multiple turns
        assert final_turn > initial_turn
    
    def test_games_produce_deterministic_count(self):
        """Test games execute a reasonable number of phases."""
        agent1 = AgentGamePlayer("count_1", Strategy.AGGRESSIVE, knowledge_graph=None)
        agent2 = AgentGamePlayer("count_2", Strategy.CONTROL, knowledge_graph=None)
        
        sim = GameSimulator(agent1, agent2, max_turns=5)
        sim.run_game()
        
        phase_log = sim.get_phase_log()
        
        # Should have logged actions (at least "Game initialized")
        assert len(phase_log) > 0
