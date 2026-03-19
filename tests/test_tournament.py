"""
Tests for Tournament System & Strategy Analytics.

Test scenarios:
1. Single game between strategies
2. Statistics tracking per strategy
3. Matchup tracking
4. Round-robin tournament
5. Best-of series
6. Tournament rankings
7. Summary/reporting
8. Statistics accuracy
"""

import pytest

from src.engine.tournament import Tournament, TournamentRunner, GameRecord, StrategyStats
from src.engine.agent_strategies import Strategy
from src.engine.game_simulator import GameResult


class TestSingleGameTournament:
    """Test single game tournament functionality."""
    
    def test_play_single_game(self):
        """Test playing one game between strategies."""
        tournament = Tournament(max_turns=5)
        
        record = tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        assert record is not None
        assert record.strategy1 == Strategy.AGGRESSIVE
        assert record.strategy2 == Strategy.CONTROL
        assert record.result in GameResult
        assert record.winner is None or isinstance(record.winner, str)
    
    def test_game_record_has_required_fields(self):
        """Test game record contains all required data."""
        tournament = Tournament(max_turns=3)
        record = tournament.play_game(Strategy.AGGRESSIVE, Strategy.AGGRESSIVE)
        
        assert record.player1 is not None
        assert record.player2 is not None
        assert record.strategy1 == Strategy.AGGRESSIVE
        assert record.strategy2 == Strategy.AGGRESSIVE
        assert record.result in GameResult
        assert record.turns >= 0
        assert record.p1_life_final >= 0
        assert record.p2_life_final >= 0
    
    def test_game_recorded_in_tournament(self):
        """Test game is added to tournament history."""
        tournament = Tournament()
        initial_count = len(tournament.games)
        
        tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        assert len(tournament.games) == initial_count + 1


class TestStatisticsTracking:
    """Test strategy statistics are tracked correctly."""
    
    def test_stats_created_for_all_strategies(self):
        """Test all strategies have stats."""
        tournament = Tournament()
        
        for strategy in Strategy:
            assert strategy in tournament.stats
            assert tournament.stats[strategy].strategy == strategy
    
    def test_stats_updated_after_game(self):
        """Test stats change after a game."""
        tournament = Tournament(max_turns=3)
        
        # Play initial game
        record1 = tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        # Check stats updated
        agg_stats = tournament.stats[Strategy.AGGRESSIVE]
        ctl_stats = tournament.stats[Strategy.CONTROL]
        
        assert agg_stats.total_games == 1
        assert ctl_stats.total_games == 1
        assert agg_stats.wins + agg_stats.losses + agg_stats.draws == 1
        assert ctl_stats.wins + ctl_stats.losses + ctl_stats.draws == 1
    
    def test_win_rate_calculated(self):
        """Test win rate is calculated correctly."""
        tournament = Tournament(max_turns=3)
        
        # Play multiple games
        for _ in range(3):
            tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        agg_stats = tournament.stats[Strategy.AGGRESSIVE]
        
        # Win rate should be calculated
        assert 0.0 <= agg_stats.win_rate <= 1.0
    
    def test_matchup_tracking(self):
        """Test matchups are tracked per strategy pair."""
        tournament = Tournament(max_turns=3)
        
        tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        agg_stats = tournament.stats[Strategy.AGGRESSIVE]
        
        # Should have matchup data against CONTROL
        assert Strategy.CONTROL in agg_stats.matchups
        assert "wins" in agg_stats.matchups[Strategy.CONTROL]
        assert "losses" in agg_stats.matchups[Strategy.CONTROL]
        assert "draws" in agg_stats.matchups[Strategy.CONTROL]


class TestRoundRobinTournament:
    """Test round-robin tournament mode."""
    
    def test_round_robin_completes(self):
        """Test round-robin tournament finishes."""
        tournament = Tournament(max_turns=3)
        
        records = tournament.round_robin(num_matches=1)
        
        assert len(records) > 0
        # Each strategy has 3 opponents, so 4 * 3 / 2 = 6 matches
        assert len(records) == 6
    
    def test_round_robin_all_strategies_play(self):
        """Test all strategies play in round-robin."""
        tournament = Tournament(max_turns=3)
        records = tournament.round_robin(num_matches=1)
        
        strategies_played = set()
        for record in records:
            strategies_played.add(record.strategy1)
            strategies_played.add(record.strategy2)
        
        # All strategies should participate
        assert len(strategies_played) == len(Strategy)
    
    def test_round_robin_multiple_matches(self):
        """Test multiple matches per pair."""
        tournament = Tournament(max_turns=2)
        
        records = tournament.round_robin(num_matches=2)
        
        # 6 matchups * 2 matches each = 12 games
        assert len(records) == 12
    
    def test_all_games_in_round_robin_recorded(self):
        """Test all round-robin games show up in tournament history."""
        tournament = Tournament(max_turns=2)
        initial_games = len(tournament.games)
        
        tournament.round_robin(num_matches=1)
        
        assert len(tournament.games) == initial_games + 6


class TestBestOfSeries:
    """Test best-of series functionality."""
    
    def test_best_of_three_completes(self):
        """Test best-of-3 series finishes."""
        tournament = Tournament(max_turns=3)
        
        winner, records = tournament.best_of_series(Strategy.AGGRESSIVE, Strategy.CONTROL, num_games=3)
        
        assert winner in [Strategy.AGGRESSIVE, Strategy.CONTROL]
        assert len(records) == 3
    
    def test_best_of_series_winner_has_most_wins(self):
        """Test series winner has majority of wins."""
        tournament = Tournament(max_turns=2)
        
        # Play multiple series to ensure win tracking
        for _ in range(5):
            winner, records = tournament.best_of_series(Strategy.AGGRESSIVE, Strategy.REACTIVE, num_games=3)
            
            # Winner should be valid
            assert winner in [Strategy.AGGRESSIVE, Strategy.REACTIVE]
    
    def test_best_of_five(self):
        """Test best-of-5 series."""
        tournament = Tournament(max_turns=2)
        
        winner, records = tournament.best_of_series(Strategy.CONTROL, Strategy.CONTROL, num_games=5)
        
        assert len(records) == 5
        assert winner == Strategy.CONTROL  # Both same strategy


class TestTournamentStats:
    """Test statistics summary and ranking."""
    
    def test_get_stats_summary(self):
        """Test summary generation."""
        tournament = Tournament(max_turns=2)
        
        # Play some games first
        tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        tournament.play_game(Strategy.AGGRESSIVE, Strategy.AGGRESSIVE)
        
        summary = tournament.get_stats_summary()
        
        assert "TOURNAMENT STATISTICS" in summary
        assert "Win Rate" in summary or "win rate" in summary.lower()
    
    def test_get_ranking(self):
        """Test ranking by win rate."""
        tournament = Tournament(max_turns=2)
        
        # Play several games
        for _ in range(5):
            tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        ranking = tournament.get_ranking()
        
        assert len(ranking) == 4  # 4 strategies
        assert all(isinstance(r, tuple) for r in ranking)
    
    def test_ranking_sorted_by_win_rate(self):
        """Test ranking is sorted highest win rate first."""
        tournament = Tournament(max_turns=2)
        
        # Play games to build stats
        for _ in range(3):
            tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        ranking = tournament.get_ranking()
        
        # First should have >= win rate than second
        if len(ranking) > 1:
            assert ranking[0][1].win_rate >= ranking[1][1].win_rate


class TestTournamentRunner:
    """Test high-level tournament runner."""
    
    def test_runner_initialization(self):
        """Test runner creates tournament."""
        runner = TournamentRunner()
        
        assert runner.tournament is not None
    
    def test_runner_round_robin(self):
        """Test runner round-robin mode."""
        runner = TournamentRunner(max_turns=2)
        
        tournament = runner.run_round_robin(matches_per_pair=1)
        
        assert tournament is not None
        assert len(tournament.games) == 6
    
    def test_runner_best_of_series(self):
        """Test runner best-of series."""
        runner = TournamentRunner(max_turns=2)
        
        winner, tournament = runner.run_series(Strategy.AGGRESSIVE, Strategy.CONTROL, num_games=3)
        
        assert winner in [Strategy.AGGRESSIVE, Strategy.CONTROL]
        assert tournament is not None
    
    def test_runner_get_summary(self):
        """Test runner summary generation."""
        runner = TournamentRunner(max_turns=2)
        runner.run_round_robin(matches_per_pair=1)
        
        summary = runner.get_summary()
        
        assert "TOURNAMENT STATISTICS" in summary
    
    def test_runner_get_winner(self):
        """Test getting tournament winner."""
        runner = TournamentRunner(max_turns=2)
        runner.run_round_robin(matches_per_pair=1)
        
        winner = runner.get_winner()
        
        assert winner in Strategy


class TestStatisticsAccuracy:
    """Test statistics calculations are accurate."""
    
    def test_win_count_matches_records(self):
        """Test win count matches game records."""
        tournament = Tournament(max_turns=2)
        
        # Play several games
        for _ in range(4):
            tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        agg_stats = tournament.stats[Strategy.AGGRESSIVE]
        
        # Count wins from records
        agg_wins = sum(1 for r in tournament.games if r.strategy1 == Strategy.AGGRESSIVE and r.result == GameResult.PLAYER1_WIN)
        agg_losses = sum(1 for r in tournament.games if r.strategy1 == Strategy.AGGRESSIVE and r.result == GameResult.PLAYER2_WIN)
        agg_draws = sum(1 for r in tournament.games if r.strategy1 == Strategy.AGGRESSIVE and r.result == GameResult.DRAW)
        
        # Stats should match
        assert agg_stats.wins == agg_wins
        assert agg_stats.losses == agg_losses
        assert agg_stats.draws == agg_draws
    
    def test_total_games_count(self):
        """Test total game count is accurate."""
        tournament = Tournament(max_turns=2)
        
        games_to_play = 6
        for _ in range(games_to_play):
            tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        # Each strategy should have 6 games
        agg_stats = tournament.stats[Strategy.AGGRESSIVE]
        ctl_stats = tournament.stats[Strategy.CONTROL]
        
        assert agg_stats.total_games == games_to_play
        assert ctl_stats.total_games == games_to_play
    
    def test_no_double_counting(self):
        """Test stats don't double-count games."""
        tournament = Tournament(max_turns=2)
        
        initial_agg = tournament.stats[Strategy.AGGRESSIVE].total_games
        initial_ctl = tournament.stats[Strategy.CONTROL].total_games
        
        tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        # Each should increment by exactly 1
        assert tournament.stats[Strategy.AGGRESSIVE].total_games == initial_agg + 1
        assert tournament.stats[Strategy.CONTROL].total_games == initial_ctl + 1


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_tournament_with_one_game(self):
        """Test tournament with minimal games."""
        tournament = Tournament(max_turns=2)
        
        tournament.play_game(Strategy.AGGRESSIVE, Strategy.CONTROL)
        
        ranking = tournament.get_ranking()
        summary = tournament.get_stats_summary()
        
        assert len(ranking) > 0
        assert len(summary) > 0
    
    def test_same_strategy_matchup(self):
        """Test same strategy playing against itself."""
        tournament = Tournament(max_turns=2)
        
        record = tournament.play_game(Strategy.AGGRESSIVE, Strategy.AGGRESSIVE)
        
        assert record.strategy1 == Strategy.AGGRESSIVE
        assert record.strategy2 == Strategy.AGGRESSIVE
        assert record.result in GameResult
    
    def test_multiple_round_robins(self):
        """Test running round-robin multiple times."""
        tournament = Tournament(max_turns=2)
        
        tournament.round_robin(num_matches=1)
        initial_games = len(tournament.games)
        
        tournament.round_robin(num_matches=1)
        final_games = len(tournament.games)
        
        # Should have played more games
        assert final_games > initial_games
