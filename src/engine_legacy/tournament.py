"""
Tournament System & Strategy Analytics.

Runs comprehensive tournaments between different agent strategies,
tracks win rates, and generates statistical analysis of strategy
effectiveness in various matchups.

Tournament modes:
1. Round-robin: Each strategy plays every other strategy
2. Single elimination: Bracket-style tournament
3. Best-of-series: Multiple games, track aggregate results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import defaultdict

from src.engine.game_simulator import GameSimulator, GameResult
from src.engine.game_execution import AgentGamePlayer
from src.engine.agent_strategies import Strategy


@dataclass
class GameRecord:
    """Record of a single game result."""
    player1: str
    player2: str
    strategy1: Strategy
    strategy2: Strategy
    winner: Optional[str]  # player_id or None for draw
    result: GameResult
    turns: int = 0
    p1_life_final: int = 0
    p2_life_final: int = 0


@dataclass
class StrategyStats:
    """Statistics for a single strategy."""
    strategy: Strategy
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_games: int = 0
    win_rate: float = 0.0
    matchups: dict[Strategy, dict[str, int]] = field(default_factory=dict)  # vs each strategy
    
    def update_from_result(self, result: GameResult, vs_strategy: Strategy, won: bool) -> None:
        """Update stats from a game result."""
        self.total_games += 1
        
        if result == GameResult.DRAW:
            self.draws += 1
        elif won:
            self.wins += 1
        else:
            self.losses += 1
        
        self.win_rate = self.wins / self.total_games if self.total_games > 0 else 0.0
        
        # Track per-matchup stats
        if vs_strategy not in self.matchups:
            self.matchups[vs_strategy] = {"wins": 0, "losses": 0, "draws": 0}
        
        if result == GameResult.DRAW:
            self.matchups[vs_strategy]["draws"] += 1
        elif won:
            self.matchups[vs_strategy]["wins"] += 1
        else:
            self.matchups[vs_strategy]["losses"] += 1


class Tournament:
    """Manages strategy tournaments."""
    
    def __init__(self, max_turns: int = 20):
        """Initialize tournament.
        
        Args:
            max_turns: Maximum turns per game
        """
        self.max_turns = max_turns
        self.games: list[GameRecord] = []
        self.stats: dict[Strategy, StrategyStats] = {
            s: StrategyStats(strategy=s) for s in Strategy
        }
    
    def play_game(self, 
                  strategy1: Strategy,
                  strategy2: Strategy,
                  game_id: str = None) -> GameRecord:
        """Play a single game between two strategies.
        
        Args:
            strategy1: First agent's strategy
            strategy2: Second agent's strategy
            game_id: Optional game identifier
            
        Returns:
            GameRecord with result
        """
        # Create agents
        agent1 = AgentGamePlayer(f"{strategy1.value}_p1_game", strategy1, knowledge_graph=None)
        agent2 = AgentGamePlayer(f"{strategy2.value}_p2_game", strategy2, knowledge_graph=None)
        
        # Run game
        sim = GameSimulator(agent1, agent2, knowledge_graph=None, max_turns=self.max_turns)
        result = sim.run_game()
        
        # Record result
        winner = None
        if result == GameResult.PLAYER1_WIN:
            winner = agent1.player_id
        elif result == GameResult.PLAYER2_WIN:
            winner = agent2.player_id

        life_by_id = {}
        if sim.game:
            life_by_id = {p.player_id: p.life_total for p in sim.game.players}
        
        record = GameRecord(
            player1=agent1.player_id,
            player2=agent2.player_id,
            strategy1=strategy1,
            strategy2=strategy2,
            winner=winner,
            result=result,
            turns=sim.game.turn_number if sim.game else 0,
            p1_life_final=life_by_id.get(agent1.player_id, 0),
            p2_life_final=life_by_id.get(agent2.player_id, 0),
        )
        
        self.games.append(record)
        
        # Update stats
        player1_won = result == GameResult.PLAYER1_WIN
        self.stats[strategy1].update_from_result(result, strategy2, player1_won)
        self.stats[strategy2].update_from_result(result, strategy1, not player1_won)
        
        return record
    
    def round_robin(self, num_matches: int = 1) -> list[GameRecord]:
        """Run round-robin tournament (each strategy plays each other).
        
        Args:
            num_matches: Times each pair plays
            
        Returns:
            List of game records
        """
        strategies = list(Strategy)
        records = []
        
        for _ in range(num_matches):
            for i, strat1 in enumerate(strategies):
                for strat2 in strategies[i+1:]:
                    record = self.play_game(strat1, strat2)
                    records.append(record)
        
        return records
    
    def best_of_series(self,
                       strategy1: Strategy,
                       strategy2: Strategy,
                       num_games: int = 3) -> tuple[Strategy, list[GameRecord]]:
        """Play best-of series between two strategies.
        
        Args:
            strategy1: First strategy
            strategy2: Second strategy
            num_games: Number of games to play
            
        Returns:
            Tuple of (winner_strategy, game_records)
        """
        records = []
        wins1 = 0
        wins2 = 0
        
        for _ in range(num_games):
            record = self.play_game(strategy1, strategy2)
            records.append(record)
            
            if record.result == GameResult.PLAYER1_WIN:
                wins1 += 1
            elif record.result == GameResult.PLAYER2_WIN:
                wins2 += 1
        
        winner = strategy1 if wins1 > wins2 else strategy2
        return winner, records
    
    def get_stats_summary(self) -> str:
        """Get formatted summary of tournament statistics.
        
        Returns:
            Multi-line string with tournament results
        """
        summary = "\n" + "="*70 + "\n"
        summary += "TOURNAMENT STATISTICS\n"
        summary += "="*70 + "\n\n"
        
        # Overall stats by strategy
        summary += "Strategy Performance (Overall):\n"
        summary += "-" * 70 + "\n"
        summary += f"{'Strategy':<15} {'Wins':<8} {'Losses':<8} {'Draws':<8} {'Win Rate':<10}\n"
        summary += "-" * 70 + "\n"
        
        for strategy in sorted(Strategy, key=lambda s: self.stats[s].win_rate, reverse=True):
            stats = self.stats[strategy]
            summary += f"{strategy.value:<15} {stats.wins:<8} {stats.losses:<8} {stats.draws:<8} {stats.win_rate:>8.1%}\n"
        
        summary += "\n"
        
        # Matchup breakdown
        summary += "Head-to-Head Matchups:\n"
        summary += "-" * 70 + "\n"
        
        for strat1 in sorted(Strategy):
            summary += f"\n{strat1.value}:\n"
            stats = self.stats[strat1]
            
            for strat2 in sorted(Strategy):
                if strat1 == strat2:
                    continue
                
                if strat2 in stats.matchups:
                    matchup = stats.matchups[strat2]
                    total = matchup["wins"] + matchup["losses"] + matchup["draws"]
                    if total > 0:
                        wr = matchup["wins"] / total
                        summary += f"  vs {strat2.value:<12} {matchup['wins']}-{matchup['losses']}-{matchup['draws']} ({wr:>5.1%})\n"
        
        summary += "\n" + "="*70 + "\n"
        summary += f"Total Games Played: {len(self.games)}\n"
        
        # Game length stats
        if self.games:
            avg_turns = sum(g.turns for g in self.games) / len(self.games)
            summary += f"Average Game Length: {avg_turns:.1f} turns\n"
        
        summary += "="*70 + "\n"
        
        return summary
    
    def get_ranking(self) -> list[tuple[Strategy, StrategyStats]]:
        """Get strategies ranked by win rate.
        
        Returns:
            List of (strategy, stats) tuples sorted by win rate
        """
        return sorted(
            [(s, self.stats[s]) for s in Strategy],
            key=lambda x: x[1].win_rate,
            reverse=True
        )


class TournamentRunner:
    """High-level tournament coordination."""
    
    def __init__(self, max_turns: int = 20):
        """Initialize runner.
        
        Args:
            max_turns: Max turns per game
        """
        self.tournament = Tournament(max_turns)
    
    def run_round_robin(self, matches_per_pair: int = 1) -> Tournament:
        """Run full round-robin tournament.
        
        Args:
            matches_per_pair: Times each pair plays
            
        Returns:
            Tournament with results
        """
        self.tournament.round_robin(matches_per_pair)
        return self.tournament
    
    def run_series(self,
                   strat1: Strategy,
                   strat2: Strategy,
                   num_games: int = 3) -> tuple[Strategy, Tournament]:
        """Run best-of series.
        
        Args:
            strat1: First strategy
            strat2: Second strategy
            num_games: Games to play
            
        Returns:
            Tuple of (winner, tournament)
        """
        winner, _ = self.tournament.best_of_series(strat1, strat2, num_games)
        return winner, self.tournament
    
    def get_summary(self) -> str:
        """Get tournament summary.
        
        Returns:
            Formatted summary string
        """
        return self.tournament.get_stats_summary()
    
    def get_winner(self) -> Strategy:
        """Get tournament winner by win rate.
        
        Returns:
            Strategy with highest win rate
        """
        ranking = self.tournament.get_ranking()
        return ranking[0][0] if ranking else None
