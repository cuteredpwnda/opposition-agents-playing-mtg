"""
Example: Running MTG Agent Tournaments

This script demonstrates how to use the tournament system to:
1. Run individual games between strategies
2. Conduct round-robin tournaments
3. Play best-of series
4. Analyze strategy statistics
5. Generate tournament reports

Usage:
  python examples/tournament_demo.py                    # Quick demo (5 games)
  python examples/tournament_demo.py --full-roundrobin  # Full 6-strategy round-robin
  python examples/tournament_demo.py --series           # Best-of series
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.tournament import Tournament, TournamentRunner
from src.engine.agent_strategies import Strategy


def demo_single_games():
    """Example: Play individual games between strategies."""
    print("\n" + "="*70)
    print("DEMO: Single Games Between Strategies")
    print("="*70)
    
    tournament = Tournament(max_turns=20)
    
    matchups = [
        (Strategy.AGGRESSIVE, Strategy.CONTROL),
        (Strategy.AGGRESSIVE, Strategy.AGGRESSIVE),
        (Strategy.CONTROL, Strategy.COMBO),
        (Strategy.REACTIVE, Strategy.AGGRESSIVE),
        (Strategy.COMBO, Strategy.REACTIVE),
    ]
    
    print(f"\nPlaying {len(matchups)} games...\n")
    
    for strat1, strat2 in matchups:
        record = tournament.play_game(strat1, strat2)
        
        result_str = "Draw"
        if record.result.value == "player1_win":
            result_str = f"{strat1.value} wins"
        elif record.result.value == "player2_win":
            result_str = f"{strat2.value} wins"
        
        print(f"  {strat1.value:12} vs {strat2.value:12} → {result_str:20} ({record.turns} turns)")
    
    print("\n" + tournament.get_stats_summary())


def demo_round_robin():
    """Example: Full round-robin tournament."""
    print("\n" + "="*70)
    print("DEMO: Round-Robin Tournament (1 match per pair)")
    print("="*70)
    
    runner = TournamentRunner(max_turns=20)
    
    print("\nRunning round-robin tournament...")
    print("(Each strategy plays every other strategy once)\n")
    
    runner.run_round_robin(matches_per_pair=1)
    
    print(runner.get_summary())
    
    print("\nRanking by Win Rate:")
    print("-" * 70)
    ranking = runner.tournament.get_ranking()
    for i, (strategy, stats) in enumerate(ranking, 1):
        print(f"  {i}. {strategy.value:12} - {stats.wins}W {stats.losses}L {stats.draws}D ({stats.win_rate:.1%} win rate)")


def demo_best_of_series():
    """Example: Best-of series tournament."""
    print("\n" + "="*70)
    print("DEMO: Best-of Series")
    print("="*70)
    
    runner = TournamentRunner(max_turns=20)
    
    series_matchups = [
        (Strategy.AGGRESSIVE, Strategy.CONTROL),
        (Strategy.AGGRESSIVE, Strategy.REACTIVE),
        (Strategy.CONTROL, Strategy.COMBO),
    ]
    
    print("\nRunning best-of-3 series between strategy pairs...\n")
    
    for strat1, strat2 in series_matchups:
        winner, tournament = runner.run_series(strat1, strat2, num_games=3)
        
        print(f"  {strat1.value:12} vs {strat2.value:12} → Winner: {winner.value}")
        
        # Get the last 3 games played (the series we just ran)
        games = tournament.games[-3:]
        for i, record in enumerate(games, 1):
            result_str = "Draw"
            if record.result.value == "player1_win":
                result_str = f"{strat1.value} wins"
            elif record.result.value == "player2_win":
                result_str = f"{strat2.value} wins"
            print(f"    Game {i}: {result_str}")
        print()


def demo_extended_analysis():
    """Example: Extended tournament with detailed analysis."""
    print("\n" + "="*70)
    print("DEMO: Extended Analysis (2 matches per pair)")
    print("="*70)
    
    runner = TournamentRunner(max_turns=20)
    
    print("\nRunning extended round-robin tournament...")
    print("(Each strategy plays every other strategy twice)\n")
    
    runner.run_round_robin(matches_per_pair=2)
    
    summary = runner.get_summary()
    print(summary)
    
    # Detailed matchup analysis
    print("\nDetailed Matchup Analysis:")
    print("-" * 70)
    
    ranking = runner.tournament.get_ranking()
    for strategy, stats in ranking:
        print(f"\n{strategy.value.upper()}:")
        
        for vs_strat in sorted(Strategy):
            if strategy == vs_strat:
                continue
            
            if vs_strat in stats.matchups:
                matchup = stats.matchups[vs_strat]
                total = matchup["wins"] + matchup["losses"] + matchup["draws"]
                if total > 0:
                    wr = matchup["wins"] / total
                    print(f"  vs {vs_strat.value:12}: {matchup['wins']}-{matchup['losses']}-{matchup['draws']} record ({wr:.1%} win rate)")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("MTG OPPOSITION AGENTS - TOURNAMENT SYSTEM DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows different tournament formats and strategy analysis.")
    print("Each game is limited to 20 turns for faster execution.\n")
    
    # Run demonstrations
    demo_single_games()
    demo_round_robin()
    demo_best_of_series()
    demo_extended_analysis()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nFor more information:")
    print("  - See src/engine/tournament.py for Tournament API")
    print("  - See tests/test_tournament.py for additional examples")
    print("  - Use TournamentRunner for high-level tournament management")
    print("\n")


if __name__ == "__main__":
    main()
