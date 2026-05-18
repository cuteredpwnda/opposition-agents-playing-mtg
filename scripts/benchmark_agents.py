#!/usr/bin/env python
"""
Simple empirical benchmark: run agents against each other and collect win rates.

Usage:
    python scripts/benchmark_agents.py --num-games 20 --seed 42
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.orchestrator.game_runner import GameRunner, GameConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_deck():
    """Create a minimal 60-card test deck."""
    deck = []
    
    # 20 lands
    for _ in range(10):
        deck.append({
            "name": "Mountain",
            "mana_cost": "",
            "type_line": "Land — Mountain",
            "oracle_text": "{T}: Add {R}.",
            "power": None,
            "toughness": None,
            "cmc": 0,
            "keywords": [],
            "set": "DOM",
        })
    for _ in range(10):
        deck.append({
            "name": "Plains",
            "mana_cost": "",
            "type_line": "Land — Plains",
            "oracle_text": "{T}: Add {W}.",
            "power": None,
            "toughness": None,
            "cmc": 0,
            "keywords": [],
            "set": "DOM",
        })
    
    # 40 creatures (1/1)
    for _ in range(20):
        deck.append({
            "name": "Goblin",
            "mana_cost": "{R}",
            "type_line": "Creature — Goblin",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "cmc": 1,
            "keywords": [],
            "set": "DOM",
        })
    for _ in range(20):
        deck.append({
            "name": "Soldier",
            "mana_cost": "{W}",
            "type_line": "Creature — Human Soldier",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "cmc": 1,
            "keywords": [],
            "set": "DOM",
        })
    
    return deck


async def run_benchmark(
    num_games: int = 20,
    max_turns: int = 20,
    seed: int = 42,
    output_dir: str = "runs/benchmark",
) -> None:
    """Run empirical benchmark comparing agents."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test deck
    test_deck = create_test_deck()
    
    # Agents to benchmark
    agents_config = [
        ("random", RandomAgent(player_id="random", name="Random")),
        ("heuristic_aggressive", HeuristicAgent(player_id="heur_agg", name="Heuristic(Aggressive)", seed=seed, prefer_aggressive=True)),
        ("heuristic_control", HeuristicAgent(player_id="heur_ctrl", name="Heuristic(Control)", seed=seed, prefer_aggressive=False)),
    ]
    
    config = GameConfig(
        format="standard",
        starting_life=20,
        max_turns=max_turns,
    )
    runner = GameRunner(config)
    
    results = {}
    
    # Run each agent 1v1 against random baseline
    for agent_name, test_agent in agents_config:
        logger.info(f"\nBenchmarking {agent_name}...")
        
        wins = 0
        losses = 0
        draw = 0
        kill_turns = []
        errors = 0
        
        for game_num in range(num_games):
            try:
                # Create a fresh instance of the test agent
                if agent_name == "random":
                    player_agent = RandomAgent(player_id=agent_name, name=agent_name)
                elif "aggressive" in agent_name:
                    player_agent = HeuristicAgent(
                        player_id=agent_name,
                        name=agent_name,
                        seed=seed + game_num,
                        prefer_aggressive=True
                    )
                else:  # control
                    player_agent = HeuristicAgent(
                        player_id=agent_name,
                        name=agent_name,
                        seed=seed + game_num,
                        prefer_aggressive=False
                    )
                
                # Opponent is always random
                opponent = RandomAgent(player_id="opponent", name="Random")
                
                agents = {agent_name: player_agent, "opponent": opponent}
                decks = {agent_name: test_deck.copy(), "opponent": test_deck.copy()}
                
                # Setup and play game
                game_state = runner._setup_game(agents, decks)
                
                while not game_state.game_over and game_state.turn_number < max_turns:
                    game_state = await runner._play_turn(game_state, agents)
                
                # Record result
                if game_state.winner:
                    if game_state.winner.player_id == agent_name:
                        wins += 1
                        kill_turns.append(game_state.turn_number)
                    else:
                        losses += 1
                else:
                    draw += 1
                
                # Log progress
                if (game_num + 1) % max(1, num_games // 5) == 0:
                    wr = wins / (game_num + 1)
                    logger.info(f"  {game_num + 1}/{num_games}: {wins}W-{losses}L-{draw}D ({wr:.1%})")
                    
            except Exception as e:
                logger.warning(f"  Game {game_num} error: {e}")
                errors += 1
        
        # Compute stats
        total = wins + losses + draw
        win_rate = wins / total if total > 0 else 0.0
        avg_kill_turn = sum(kill_turns) / len(kill_turns) if kill_turns else 0.0
        
        results[agent_name] = {
            "wins": wins,
            "losses": losses,
            "draws": draw,
            "total_games": total,
            "win_rate": win_rate,
            "avg_kill_turn": avg_kill_turn,
            "errors": errors,
        }
        
        logger.info(f"{agent_name.upper()}: {win_rate:.1%} ({wins}W-{losses}L-{draw}D)")
        if kill_turns:
            logger.info(f"  Avg kill turn: {avg_kill_turn:.1f}")
    
    # Write results
    summary_file = output_path / "results.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("EMPIRICAL AGENT BENCHMARK")
    print("="*80)
    print(f"{'Agent':<25} {'Win Rate':<12} {'Record':<20} {'Avg Kill Turn':<15}")
    print("-"*80)
    for name, stats in sorted(results.items()):
        wr = stats['win_rate']
        record = f"{stats['wins']}W-{stats['losses']}L-{stats['draws']}D"
        akt = stats['avg_kill_turn']
        print(f"{name:<25} {wr:>6.1%}        {record:<20} {akt:>6.1f}")
    print("="*80)
    print(f"\nResults written to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Magic agents against each other",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=20,
        help="Number of games per agent",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Max turns per game",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/benchmark",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    asyncio.run(run_benchmark(
        num_games=args.num_games,
        max_turns=args.max_turns,
        seed=args.seed,
        output_dir=args.output_dir,
    ))
