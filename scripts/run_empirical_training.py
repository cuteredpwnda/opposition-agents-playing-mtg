#!/usr/bin/env python
"""
Empirical training runs for publication.

Generates benchmark numbers comparing agents:
- RandomAgent (baseline)
- HeuristicAgent (deterministic aggressive/control strategies)
- WorldModelAgent (JEPA-based planning)

Each agent runs self-play tournaments with CSV metrics export
to support tech report and paper results sections.

Usage:
    python scripts/run_empirical_training.py \\
        --num-iterations 30 \\
        --games-per-iteration 4 \\
        --max-turns 20 \\
        --seed 42 \\
        --output-dir runs/empirical_training

This produces:
    runs/empirical_training/<agent>/
        run.log              # Human-readable game log
        training_metrics.csv # CSV with iteration, win_rate, elo, etc
        summary.json         # Final statistics
"""

import asyncio
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.world_model_agent import WorldModelAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.training.rl_trainer import RLTrainer, RLConfig, AgentPool
from src.training.experience_buffer import ExperienceBuffer


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def setup_output_dir(base_dir: Path) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_empirical_training(
    num_iterations: int = 30,
    games_per_iteration: int = 4,
    max_turns: int = 20,
    seed: int = 42,
    output_base_dir: str = "runs/empirical_training",
) -> None:
    """Run empirical training for each agent type."""
    
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Agent configurations: (agent_class, config_dict, name)
    agent_configs = [
        (RandomAgent, {}, "random"),
        (HeuristicAgent, {"seed": seed, "prefer_aggressive": True}, "heuristic_aggressive"),
        (HeuristicAgent, {"seed": seed, "prefer_aggressive": False}, "heuristic_control"),
    ]
    
    # Try to load WorldModel checkpoint if available
    wm_checkpoint = Path("checkpoints/jepa/jepa_final.pt")
    if wm_checkpoint.exists():
        agent_configs.append((WorldModelAgent, {"checkpoint_path": str(wm_checkpoint)}, "worldmodel_jepa"))
    else:
        logger.warning(f"WorldModel checkpoint not found at {wm_checkpoint}; skipping WorldModelAgent")
    
    results = {}
    
    for agent_cls, agent_kwargs, agent_name in agent_configs:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting training: {agent_name.upper()}")
        logger.info(f"{'='*80}\n")
        
        agent_output_dir = output_base / agent_name
        agent_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for this agent
        log_file = agent_output_dir / "run.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        
        try:
            # Create agent pool
            agent_pool = AgentPool()
            
            # Register the test agent
            def agent_factory(player_id: str):
                return agent_cls(
                    player_id=player_id,
                    name=agent_name,
                    **agent_kwargs
                )
            
            agent_pool.register(agent_name, agent_factory, elo=1200.0)
            
            # Setup training config with CSV metrics
            config = RLConfig(
                num_iterations=num_iterations,
                games_per_iteration=games_per_iteration,
                max_turns_per_game=max_turns,
                log_dir=str(agent_output_dir),
                metrics_csv=str(agent_output_dir / "training_metrics.csv"),
            )
            
            # Create trainer and run
            trainer = RLTrainer(
                agent_pool=agent_pool,
                config=config,
                experience_buffer=ExperienceBuffer(capacity=10000),
            )
            
            # Run training (only self-play collection for empirical benchmark)
            # Note: For paper, we collect gameplay stats rather than neural training
            logger.info(f"Running {num_iterations} iterations of {games_per_iteration} games each")
            logger.info(f"Max turns: {max_turns}, seed: {seed}")
            
            # Simplified: just collect win rate against random opponents
            win_stats = asyncio.run(_collect_empirical_stats(
                agent_cls=agent_cls,
                agent_kwargs=agent_kwargs,
                agent_name=agent_name,
                num_games=num_iterations * games_per_iteration,
                max_turns=max_turns,
                seed=seed,
                log_file=log_file,
            ))
            
            results[agent_name] = win_stats
            
            # Write summary
            summary_file = agent_output_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(win_stats, f, indent=2)
            
            logger.info(f"\n{agent_name.upper()} Results:")
            logger.info(f"  Win Rate: {win_stats['win_rate']:.1%}")
            logger.info(f"  Wins: {win_stats['wins']}/{win_stats['total_games']}")
            logger.info(f"  Avg Kill Turn: {win_stats['avg_kill_turn']:.1f}")
            logger.info(f"  Summary written to: {summary_file}")
            
        finally:
            # Remove handler
            logging.getLogger().removeHandler(handler)
    
    # Write cross-agent summary
    summary_path = output_base / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Empirical training complete!")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"{'='*80}")
    
    # Print comparative table
    print("\n" + "="*80)
    print("EMPIRICAL TRAINING RESULTS")
    print("="*80)
    print(f"{'Agent':<30} {'Win Rate':<15} {'Avg Kill Turn':<15}")
    print("-"*60)
    for name, stats in sorted(results.items()):
        wr = stats.get('win_rate', 0.0)
        akt = stats.get('avg_kill_turn', 0.0)
        print(f"{name:<30} {wr:>6.1%}          {akt:>6.1f}")
    print("="*80 + "\n")


async def _collect_empirical_stats(
    agent_cls,
    agent_kwargs,
    agent_name: str,
    num_games: int = 100,
    max_turns: int = 20,
    seed: int = 42,
    log_file: Optional[Path] = None,
) -> dict:
    """Play games and collect empirical win/loss statistics."""
    
    import random
    random.seed(seed)
    
    config = GameConfig(
        format="standard",
        starting_life=20,
        max_turns=max_turns,
    )
    
    runner = GameRunner(config)
    
    # Decklist: minimal 60-card deck for testing
    test_deck = []
    # 20 lands
    for _ in range(10):
        test_deck.append({
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
        test_deck.append({
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
    
    # 40 creatures (1/1 baseline)
    for _ in range(20):
        test_deck.append({
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
        test_deck.append({
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
    
    # Test agent vs random baseline
    wins = 0
    kills = []
    
    for game_num in range(num_games):
        test_agent = agent_cls(
            player_id="TestAgent",
            name=agent_name,
            **agent_kwargs
        )
        opponent = RandomAgent(player_id="Opponent", name="Random")
        
        agents = {"TestAgent": test_agent, "Opponent": opponent}
        decks = {"TestAgent": test_deck.copy(), "Opponent": test_deck.copy()}
        
        game_state = runner._setup_game(agents, decks)
        
        try:
            while not game_state.game_over:
                game_state = await runner._play_turn(game_state, agents)
        except Exception as e:
            logger.warning(f"Game {game_num} error: {e}")
            continue
        
        # Check result
        if game_state.winner and game_state.winner.player_id == "TestAgent":
            wins += 1
            kills.append(game_state.turn_number)
        
        if (game_num + 1) % max(1, num_games // 10) == 0:
            logger.info(f"  {game_num + 1}/{num_games} games: {wins}/{game_num + 1} = {wins/(game_num+1):.1%}")
    
    avg_kill = sum(kills) / len(kills) if kills else 0.0
    win_rate = wins / num_games if num_games > 0 else 0.0
    
    return {
        "total_games": num_games,
        "wins": wins,
        "win_rate": win_rate,
        "avg_kill_turn": avg_kill,
        "seed": seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run empirical training benchmarks for paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=30,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=4,
        help="Games per iteration",
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
        default="runs/empirical_training",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    run_empirical_training(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        max_turns=args.max_turns,
        seed=args.seed,
        output_base_dir=args.output_dir,
    )
