"""
Simple MTG game runner for testing.

Usage:
  python main.py                     # Random vs Random (standard)
  python main.py --ollama            # Ollama vs Random
  python main.py --commander         # 4-player Commander
  python main.py --fusion            # LLM-Fusion agent vs Random
  python main.py --rl-train          # Run RL self-play training
  python main.py --transfer           # Standard→Commander transfer learning
  python main.py --help              # Show options
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.random_agent import RandomAgent
from src.agents.llm_agent import OllamaAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.world_model.data_sources.self_play_collector import SelfPlayCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_simple_deck() -> list[dict]:
    """Create a minimal 40-card test deck."""
    deck = []
    
    # 20 basic lands
    for i in range(10):
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
    
    for i in range(10):
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
    
    # 8 creatures costing {1}{W}
    for i in range(8):
        deck.append({
            "name": "Soldier",
            "mana_cost": "{1}{W}",
            "type_line": "Creature — Soldier",
            "oracle_text": "",
            "power": "2",
            "toughness": "2",
            "cmc": 2,
            "keywords": [],
            "set": "DOM",
        })
    
    # 12 creatures costing {R}
    for i in range(12):
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
    
    return deck


async def main(args):
    """Run a test game."""
    logger.info("=" * 60)
    logger.info("MTG Agent Framework")
    logger.info("=" * 60)

    # Determine format and player count
    if args.commander:
        game_format = "commander"
        starting_life = 40
        max_turns = 30
        player_names = ["Alice", "Bob", "Charlie", "Diana"]
    else:
        game_format = "standard"
        starting_life = 20
        max_turns = 15
        player_names = ["Alice", "Bob"]

    config = GameConfig(format=game_format, starting_life=starting_life, max_turns=max_turns)
    collector = SelfPlayCollector()
    runner = GameRunner(config, self_play_collector=collector)
    
    # Create agents
    def make_agent(player_id: str, agent_type: str):
        if agent_type == "human":
            from src.agents.human_agent import HumanAgent
            return HumanAgent(player_id=player_id, name=f"Human {player_id}")
        if agent_type == "ollama":
            return OllamaAgent(player_id=player_id, name=f"Ollama {player_id}")
        if agent_type == "fusion":
            try:
                from src.agents.llm_fusion_agent import LLMFusionAgent
                return LLMFusionAgent(player_id=player_id, name=f"Fusion {player_id}")
            except Exception:
                logger.warning("LLM-Fusion agent unavailable, falling back to Random")
        return RandomAgent(player_id=player_id, name=f"Random {player_id}")

    # First player uses requested agent type, rest are random unless commander with human
    if args.human:
        first_type = "human"
    else:
        first_type = "fusion" if args.fusion else ("ollama" if args.ollama else "random")

    agents = {}
    for i, name in enumerate(player_names):
        atype = first_type if i == 0 else "random"
        agents[name] = make_agent(name, atype)
    
    # Build decks (larger for Commander)
    deck = build_simple_deck()
    if game_format == "commander":
        deck = (deck * 3)[:100]  # 100-card Commander decks
    
    decks = {name: deck.copy() for name in player_names}
    
    logger.info("\nGame configuration:")
    logger.info(f"  Format: {game_format}")
    logger.info(f"  Starting life: {starting_life}")
    logger.info(f"  Max turns: {max_turns}")
    logger.info(f"  Players: {', '.join(f'{n} ({type(agents[n]).__name__})' for n in player_names)}")
    logger.info(f"  Deck size: {len(deck)} cards per player")
    
    # Run the game
    logger.info("\n" + "=" * 60)
    logger.info("GAME START")
    logger.info("=" * 60 + "\n")
    
    try:
        result = await runner.run_game(agents, decks)
        
        logger.info("\n" + "=" * 60)
        logger.info("GAME OVER")
        logger.info("=" * 60)
        logger.info(f"Winner: {result.winner}")
        logger.info(f"Duration: {result.turns} turns")
        logger.info(f"Total events: {len(result.log)}")
        
        if result.log:
            logger.info("\nFirst 30 events:")
            for event in result.log[:30]:
                logger.info(f"  {event}")
            
            if len(result.log) > 30:
                logger.info(f"\n  ... ({len(result.log) - 30} more events)")
        
        logger.info("\nTest completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\nGame error: {e}", exc_info=True)
        return 1


async def run_rl_training(args):
    """Run RL self-play training."""
    logger.info("=" * 60)
    logger.info("RL Self-Play Training")
    logger.info("=" * 60)

    from src.training.rl_trainer import RLTrainer, RLConfig

    game_format = "commander" if args.commander else "standard"
    config = RLConfig(
        game_format=game_format,
        starting_life=40 if args.commander else 20,
        num_players=4 if args.commander else 2,
        num_iterations=args.rl_iters,
        games_per_iteration=args.rl_games,
    )

    trainer = RLTrainer(config)
    await trainer.train()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTG Agent Arena")
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama agent for first player (requires: ollama run mistral)"
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Use LLM-Fusion agent for first player"
    )
    parser.add_argument(
        "--commander",
        action="store_true",
        help="4-player Commander format"
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Let human play first seat (interactive)"
    )
    parser.add_argument(
        "--rl-train",
        action="store_true",
        help="Run RL self-play training instead of a demo game"
    )
    parser.add_argument(
        "--rl-iters",
        type=int, default=20,
        help="Number of RL training iterations (default: 20)"
    )
    parser.add_argument(
        "--rl-games",
        type=int, default=5,
        help="Games per RL iteration (default: 5)"
    )
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="Run Standard→Commander transfer learning curriculum"
    )
    args = parser.parse_args()
    
    if args.transfer:
        from src.training.transfer_learning import TransferTrainer, TransferConfig
        tc = TransferConfig(
            standard_iterations=max(args.rl_iters // 2, 10),
            commander_iterations=max(args.rl_iters // 3, 10),
            joint_iterations=max(args.rl_iters // 5, 5),
        )
        exit_code = asyncio.run(TransferTrainer(tc).run()) or 0
        sys.exit(exit_code)
    elif args.rl_train:
        exit_code = asyncio.run(run_rl_training(args))
    else:
        exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
