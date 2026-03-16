"""
Simple MTG game runner for testing.

Run with: python main.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig

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


async def main():
    """Run a test game."""
    logger.info("=" * 60)
    logger.info("MTG Agent Framework — Phase 1 Test")
    logger.info("=" * 60)
    
    config = GameConfig(format="standard", starting_life=20, max_turns=15)
    runner = GameRunner(config)
    
    # Create 2 random agents
    agent1 = RandomAgent(player_id="Alice", name="Random Alice")
    agent2 = RandomAgent(player_id="Bob", name="Random Bob")
    
    agents = {
        "Alice": agent1,
        "Bob": agent2,
    }
    
    # Build test decks
    deck = build_simple_deck()
    decks = {
        "Alice": deck.copy(),
        "Bob": deck.copy(),
    }
    
    logger.info("\nGame configuration:")
    logger.info(f"  Format: {config.format}")
    logger.info(f"  Starting life: {config.starting_life}")
    logger.info(f"  Max turns: {config.max_turns}")
    logger.info(f"  Players: {', '.join(agents.keys())}")
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
        
        logger.info("\n✅ Test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Game error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
