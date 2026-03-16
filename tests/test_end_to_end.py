"""
End-to-end test: Two random agents play a game.

Run with: python -m pytest tests/test_end_to_end.py -v -s
"""

import asyncio
import logging
from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_test_deck() -> list[dict]:
    """Create a simple 60-card deck with basic lands and creatures for testing."""
    deck = []
    
    # 24 basic lands
    for i in range(12):
        deck.append({
            "name": "Plains",
            "mana_cost": "",
            "type_line": "Land — Plains",
            "oracle_text": "{T}: Add {W}.",
            "power": None,
            "toughness": None,
            "cmc": 0,
            "keywords": [],
        })
    
    for i in range(12):
        deck.append({
            "name": "Mountain",
            "mana_cost": "",
            "type_line": "Land — Mountain",
            "oracle_text": "{T}: Add {R}.",
            "power": None,
            "toughness": None,
            "cmc": 0,
            "keywords": [],
        })
    
    # 12 simple 2/2 creatures costing {1}{W}
    for i in range(12):
        deck.append({
            "name": "Soldier Token",
            "mana_cost": "{1}{W}",
            "type_line": "Creature — Soldier",
            "oracle_text": "",
            "power": "2",
            "toughness": "2",
            "cmc": 2,
            "keywords": [],
        })
    
    # 12 simple 1/1 creatures costing {R}
    for i in range(12):
        deck.append({
            "name": "Goblin Token",
            "mana_cost": "{R}",
            "type_line": "Creature — Goblin",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "cmc": 1,
            "keywords": [],
        })
    
    return deck[:60]  # Exactly 60 cards


async def test_random_vs_random():
    """Test: Two random agents play a game."""
    logger.info("Starting random vs random game...")
    
    config = GameConfig(format="standard", starting_life=20, max_turns=20)
    runner = GameRunner(config)
    
    # Create 2 random agents
    agent1 = RandomAgent(player_id="Player1", name="Random 1")
    agent2 = RandomAgent(player_id="Player2", name="Random 2")
    
    agents = {
        "Player1": agent1,
        "Player2": agent2,
    }
    
    # Build simple test decks
    deck = build_test_deck()
    decks = {
        "Player1": deck.copy(),
        "Player2": deck.copy(),
    }
    
    # Run the game
    result = await runner.run_game(agents, decks)
    
    logger.info(f"Game completed!")
    logger.info(f"  Winner: {result.winner}")
    logger.info(f"  Turns: {result.turns}")
    logger.info(f"  Log length: {len(result.log)} events")
    
    if result.log:
        logger.info("\nFirst 20 game events:")
        for event in result.log[:20]:
            logger.info(f"    {event}")
    
    assert result.game_over or result.turns == config.max_turns, "Game should complete"
    logger.info("\n✓ Test passed!")


if __name__ == "__main__":
    asyncio.run(test_random_vs_random())
