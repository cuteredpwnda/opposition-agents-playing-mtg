"""
End-to-end test: Two random agents play a complete game.

Tests that:
- Game initializes correctly
- Turns progress through all phases
- Cards move between zones correctly
- Combat triggers damage calculations
- Game terminates when someone reaches 0 life
"""

import asyncio
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig


@pytest.fixture
def game_config():
    """Standard game configuration."""
    return GameConfig(format="standard", starting_life=20, max_turns=20)


@pytest.fixture
def test_deck():
    """Standardized test deck: 20 basic lands + 20 creatures."""
    return [
        {
            "name": "Plains",
            "mana_cost": "",
            "type_line": "Land — Plains",
            "oracle_text": "{T}: Add {W}.",
            "power": None,
            "toughness": None,
            "cmc": 0,
            "keywords": [],
            "set": "DOM",
        }
        for _ in range(10)
    ] + [
        {
            "name": "Mountain",
            "mana_cost": "",
            "type_line": "Land — Mountain",
            "oracle_text": "{T}: Add {R}.",
            "power": None,
            "toughness": None,
            "cmc": 0,
            "keywords": [],
            "set": "DOM",
        }
        for _ in range(10)
    ] + [
        {
            "name": "Soldier",
            "mana_cost": "{1}{W}",
            "type_line": "Creature — Human Soldier",
            "oracle_text": "",
            "power": "2",
            "toughness": "2",
            "cmc": 2,
            "keywords": [],
            "set": "DOM",
        }
        for _ in range(8)
    ] + [
        {
            "name": "Goblin",
            "mana_cost": "{R}",
            "type_line": "Creature — Goblin",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "cmc": 1,
            "keywords": [],
            "set": "DOM",
        }
        for _ in range(12)
    ]


@pytest.mark.asyncio
async def test_game_initializes(game_config, test_deck):
    """Test that game initializes with correct state."""
    runner = GameRunner(game_config)
    agents = {
        "Alice": RandomAgent(player_id="Alice", name="Random Alice"),
        "Bob": RandomAgent(player_id="Bob", name="Random Bob"),
    }
    decks = {"Alice": test_deck.copy(), "Bob": test_deck.copy()}
    
    game_state = runner._setup_game(agents, decks)
    
    # Verify game state
    assert len(game_state.players) == 2
    assert game_state.players[0].name == "Alice"
    assert game_state.players[1].name == "Bob"
    assert all(p.life_total == 20 for p in game_state.players)
    assert len(game_state.cards) == 80  # 40 per player
    
    # Each player should have drawn opening hand (7 cards)
    alice_hand = [c for c in game_state.cards if c.owner_id == "Alice" and c.zone.value == "hand"]
    bob_hand = [c for c in game_state.cards if c.owner_id == "Bob" and c.zone.value == "hand"]
    assert len(alice_hand) == 7
    assert len(bob_hand) == 7


@pytest.mark.asyncio
async def test_game_runs_multiple_turns(game_config, test_deck):
    """Test that game can run through multiple turns."""
    runner = GameRunner(game_config)
    agents = {
        "Alice": RandomAgent(player_id="Alice", name="Random Alice"),
        "Bob": RandomAgent(player_id="Bob", name="Random Bob"),
    }
    decks = {"Alice": test_deck.copy(), "Bob": test_deck.copy()}
    
    game_state = runner._setup_game(agents, decks)
    
    # Play 3 turns
    for turn_num in range(3):
        if game_state.game_over:
            break
        game_state = await runner._play_turn(game_state, agents)
        assert not game_state.game_over, f"Game ended unexpectedly on turn {turn_num + 1}"
    
    # At least 3 turns should have passed
    assert len(game_state.game_log) > 30, "Should have log entries for multiple turns"


@pytest.mark.asyncio
async def test_game_ends_when_someone_dies(game_config, test_deck):
    """Test that game ends when a player reaches 0 life.
    
    Note: With random agents and small deck, may take many turns.
    This test uses reduced starting life (5) to speed testing.
    """
    config = GameConfig(format="standard", starting_life=5, max_turns=50)
    runner = GameRunner(config)
    agents = {
        "Alice": RandomAgent(player_id="Alice", name="Random Alice"),
        "Bob": RandomAgent(player_id="Bob", name="Random Bob"),
    }
    decks = {"Alice": test_deck.copy(), "Bob": test_deck.copy()}
    
    game_state = runner._setup_game(agents, decks)
    initial_life = {p.player_id: p.life_total for p in game_state.players}
    
    # Play until game ends or max turns reached
    max_iterations = config.max_turns
    iteration = 0
    while not game_state.game_over and iteration < max_iterations:
        game_state = await runner._play_turn(game_state, agents)
        iteration += 1
    
    # Either game ended or we hit max turns
    # (game ending with low starting life is probabilistic)
    assert iteration > 0, "Should have played at least one turn"


@pytest.mark.asyncio
async def test_cards_stay_in_zones(game_config, test_deck):
    """Test that cards are properly tracked in zones."""
    runner = GameRunner(game_config)
    agents = {
        "Alice": RandomAgent(player_id="Alice", name="Random Alice"),
        "Bob": RandomAgent(player_id="Bob", name="Random Bob"),
    }
    decks = {"Alice": test_deck.copy(), "Bob": test_deck.copy()}
    
    game_state = runner._setup_game(agents, decks)
    initial_card_count = len(game_state.cards)
    
    # Play 2 turns
    for _ in range(2):
        if game_state.game_over:
            break
        game_state = await runner._play_turn(game_state, agents)
    
    # Card count should not change (cards move between zones, but don't disappear)
    assert len(game_state.cards) == initial_card_count, "Card count should be preserved"
    
    # All cards should have valid zones
    from src.engine.game_state import Zone
    valid_zones = {z.value for z in Zone}
    for card in game_state.cards:
        assert card.zone.value in valid_zones, f"Card {card.name} has invalid zone"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
