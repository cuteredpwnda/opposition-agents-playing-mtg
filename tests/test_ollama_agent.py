#!/usr/bin/env python
"""Test Ollama agent in a game.

Run with Ollama:
  ollama run mistral
  python tests/test_ollama_agent.py

Falls back to RandomAgent if Ollama is unavailable.
"""

import asyncio
import pytest

from src.agents.random_agent import RandomAgent
from src.agents.llm_agent import OllamaAgent
from src.orchestrator.game_runner import GameRunner, GameConfig


@pytest.mark.asyncio
async def test_ollama_agent_in_game():
    """Test OllamaAgent playing a full game."""
    config = GameConfig(format="standard", starting_life=20, max_turns=10)
    runner = GameRunner(config)
    
    # Create agents
    agents = {
        "Alice": OllamaAgent(player_id="Alice", name="Ollama Alice"),
        "Bob": RandomAgent(player_id="Bob", name="Random Bob"),
    }
    
    # Simple balanced deck
    deck = [
        {"name": "Mountain", "mana_cost": "", "type_line": "Land",
         "oracle_text": "{T}: Add {R}.", "power": None, "toughness": None,
         "cmc": 0, "keywords": [], "set": "DOM"}
        for _ in range(10)
    ] + [
        {"name": "Plains", "mana_cost": "", "type_line": "Land",
         "oracle_text": "{T}: Add {W}.", "power": None, "toughness": None,
         "cmc": 0, "keywords": [], "set": "DOM"}
        for _ in range(10)
    ] + [
        {"name": "Goblin", "mana_cost": "{R}", "type_line": "Creature — Goblin",
         "oracle_text": "", "power": "1", "toughness": "1", "cmc": 1,
         "keywords": [], "set": "DOM"}
        for _ in range(10)
    ] + [
        {"name": "Soldier", "mana_cost": "{W}", "type_line": "Creature — Human Soldier",
         "oracle_text": "", "power": "1", "toughness": "1", "cmc": 1,
         "keywords": [], "set": "DOM"}
        for _ in range(10)
    ]
    
    decks = {"Alice": deck.copy(), "Bob": deck.copy()}
    
    # Run game
    result = await runner.run_game(agents, decks)
    
    # Verify game completed
    assert result.game_over, "Game should have completed"
    assert result.winner or result.is_draw, "Should have a winner or be a draw"
    
    print(f"\n{'='*60}")
    print(f"GAME COMPLETED")
    print(f"{'='*60}")
    if result.winner:
        print(f"Winner: {result.winner.name} ({result.winner.player_id})")
    else:
        print("Draw")
    print(f"Turns: {result.turns}")
    print(f"Final life totals:")
    for player in result.state.players:
        print(f"  {player.name}: {player.life_total}")
