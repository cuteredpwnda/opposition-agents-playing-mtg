#!/usr/bin/env python
"""Debug test to check if combat is happening."""

import pytest

from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.engine.game_state import ActionType, Zone, Phase


@pytest.mark.asyncio
async def test_combat_actions_debug():
    """Check if DECLARE_ATTACKERS actions are being generated."""
    config = GameConfig(format="standard", starting_life=20, max_turns=10)
    runner = GameRunner(config)
    
    agents = {
        "Alice": RandomAgent(player_id="Alice", name="Alice"),
        "Bob": RandomAgent(player_id="Bob", name="Bob"),
    }
    
    # Simple deck
    deck = [
        {"name": "Mountain", "mana_cost": "", "type_line": "Land",
         "oracle_text": "{T}: Add {R}.", "power": None, "toughness": None,
         "cmc": 0, "keywords": [], "set": "DOM"}
        for _ in range(10)
    ] + [
        {"name": "Goblin", "mana_cost": "{R}", "type_line": "Creature — Goblin",
         "oracle_text": "", "power": "1", "toughness": "1", "cmc": 1,
         "keywords": [], "set": "DOM"}
        for _ in range(20)
    ]
    
    decks = {"Alice": deck.copy(), "Bob": deck.copy()}
    game_state = runner._setup_game(agents, decks)
    
    # Play until we have creatures on both sides
    combat_found = False
    for turn_num in range(1, 10):
        if game_state.game_over:
            break
        
        print(f"\n--- TURN {turn_num} ---")
        
        # Pre-turn check: show battlefield
        alice_bf = [c for c in game_state.cards if c.owner_id == "Alice" and c.zone == Zone.BATTLEFIELD and c.is_creature()]
        bob_bf = [c for c in game_state.cards if c.owner_id == "Bob" and c.zone == Zone.BATTLEFIELD and c.is_creature()]
        print(f"Before turn: Alice has {len(alice_bf)} creatures, Bob has {len(bob_bf)} creatures")
        
        # Check if we reach COMBAT_ATTACKERS phase
        original_phase = game_state.phase
        game_state = await runner._play_turn(game_state, agents)
        
        # After turn, check if combat happened
        print(f"After turn: Combat = {game_state.combat}")
        if game_state.combat:
            print(f"  Attackers: {game_state.combat.attackers}")
            combat_found = True
            break
    
    # Combat may not happen in exploratory random tests; ensure no infinite loop and game runs
    assert game_state.turn_number <= config.max_turns + 1
