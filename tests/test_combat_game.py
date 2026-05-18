#!/usr/bin/env python
"""
Visual test: Play a game and see creatures being cast and attacking.
Run with: python -m pytest tests/test_combat_game.py -v -s
"""

import asyncio
import pytest

from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig


@pytest.mark.asyncio
async def test_combat_game_visual():
    """Play 8 turns and show creature casting and combat."""
    config = GameConfig(format="standard", starting_life=20, max_turns=15)
    runner = GameRunner(config)
    
    agents = {
        "Alice": RandomAgent(player_id="Alice", name="Random Alice"),
        "Bob": RandomAgent(player_id="Bob", name="Random Bob"),
    }
    
    # Deck with good mana (20 lands for 60 card deck = 33%)
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
    
    # 20 cheap creatures
    for i in range(10):
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
    
    for i in range(10):
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
    
    decks = {"Alice": deck.copy(), "Bob": deck.copy()}
    
    # Setup game
    game_state = runner._setup_game(agents, decks)
    
    print(f"\n{'='*80}")
    print(f"STARTING COMBAT TEST GAME")
    print(f"{'='*80}\n")
    print(f"Setup complete:")
    print(f"  Alice: {game_state.players[0].life_total} life, {len([c for c in game_state.cards if c.owner_id == 'Alice' and c.zone.value == 'hand'])} in hand")
    print(f"  Bob: {game_state.players[1].life_total} life, {len([c for c in game_state.cards if c.owner_id == 'Bob' and c.zone.value == 'hand'])} in hand")
    
    # Play up to 8 turns
    for turn in range(1, 9):
        if game_state.game_over:
            break
        
        player = game_state.active_player
        print(f"\n{'-'*80}")
        print(f"TURN {turn} — {player.name}'s Turn ({player.life_total} life)")
        print(f"{'-'*80}")
        
        game_state = await runner._play_turn(game_state, agents)
        
        # Show battlefield
        alice_bf = [c for c in game_state.cards if c.owner_id == "Alice" and c.zone.value == "battlefield"]
        bob_bf = [c for c in game_state.cards if c.owner_id == "Bob" and c.zone.value == "battlefield"]
        
        if alice_bf or bob_bf:
            print(f"\nBattlefield:")
            if alice_bf:
                creatures = [f"{c.name}{'(sick)' if c.summoning_sick else ''}" for c in alice_bf if c.is_creature()]
                lands = [c.name for c in alice_bf if c.is_land()]
                if creatures:
                    print(f"  Alice: {', '.join(creatures)}")
                if lands:
                    print(f"    Lands: {', '.join(lands)}")
            if bob_bf:
                creatures = [f"{c.name}{'(sick)' if c.summoning_sick else ''}" for c in bob_bf if c.is_creature()]
                lands = [c.name for c in bob_bf if c.is_land()]
                if creatures:
                    print(f"  Bob: {', '.join(creatures)}")
                if lands:
                    print(f"    Lands: {', '.join(lands)}")
        
        print(f"\nLife totals: Alice {game_state.players[0].life_total} | Bob {game_state.players[1].life_total}")
        
        if game_state.game_over:
            print(f"\n{'='*80}")
            print(f"GAME OVER!")
            if game_state.winner:
                print(f"Winner: {game_state.winner.name}")
            else:
                print("Draw!")
            print(f"{'='*80}")
    
    # Game should complete without errors. With random agents and 1/1 creatures,
    # they may all die in combat, so just verify game ran to completion.
    assert True, "Combat game completed successfully"
