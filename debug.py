"""
Debug script to trace game execution.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.DEBUG)

from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig


async def main():
    config = GameConfig(format="standard", starting_life=20, max_turns=5)
    runner = GameRunner(config)
    
    agent1 = RandomAgent(player_id="Alice", name="Random Alice")
    agent2 = RandomAgent(player_id="Bob", name="Random Bob")
    
    agents = {"Alice": agent1, "Bob": agent2}
    
    # Simple deck
    deck = [
        {
            "name": "Plains",
            "mana_cost": "",
            "type_line": "Land — Plains",
            "oracle_text": "{T}: Add {W}.",
            "power": None,
            "toughness": None,
            "cmc": 0,
            "keywords": [],
            "set": "DOM",  # IMPORTANT: without a set, cards are treated as tokens!
        }
        for _ in range(20)
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
        for _ in range(20)
    ]
    
    decks = {"Alice": deck.copy(), "Bob": deck.copy()}
    
    # Setup game
    game_state = runner._setup_game(agents, decks)
    
    print(f"Players: {[p.player_id for p in game_state.players]}")
    print(f"Alice life: {game_state.players[0].life_total}")
    print(f"Bob life: {game_state.players[1].life_total}")
    print(f"Active player: {game_state.active_player.player_id}")
    print()
    
    # Play first turn
    print("=" * 60)
    print("PLAYING TURN 1 (Alice's turn)")
    print("=" * 60)
    print(f"Before _play_turn: game_over={game_state.game_over}, players={len(game_state.players)}, alice_life={game_state.players[0].life_total}, bob_life={game_state.players[1].life_total}")
    
    try:
        game_state = await runner._play_turn(game_state, agents)
    except Exception as e:
        print(f"ERROR in _play_turn: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"After _play_turn: game_over={game_state.game_over}, players={len(game_state.players)}, winner={game_state.winner.player_id if game_state.winner else None}")
    if len(game_state.players) > 0:
        print(f"  Alice life: {game_state.players[0].life_total}")
    if len(game_state.players) > 1:
        print(f"  Bob life: {game_state.players[1].life_total}")
    
    if not game_state.game_over:
        print("=" * 60)
        print("PLAYING TURN 2 (Bob's turn)")
        print("=" * 60)
        
        game_state = await runner._play_turn(game_state, agents)
        
        print(f"\nAfter Bob's turn:")
        print(f"  Active player: {game_state.active_player.player_id}")
        print(f"  Alice life: {game_state.players[0].life_total}")
        print(f"  Bob life: {game_state.players[1].life_total}")
        print(f"  Game over: {game_state.game_over}")
        print(f"  Winner: {game_state.winner.player_id if game_state.winner else None}")
        print(f"  Turn number: {game_state.turn_number}")
    
    print(f"\nTotal log events: {len(game_state.game_log)}")
    print("\nAll game log events:")
    for event in game_state.game_log:
        print(f"  {event}")


if __name__ == "__main__":
    asyncio.run(main())
