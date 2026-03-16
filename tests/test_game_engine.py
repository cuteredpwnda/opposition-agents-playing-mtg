#!/usr/bin/env python3
"""
Quick integration test to verify the game engine works end-to-end.
"""

import asyncio
import sys
from src.engine.game_state import GameState, PlayerState, Phase, Zone
from src.engine.rules_engine import RulesEngine
from src.agents.random_agent import RandomAgent


async def test_basic_game():
    """Run a simple game with random agents to verify engine integrity."""
    
    print("=" * 60)
    print("MTG Game Engine Integration Test")
    print("=" * 60)
    
    # Setup
    engine = RulesEngine()
    players = [
        PlayerState(player_id="player_1", name="Alice", life_total=20),
        PlayerState(player_id="player_2", name="Bob", life_total=20),
    ]
    
    # Create simple mock game state with a few cards
    from src.engine.game_state import CardInstance
    cards = []
    
    # Add some lands and creatures to player 1's hand
    for i in range(7):
        card = CardInstance(
            instance_id=f"p1_card_{i}",
            card_data={
                "name": f"Plains" if i < 3 else "Goblin",
                "type_line": "Land" if i < 3 else "Creature",
                "mana_cost": "" if i < 3 else "{1}",
                "cmc": 0 if i < 3 else 1,
                "oracle_text": "",
                "power": None if i < 3 else "1",
                "toughness": None if i < 3 else "1",
            },
            zone=Zone.HAND,
            owner_id="player_1",
            controller_id="player_1",
        )
        cards.append(card)
    
    # Add lands to player 2's hand
    for i in range(7):
        card = CardInstance(
            instance_id=f"p2_card_{i}",
            card_data={
                "name": "Swamp",
                "type_line": "Land",
                "mana_cost": "",
                "cmc": 0,
                "oracle_text": "",
            },
            zone=Zone.HAND,
            owner_id="player_2",
            controller_id="player_2",
        )
        cards.append(card)
    
    # Add libraries
    for i in range(53):
        card = CardInstance(
            instance_id=f"p1_lib_{i}",
            card_data={"name": "Library Card", "type_line": ""},
            zone=Zone.LIBRARY,
            owner_id="player_1",
            controller_id="player_1",
        )
        cards.append(card)
        
        card = CardInstance(
            instance_id=f"p2_lib_{i}",
            card_data={"name": "Library Card", "type_line": ""},
            zone=Zone.LIBRARY,
            owner_id="player_2",
            controller_id="player_2",
        )
        cards.append(card)
    
    state = GameState(
        format="standard",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=players,
        cards=cards,
    )
    
    print(f"\n✓ Game state initialized with {len(cards)} cards")
    print(f"  Players: {', '.join(p.name for p in state.players)}")
    print(f"  Starting life: {state.players[0].life_total}")
    
    # Test legal action generation
    print(f"\n✓ Testing legal action generation...")
    legal_actions = engine.get_legal_actions(state, "player_1")
    print(f"  Legal actions for player_1: {len(legal_actions)} total")
    
    # Count action types
    action_types = {}
    for action in legal_actions:
        atype = action.action_type.value
        action_types[atype] = action_types.get(atype, 0) + 1
    
    for atype, count in sorted(action_types.items()):
        print(f"    - {atype}: {count}")
    
    if len(legal_actions) == 0:
        print("  ✗ ERROR: No legal actions generated!")
        return False
    
    # Test action execution
    print(f"\n✓ Testing action execution...")
    action = legal_actions[0]
    print(f"  Executing: {action.action_type.value}")
    
    new_state = engine.execute_action(state, action)
    print(f"  ✓ Action executed successfully")
    
    # Test SBA checking
    print(f"\n✓ Testing state-based actions...")
    events = engine.check_state_based_actions(new_state)
    print(f"  SBA events: {len(events)}")
    for event in events[:5]:
        print(f"    - {event}")
    
    print(f"\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_basic_game())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
