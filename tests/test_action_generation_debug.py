#!/usr/bin/env python
"""Debug test to understand action generation and selection."""

import pytest

from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.engine.game_state import ActionType, Zone


@pytest.mark.asyncio
async def test_action_generation_debug():
    """Trace what's happening with action generation and agent selection."""
    config = GameConfig(format="standard", starting_life=20, max_turns=3)
    runner = GameRunner(config)
    
    agents = {
        "Alice": RandomAgent(player_id="Alice", name="Alice"),
        "Bob": RandomAgent(player_id="Bob", name="Bob"),
    }
    
    # Simple deck: 20 lands, 20 creatures
    deck = [
        {"name": "Mountain", "mana_cost": "", "type_line": "Land", "oracle_text": "{T}: Add {R}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "DOM"}
        for _ in range(10)
    ] + [
        {"name": "Plains", "mana_cost": "", "type_line": "Land", "oracle_text": "{T}: Add {W}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "DOM"}
        for _ in range(10)
    ] + [
        {"name": "Goblin", "mana_cost": "{R}", "type_line": "Creature — Goblin",
         "oracle_text": "", "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "DOM"}
        for _ in range(10)
    ] + [
        {"name": "Soldier", "mana_cost": "{W}", "type_line": "Creature — Human Soldier",
         "oracle_text": "", "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "DOM"}
        for _ in range(10)
    ]
    
    decks = {"Alice": deck.copy(), "Bob": deck.copy()}
    game_state = runner._setup_game(agents, decks)
    
    # Play turn 1
    print(f"\n{'='*80}")
    print(f"STARTING GAME - TURN 1, MAIN_1 PHASE")
    print(f"{'='*80}\n")
    
    from src.engine.game_state import Phase
    from src.engine.phases import is_main_phase
    
    # Manually step through to MAIN_1
    while not is_main_phase(game_state.phase):
        from src.engine.phases import PHASE_ORDER, advance_phase
        game_state.phase = PHASE_ORDER[PHASE_ORDER.index(game_state.phase) + 1]
    
    # Now we're in MAIN_1
    alice_id = game_state.active_player.player_id
    print(f"Active player: {alice_id}")
    
    # Show Alice's hand
    alice_hand = [c for c in game_state.cards if c.owner_id == alice_id and c.zone == Zone.HAND]
    print(f"\nAlice's hand ({len(alice_hand)} cards):")
    for card in alice_hand:
        print(f"  - {card.name} (cost: {card.mana_cost})")
    
    # Show legal actions
    legal = runner.engine.get_legal_actions(game_state, alice_id)
    print(f"\nLegal actions ({len(legal)} total):")
    action_types = {}
    for action in legal:
        t = action.action_type.name
        action_types[t] = action_types.get(t, 0) + 1
    for action_type, count in action_types.items():
        print(f"  {action_type}: {count}")
    
    # Step 1: Alice takes first action
    print(f"\n--- ACTION 1 ---")
    action1 = await agents[alice_id].decide_action(game_state, legal)
    print(f"Chosen action: {action1.action_type.name}")
    if action1.card_instance_id:
        card = next((c for c in game_state.cards if c.instance_id == action1.card_instance_id), None)
        if card:
            print(f"  Card: {card.name}")
    
    game_state = runner.engine.execute_action(game_state, action1)
    runner.engine.check_state_based_actions(game_state)
    
    # Show mana pool after action 1
    alice_player = next((p for p in game_state.players if p.player_id == alice_id), None)
    print(f"\nAlice's mana pool: {alice_player.mana_pool}")
    print(f"Alice's land plays remaining: {alice_player.land_plays_remaining}")
    
    # Show new legal actions
    legal2 = runner.engine.get_legal_actions(game_state, alice_id)
    print(f"\nLegal actions after action 1 ({len(legal2)} total):")
    action_types = {}
    for action in legal2:
        t = action.action_type.name
        action_types[t] = action_types.get(t, 0) + 1
    for action_type, count in action_types.items():
        print(f"  {action_type}: {count}")
    
    # Specifically check for CAST_SPELL actions
    cast_spells = [a for a in legal2 if a.action_type == ActionType.CAST_SPELL]
    if cast_spells:
        print(f"\nCAST_SPELL actions ({len(cast_spells)}):")
        for action in cast_spells:
            card = next((c for c in game_state.cards if c.instance_id == action.card_instance_id), None)
            if card:
                print(f"  - {card.name} ({card.mana_cost})")
    
    # Show battlefield
    alice_bf = [c for c in game_state.cards if c.owner_id == alice_id and c.zone == Zone.BATTLEFIELD]
    print(f"\nAlice's battlefield:")
    for card in alice_bf:
        print(f"  - {card.name} (tapped: {card.tapped})")
    
    # ACTION 2: Tap the Plains for mana
    if legal2:
        print(f"\n--- ACTION 2 ---")
        action2 = await agents[alice_id].decide_action(game_state, legal2)
        print(f"Chosen action: {action2.action_type.name}")
        if action2.card_instance_id:
            card = next((c for c in game_state.cards if c.instance_id == action2.card_instance_id), None)
            if card:
                print(f"  Card: {card.name}")
        
        game_state = runner.engine.execute_action(game_state, action2)
        runner.engine.check_state_based_actions(game_state)
        
        alice_player = next((p for p in game_state.players if p.player_id == alice_id), None)
        print(f"\nAlice's mana pool: {alice_player.mana_pool}")
        
        # Show new legal actions
        legal3 = runner.engine.get_legal_actions(game_state, alice_id)
        print(f"\nLegal actions after action 2 ({len(legal3)} total):")
        action_types = {}
        for action in legal3:
            t = action.action_type.name
            action_types[t] = action_types.get(t, 0) + 1
        for action_type, count in action_types.items():
            print(f"  {action_type}: {count}")
        
        # Specifically check for CAST_SPELL actions
        cast_spells = [a for a in legal3 if a.action_type == ActionType.CAST_SPELL]
        if cast_spells:
            print(f"\nCAST_SPELL actions ({len(cast_spells)}):")
            for action in cast_spells:
                card = next((c for c in game_state.cards if c.instance_id == action.card_instance_id), None)
                if card:
                    print(f"  - {card.name} ({card.mana_cost})")
        else:
            print("\nNo CAST_SPELL actions available")
            # Show why
            alice_hand = [c for c in game_state.cards if c.owner_id == alice_id and c.zone == Zone.HAND]
            print(f"  Creatures in hand: {[c.name for c in alice_hand if c.is_creature()]}")
