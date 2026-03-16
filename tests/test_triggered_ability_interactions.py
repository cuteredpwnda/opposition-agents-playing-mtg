"""
Integration tests for triggered abilities in full gameplay (Phase 3)

Tests ETB triggers interacting with:
- Multiple creatures with triggers entering in sequence
- Triggers improving hand size and game state
- Trigger effect persistence across resolutions
- Triggers with various card types
"""

import pytest
from src.engine.game_state import GameState, Phase, Zone, CardInstance, PlayerState, TriggerType, Trigger
from src.engine.rules_engine import RulesEngine
from src.engine.triggers import check_enters_battlefield_triggers, resolve_trigger


def create_game_with_library_support():
    """Create a game state with proper library for draw triggers."""
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
        PlayerState(player_id="Bob", name="Bob", life_total=20),
    ]
    
    cards = []
    
    # Add library cards for Alice
    for i in range(20):
        card = CardInstance(
            instance_id=f"alice_lib_{i}",
            card_data={
                "name": "Island",
                "type_line": "Basic Land — Island",
                "mana_cost": "",
                "cmc": 0,
                "oracle_text": "",
                "set": "UNH"
            },
            zone=Zone.LIBRARY,
            owner_id="Alice",
            controller_id="Alice"
        )
        cards.append(card)
    
    # Add library cards for Bob
    for i in range(20):
        card = CardInstance(
            instance_id=f"bob_lib_{i}",
            card_data={
                "name": "Mountain",
                "type_line": "Basic Land — Mountain",
                "mana_cost": "",
                "cmc": 0,
                "oracle_text": "",
                "set": "UNH"
            },
            zone=Zone.LIBRARY,
            owner_id="Bob",
            controller_id="Bob"
        )
        cards.append(card)
    
    game = GameState(players=players, cards=cards)
    return game


def test_etb_trigger_fires_during_full_game():
    """Test that ETB triggers actually fire with proper game state."""
    game = create_game_with_library_support()
    rules = RulesEngine()
    
    # Create Mulldrifter in hand for Alice
    mulldrifter = CardInstance(
        instance_id="mulldrifter_1",
        card_data={
            "name": "Mulldrifter",
            "type_line": "Creature — Elemental",
            "mana_cost": "{2}{U}",
            "cmc": 3,
            "power": "2",
            "toughness": "2",
            "oracle_text": "When Mulldrifter enters, draw a card.",
            "set": "MH1"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(mulldrifter)
    
    # Move to battlefield (simulate casting)
    mulldrifter.zone = Zone.BATTLEFIELD
    
    # Check for ETB triggers
    triggers = check_enters_battlefield_triggers(game, mulldrifter)
    
    # Should have 1 trigger
    assert len(triggers) == 1, f"Expected 1 trigger, got {len(triggers)}"
    
    # Track hand before trigger
    alice_hand_before = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    
    # Resolve the trigger
    resolve_trigger(game, triggers[0])
    
    # Track hand after trigger
    alice_hand_after = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    
    # Hand should have increased by 1 from draw
    assert alice_hand_after == alice_hand_before + 1, \
        f"Draw trigger should increase hand. Before: {alice_hand_before}, After: {alice_hand_after}"


def test_multiple_creatures_with_draw_triggers():
    """Test multiple creatures with draw triggers entering in sequence."""
    game = create_game_with_library_support()
    rules = RulesEngine()
    
    # Create multiple Mulldrifters
    mulldrifters = []
    for i in range(3):
        card = CardInstance(
            instance_id=f"mulldrifter_{i}",
            card_data={
                "name": "Mulldrifter",
                "type_line": "Creature — Elemental",
                "mana_cost": "{2}{U}",
                "cmc": 3,
                "power": "2",
                "toughness": "2",
                "oracle_text": "When Mulldrifter enters, draw a card.",
                "set": "MH1"
            },
            zone=Zone.HAND,
            owner_id="Alice",
            controller_id="Alice"
        )
        mulldrifters.append(card)
        game.cards.append(card)
    
    # Move all to battlefield
    for card in mulldrifters:
        card.zone = Zone.BATTLEFIELD
    
    # Collect all triggers
    all_triggers = []
    for card in mulldrifters:
        triggers = check_enters_battlefield_triggers(game, card)
        all_triggers.extend(triggers)
    
    # Should have 3 triggers
    assert len(all_triggers) == 3, f"Expected 3 triggers, got {len(all_triggers)}"
    
    # Resolve all triggers
    for trigger in all_triggers:
        resolve_trigger(game, trigger)
    
    # Check hand increased by 3
    alice_hand = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    assert alice_hand == 3, f"Expected 3 cards drawn, got {alice_hand}"


def test_trigger_hand_improvement():
    """Verify that ETB draw triggers improve hand size."""
    game = create_game_with_library_support()
    
    # Start with empty hand, full library
    alice_hand_initial = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    assert alice_hand_initial == 0, "Hand should start empty"
    
    alice_lib_initial = len([c for c in game.cards if c.zone == Zone.LIBRARY and c.controller_id == "Alice"])
    assert alice_lib_initial == 20, "Library should have 20 cards"
    
    # Create and resolve draw trigger
    trigger = Trigger(
        trigger_id="test_draw",
        source_card_id="mulldrifter",
        controller_id="Alice",
        trigger_type=TriggerType.ENTERS_BATTLEFIELD,
        description="When Mulldrifter enters, draw a card"
    )
    
    # Add trigger effect to game engine (using resolve_trigger)
    resolve_trigger(game, trigger)
    
    # Verify hand increased
    alice_hand_after_1 = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    assert alice_hand_after_1 == 1, f"Expected 1 card in hand after trigger, got {alice_hand_after_1}"
    
    # Resolve another trigger
    trigger2 = Trigger(
        trigger_id="test_draw_2",
        source_card_id="mulldrifter_2",
        controller_id="Alice",
        trigger_type=TriggerType.ENTERS_BATTLEFIELD,
        description="When Mulldrifter enters, draw a card"
    )
    resolve_trigger(game, trigger2)
    
    # Verify hand increased to 2
    alice_hand_after_2 = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    assert alice_hand_after_2 == 2, f"Expected 2 cards in hand after second trigger, got {alice_hand_after_2}"


def test_trigger_effect_persistence():
    """Test that trigger effects persist correctly across game state."""
    game = create_game_with_library_support()
    
    # Initial state
    alice_life_initial = next((p.life_total for p in game.players if p.player_id == "Alice"), 0)
    bob_life_initial = next((p.life_total for p in game.players if p.player_id == "Bob"), 0)
    alice_hand_initial = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    
    assert alice_life_initial == 20
    assert bob_life_initial == 20
    assert alice_hand_initial == 0
    
    # Test 1: Draw effect
    trigger_draw = Trigger(
        trigger_id="draw_trigger",
        source_card_id="card1",
        controller_id="Alice",
        trigger_type=TriggerType.ENTERS_BATTLEFIELD,
        description="Draw a card"
    )
    resolve_trigger(game, trigger_draw)
    
    alice_hand_after_draw = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    assert alice_hand_after_draw == alice_hand_initial + 1, "Draw should add 1 to hand"
    
    # Find the drawn card and verify it exists
    drawn_cards = [c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"]
    assert len(drawn_cards) == 1
    assert drawn_cards[0].name == "Island"


def test_trigger_with_no_library_cards():
    """Test draw trigger when library is empty (shouldn't crash)."""
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
    ]
    
    game = GameState(players=players, cards=[])
    
    # Create trigger with empty library
    trigger = Trigger(
        trigger_id="draw_empty",
        source_card_id="mulldrifter",
        controller_id="Alice",
        trigger_type=TriggerType.ENTERS_BATTLEFIELD,
        description="Draw a card"
    )
    
    # Should not crash even with empty library
    result = resolve_trigger(game, trigger)
    
    # Hand should still be empty if library was empty
    alice_hand = len([c for c in game.cards if c.zone == Zone.HAND and c.controller_id == "Alice"])
    assert alice_hand == 0, "Game shouldn't crash with empty library"


def test_creatures_with_no_triggers():
    """Test that vanilla creatures don't create triggers."""
    game = create_game_with_library_support()
    
    # Create vanilla creature (no oracle text)
    vanilla = CardInstance(
        instance_id="vanilla_creature",
        card_data={
            "name": "Grizzly Bears",
            "type_line": "Creature — Bear",
            "mana_cost": "{1}{G}",
            "cmc": 2,
            "power": "2",
            "toughness": "2",
            "oracle_text": "",  # No triggers
            "set": "UNH"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(vanilla)
    
    # Move to battlefield
    vanilla.zone = Zone.BATTLEFIELD
    
    # Check for triggers
    triggers = check_enters_battlefield_triggers(game, vanilla)
    
    # Should have no triggers
    assert len(triggers) == 0, f"Vanilla creature shouldn't have triggers, but got {len(triggers)}"

