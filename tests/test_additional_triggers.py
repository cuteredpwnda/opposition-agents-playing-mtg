"""
Tests for additional trigger types beyond ETB

Tests for:
- ATTACKS triggers (creature attacks)
- CREATURE_DIES triggers (creature dies)
- CAST triggers (spell cast)
"""

import pytest
from src.engine.game_state import GameState, Phase, Zone, CardInstance, PlayerState, TriggerType, Trigger
from src.engine.rules_engine import RulesEngine
from src.engine.triggers import (
    parse_triggers, 
    check_attack_triggers,
    check_death_triggers,
    check_cast_triggers,
    resolve_trigger
)


def create_game_with_full_support():
    """Create game with proper library and card support."""
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
        PlayerState(player_id="Bob", name="Bob", life_total=20),
    ]
    
    cards = []
    
    # Add library cards for both players
    for player_id, name in [("Alice", "Island"), ("Bob", "Mountain")]:
        for i in range(20):
            card = CardInstance(
                instance_id=f"{player_id}_lib_{i}",
                card_data={
                    "name": name,
                    "type_line": "Basic Land",
                    "mana_cost": "",
                    "cmc": 0,
                    "oracle_text": "",
                    "set": "UNH"
                },
                zone=Zone.LIBRARY,
                owner_id=player_id,
                controller_id=player_id
            )
            cards.append(card)
    
    game = GameState(players=players, cards=cards)
    return game


def test_parse_attack_triggers():
    """Test parsing of attack triggers from oracle text."""
    card_with_attack = CardInstance(
        instance_id="goblin_guide",
        card_data={
            "name": "Goblin Guide",
            "type_line": "Creature — Goblin Scout",
            "mana_cost": "{R}",
            "cmc": 1,
            "power": "2",
            "toughness": "2",
            "oracle_text": "Whenever Goblin Guide attacks, defending player reveals the top card of their library.",
            "set": "ZEN"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(card_with_attack)
    attack_triggers = [t for t in triggers if t.trigger_type == TriggerType.ATTACKS]
    
    assert len(attack_triggers) == 1, f"Expected 1 attack trigger, got {len(attack_triggers)}"
    assert "defend" in attack_triggers[0].description.lower()


def test_parse_death_triggers():
    """Test parsing of death triggers from oracle text."""
    card_with_death = CardInstance(
        instance_id="young_pyre",
        card_data={
            "name": "Young Pyromancer",
            "type_line": "Creature — Human Shaman",
            "mana_cost": "{1}{R}",
            "cmc": 2,
            "power": "2",
            "toughness": "1",
            "oracle_text": "When Young Pyromancer dies, create a 1/1 red Elemental creature token.",
            "set": "UMA"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(card_with_death)
    death_triggers = [t for t in triggers if t.trigger_type == TriggerType.CREATURE_DIES]
    
    assert len(death_triggers) == 1, f"Expected 1 death trigger, got {len(death_triggers)}"
    assert "create" in death_triggers[0].description.lower()


def test_parse_cast_triggers():
    """Test parsing of cast triggers from oracle text."""
    card_with_cast = CardInstance(
        instance_id="talrand",
        card_data={
            "name": "Talrand, Sky Summoner",
            "type_line": "Legendary Creature — Human Wizard",
            "mana_cost": "{1}{U}{U}",
            "cmc": 3,
            "power": "2",
            "toughness": "2",
            "oracle_text": "Whenever you cast an instant or sorcery spell, create a 2/2 blue Drake creature token with flying.",
            "set": "M21"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(card_with_cast)
    cast_triggers = [t for t in triggers if t.trigger_type == TriggerType.CAST]
    
    assert len(cast_triggers) == 1, f"Expected 1 cast trigger, got {len(cast_triggers)}"
    assert "create" in cast_triggers[0].description.lower()


def test_check_attack_triggers():
    """Test detecting attack triggers when creature attacks."""
    game = create_game_with_full_support()
    
    # Create a creature with attack trigger
    attacker = CardInstance(
        instance_id="anax",
        card_data={
            "name": "Anax, Hardened in the Forge",
            "type_line": "Creature — Satyr",
            "mana_cost": "{1}{R}{R}",
            "cmc": 3,
            "power": "3",
            "toughness": "2",
            "oracle_text": "Whenever Anax attacks, you gain 1 life.",
            "set": "THB"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(attacker)
    
    # Check for attack triggers
    triggers = check_attack_triggers(game, attacker)
    
    assert len(triggers) >= 1, f"Expected at least 1 trigger, got {len(triggers)}"
    assert any(t.trigger_type == TriggerType.ATTACKS for t in triggers)


def test_check_death_triggers():
    """Test detecting death triggers when creature dies."""
    game = create_game_with_full_support()
    
    # Create a creature with death trigger
    creature = CardInstance(
        instance_id="zulaport",
        card_data={
            "name": "Zulaport Cutthroat",
            "type_line": "Creature — Human Rogue Ally",
            "mana_cost": "{1}{B}",
            "cmc": 2,
            "power": "1",
            "toughness": "1",
            "oracle_text": "Whenever Zulaport Cutthroat or another creature you control dies, each opponent loses 1 life.",
            "set": "BFZ"
        },
        zone=Zone.GRAVEYARD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    # Check for death triggers
    triggers = check_death_triggers(game, creature)
    
    assert len(triggers) >= 1, f"Expected at least 1 trigger, got {len(triggers)}"
    assert any(t.trigger_type == TriggerType.CREATURE_DIES for t in triggers)


def test_check_cast_triggers():
    """Test detecting cast triggers when spell cast."""
    game = create_game_with_full_support()
    
    # Create a permanent with cast trigger
    permanent = CardInstance(
        instance_id="monastery",
        card_data={
            "name": "Monastery Mentor",
            "type_line": "Creature — Human Monk",
            "mana_cost": "{1}{W}",
            "cmc": 2,
            "power": "2",
            "toughness": "2",
            "oracle_text": "Whenever you cast a spell, create a 1/1 white Monk creature token.",
            "set": "FRF"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(permanent)
    
    # Create a spell being cast
    spell = CardInstance(
        instance_id="lightning_bolt",
        card_data={
            "name": "Lightning Bolt",
            "type_line": "Instant",
            "mana_cost": "{R}",
            "cmc": 1,
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "set": "A25"
        },
        zone=Zone.STACK,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    # Check for cast triggers
    triggers = check_cast_triggers(game, "Alice", spell)
    
    assert len(triggers) >= 1, f"Expected at least 1 trigger, got {len(triggers)}"
    assert any(t.trigger_type == TriggerType.CAST for t in triggers)


def test_resolve_attack_trigger():
    """Test resolving attack trigger effect."""
    game = create_game_with_full_support()
    
    alice = next((p for p in game.players if p.player_id == "Alice"), None)
    alice_initial_life = alice.life_total
    
    # Create attack trigger
    trigger = Trigger(
        trigger_id="attack_test",
        source_card_id="attacker",
        controller_id="Alice",
        trigger_type=TriggerType.ATTACKS,
        description="you gain 1 life"
    )
    
    # Resolve trigger
    resolve_trigger(game, trigger)
    
    # Verify life increased
    assert alice.life_total == alice_initial_life + 1, \
        f"Attack trigger should gain 1 life. Before: {alice_initial_life}, After: {alice.life_total}"


def test_resolve_death_trigger():
    """Test resolving death trigger effect."""
    game = create_game_with_full_support()
    
    bob = next((p for p in game.players if p.player_id == "Bob"), None)
    bob_initial_life = bob.life_total
    
    # Create death trigger
    trigger = Trigger(
        trigger_id="death_test",
        source_card_id="dying_creature",
        controller_id="Alice",
        trigger_type=TriggerType.CREATURE_DIES,
        description="each opponent loses 1 life"
    )
    
    # Resolve trigger
    resolve_trigger(game, trigger)
    
    # Verify opponent lost life
    assert bob.life_total == bob_initial_life - 1, \
        f"Death trigger should deal 1 damage. Before: {bob_initial_life}, After: {bob.life_total}"


def test_resolve_cast_trigger():
    """Test resolving cast trigger effect."""
    game = create_game_with_full_support()
    
    alice = next((p for p in game.players if p.player_id == "Alice"), None)
    alice_hand_before = len([c for c in game.cards if c.zone == Zone.HAND and c.owner_id == "Alice"])
    
    # Create cast trigger (draw effect)
    trigger = Trigger(
        trigger_id="cast_test",
        source_card_id="caster",
        controller_id="Alice",
        trigger_type=TriggerType.CAST,
        description="you draw a card"
    )
    
    # Resolve trigger
    resolve_trigger(game, trigger)
    
    # Verify hand increased
    alice_hand_after = len([c for c in game.cards if c.zone == Zone.HAND and c.owner_id == "Alice"])
    assert alice_hand_after == alice_hand_before + 1, \
        f"Cast trigger should draw 1 card. Before: {alice_hand_before}, After: {alice_hand_after}"


def test_multiple_trigger_types():
    """Test that creature can have multiple trigger types."""
    creature = CardInstance(
        instance_id="complex",
        card_data={
            "name": "Complex Creature",
            "type_line": "Creature — test",
            "mana_cost": "{3}",
            "cmc": 3,
            "power": "2",
            "toughness": "2",
            "oracle_text": """When Complex Creature enters, draw a card. 
                              Whenever Complex Creature attacks, you gain 1 life. 
                              When Complex Creature dies, each opponent loses 1 life.""",
            "set": "TST"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(creature)
    
    trigger_types = {t.trigger_type for t in triggers}
    assert TriggerType.ENTERS_BATTLEFIELD in trigger_types
    assert TriggerType.ATTACKS in trigger_types
    assert TriggerType.CREATURE_DIES in trigger_types
    
    assert len(triggers) >= 3, f"Expected at least 3 triggers, got {len(triggers)}"


def test_trigger_without_colon():
    """Test triggers that might have slightly different formatting."""
    card = CardInstance(
        instance_id="test_card",
        card_data={
            "name": "Test Card",
            "type_line": "Creature — test",
            "mana_cost": "{2}",
            "cmc": 2,
            "power": "1",
            "toughness": "1",
            "oracle_text": "Whenever this creature attacks, defending player loses 1 life.",
            "set": "TST"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(card)
    attack_triggers = [t for t in triggers if t.trigger_type == TriggerType.ATTACKS]
    
    # Should detect attack trigger with proper formatting
    assert len(attack_triggers) >= 1, "Should detect attack trigger with proper oracle formatting"


def test_card_without_triggers():
    """Test que cards without triggers don't create spurious triggers."""
    vanilla = CardInstance(
        instance_id="vanilla",
        card_data={
            "name": "Vanilla Creature",
            "type_line": "Creature — test",
            "mana_cost": "{2}{G}",
            "cmc": 3,
            "power": "3",
            "toughness": "3",
            "oracle_text": "",
            "set": "TST"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(vanilla)
    
    assert len(triggers) == 0, f"Vanilla card should have no triggers, but got {len(triggers)}"


def test_multiple_damage_values():
    """Test triggers with different damage/life values."""
    test_cases = [
        ("deal 1 damage", 1),
        ("deal 2 damage", 2),
        ("gain 3 life", 3),
        ("gain 5 life", 5),
    ]
    
    game = create_game_with_full_support()
    bob = next((p for p in game.players if p.player_id == "Bob"), None)
    
    for description, expected_value in test_cases:
        bob.life_total = 20  # Reset
        
        trigger = Trigger(
            trigger_id="test",
            source_card_id="card",
            controller_id="Alice",
            trigger_type=TriggerType.ATTACKS,
            description=description
        )
        
        resolve_trigger(game, trigger)
        
        if "gain" in description:
            alice = next((p for p in game.players if p.player_id == "Alice"), None)
            assert alice.life_total == 20 + expected_value, \
                f"Trigger '{description}' should gain {expected_value} life"
            alice.life_total = 20  # Reset
        else:
            assert bob.life_total == 20 - expected_value, \
                f"Trigger '{description}' should deal {expected_value} damage"


def test_trigger_parsing_case_insensitive():
    """Test that trigger parsing is case-insensitive."""
    card = CardInstance(
        instance_id="test",
        card_data={
            "name": "Test",
            "type_line": "Creature",
            "mana_cost": "{1}",
            "cmc": 1,
            "power": "1",
            "toughness": "1",
            "oracle_text": "WHENEVER THIS CREATURE ATTACKS, YOU GAIN 1 LIFE.",
            "set": "TST"
        },
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(card)
    
    # Should detect attack trigger even with uppercase text
    attack_triggers = [t for t in triggers if t.trigger_type == TriggerType.ATTACKS]
    assert len(attack_triggers) >= 1, "Should detect attack trigger in uppercase text"
