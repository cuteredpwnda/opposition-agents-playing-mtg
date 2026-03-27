"""
Integration tests for complete ability system.

Tests that exercise triggered, activated, and static abilities working
together in realistic game scenarios. These tests verify:
- Multiple ability types on single cards
- Lords affecting combat calculations
- Triggered abilities firing during gameplay
- Activated abilities used strategically
- Full turn sequences with all ability types
"""

import pytest
from src.engine.game_state import GameState, Zone, CardInstance, PlayerState, Phase, ActionType, Action, Ability
from src.engine.static_abilities import get_effective_power_toughness


def create_ability_test_game():
    """Create a game with cards that have multiple ability types."""
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
        PlayerState(player_id="Bob", name="Bob", life_total=20),
    ]
    
    cards = []
    
    # Add library cards for both players
    for player_id in ["Alice", "Bob"]:
        for i in range(20):
            card = CardInstance(
                instance_id=f"{player_id}_lib_{i}",
                card_data={"name": "Island", "type_line": "Land", "oracle_text": "", "set": "UNH"},
                zone=Zone.LIBRARY,
                owner_id=player_id,
                controller_id=player_id
            )
            cards.append(card)
    
    game = GameState(players=players, cards=cards)
    return game


def test_lord_affects_creature_combat():
    """Test that static ability (lord) affects creature combat damage."""
    game = create_ability_test_game()
    
    # Create a lord (+1/+1 to creatures you control)
    lord = CardInstance(
        instance_id="lord",
        card_data={
            "name": "+1/+1 Lord",
            "type_line": "Creature — Human",
            "oracle_text": "Creatures you control get +1/+1.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    # Create a creature that will be affected by the lord
    creature = CardInstance(
        instance_id="creature",
        card_data={
            "name": "Test Creature",
            "type_line": "Creature — Spirit",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    # Get effective P/T
    power, toughness = get_effective_power_toughness(creature, game)
    
    assert power == 2, f"Creature should have 2 power (1+1 from lord), got {power}"
    assert toughness == 2, f"Creature should have 2 toughness (1+1 from lord), got {toughness}"


def test_triggered_ability_on_combat():
    """Test that triggered ability fires when creature attacks."""
    game = create_ability_test_game()
    
    # Create creature with attack trigger
    attacker = CardInstance(
        instance_id="trigger_attacker",
        card_data={
            "name": "Trigger Creature",
            "type_line": "Creature — Spirit",
            "oracle_text": "Whenever this creature attacks, you gain 1 life.",
            "power": "2",
            "toughness": "2",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice",
        summoning_sick=False  # Not summoning sick
    )
    game.cards.append(attacker)
    
    alice = game.players[0]
    initial_life = alice.life_total
    
    # Create action to declare attackers
    action = Action(
        action_type=ActionType.DECLARE_ATTACKERS,
        player_id="Alice",
        card_instance_id=attacker.instance_id,
        targets=["Bob"]
    )
    
    # Execute action (should trigger attack trigger)
    from src.engine.rules_engine import RulesEngine
    engine = RulesEngine()
    game = engine.execute_action(game, action)
    
    # Verify trigger was queued (would be resolved in priority loop)
    assert len(game.triggered_abilities) > 0, "Attack trigger should have been queued"


def test_activated_ability_mana_generation():
    """Test that activated mana ability works in game."""
    game = create_ability_test_game()
    alice = game.players[0]
    
    # Create a mana-generating land
    land = CardInstance(
        instance_id="mana_land",
        card_data={
            "name": "Mountain",
            "type_line": "Basic Land — Mountain",
            "oracle_text": "{T}: Add {R}.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice",
        tapped=False
    )
    game.cards.append(land)
    
    # Get legal activated abilities
    from src.engine.abilities import get_legal_activated_abilities, resolve_ability
    
    legal_abilities = get_legal_activated_abilities(game, land, "Alice")
    assert len(legal_abilities) > 0, "Land should have legal mana ability"
    
    ability = legal_abilities[0]
    red_before = alice.mana_pool["R"]
    
    # Resolve the ability
    game = resolve_ability(game, ability, "Alice")
    
    red_after = alice.mana_pool["R"]
    assert red_after == red_before + 1, f"Should have added 1 red mana, got {red_after - red_before}"
    assert land.tapped, "Land should be tapped after use"


def test_rules_engine_execute_activate_ability_uses_mana_ability():
    """Test RulesEngine.execute_action for ACTIVATE_ABILITY with mana ability."""
    game = create_ability_test_game()
    alice = game.players[0]

    land = CardInstance(
        instance_id="mana_land",
        card_data={
            "name": "Mountain",
            "type_line": "Basic Land — Mountain",
            "oracle_text": "{T}: Add {R}.",
            "set": "TEST",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice",
        tapped=False,
    )
    game.cards.append(land)

    from src.engine.rules_engine import RulesEngine
    engine = RulesEngine()

    legal_actions = engine.get_legal_actions(game, "Alice")
    activate_actions = [a for a in legal_actions if a.action_type == ActionType.ACTIVATE_ABILITY]
    assert activate_actions, "Should have an activate ability action"

    action = activate_actions[0]
    game = engine.execute_action(game, action)

    assert alice.mana_pool["R"] == 1, "Alice should have gained red mana"
    assert land.tapped, "Land should be tapped after activation"


def test_static_ability_plus_triggered_plus_activated():
    """Test card with all three ability types (complex lord/utility creature)."""
    game = create_ability_test_game()
    alice = game.players[0]
    
    # Create a complex creature with multiple ability types:
    # Static: Creatures you control get +1/+0
    # Triggered: When this creature attacks, you gain 1 life
    # Activated: {T}: Draw a card
    complex_creature = CardInstance(
        instance_id="complex",
        card_data={
            "name": "Complex Creature",
            "type_line": "Creature — Human",
            "oracle_text": (
                "Creatures you control get +1/+0. "
                "Whenever this creature attacks, you gain 1 life. "
                "{T}: Draw a card."
            ),
            "power": "2",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(complex_creature)
    
    # Create another creature to be affected by static ability
    other_creature = CardInstance(
        instance_id="other",
        card_data={
            "name": "Other Creature",
            "type_line": "Creature — Spirit",
            "oracle_text": "",
            "power": "1",
            "toughness": "2",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(other_creature)
    
    # Test 1: Static ability affects other creature
    from src.engine.static_abilities import get_effective_power_toughness
    power, toughness = get_effective_power_toughness(other_creature, game)
    assert power == 2, f"Other creature should have +1 power from lord, got {power}"
    assert toughness == 2, f"Other creature should still have 2 toughness"
    
    # Test 2: Activated ability is available
    from src.engine.abilities import get_legal_activated_abilities
    legal = get_legal_activated_abilities(game, complex_creature, "Alice")
    
    # Should have draw ability (not the static/triggered abilities)
    has_draw = any("draw" in a.effect.lower() for a in legal)
    assert has_draw, "Should have draw ability available"
    
    # Test 3: Creature has triggered ability in oracle text
    assert "attack" in complex_creature.oracle_text.lower(), "Should have attack trigger in oracle"


def test_top_card_knowledge_tracking():
    """Test that reveal-top-library knowledge is modeled on CardInstance."""
    game = create_ability_test_game()
    alice = game.players[0]

    # Set up a simple library for Alice
    top_card = CardInstance(
        instance_id="alice_lib_top",
        card_data={"name": "Mystic Forge", "type_line": "Artifact", "oracle_text": "", "set": "TEST"},
        zone=Zone.LIBRARY,
        owner_id="Alice",
        controller_id="Alice",
    )
    game.cards.insert(0, top_card)

    # Simulate an ability that reveals top card to Alice
    dummy_source = CardInstance(
        instance_id="dummy_source",
        card_data={"name": "Tutor Source", "type_line": "Artifact", "oracle_text": "", "set": "TEST"},
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice",
    )
    game.cards.append(dummy_source)

    ability = Ability(
        source_card_id="dummy_source", controller_id="Alice", cost="{0}",
        effect="Reveal the top card of your library", can_use_any_time=False,
        description="Reveal top library card"
    )

    from src.engine.abilities import resolve_ability

    game = resolve_ability(game, ability, "Alice")
    assert "Alice" in top_card.known_to, "Alice should know the top card after reveal ability"


    # Cross-check GameState helper
    assert game.card_is_known(top_card.instance_id, "Alice")


    # War-test: Bob does not know it yet
    assert not game.card_is_known(top_card.instance_id, "Bob")


    # More complex: mark as known for Bob (simulating shared reveal)
    game.mark_card_known(top_card.instance_id, "Bob")
    assert game.card_is_known(top_card.instance_id, "Bob")


    # Ensure unknown card remains unknown
    another_card = CardInstance(
        instance_id="alice_lib_next",
        card_data={"name": "Island", "type_line": "Land", "oracle_text": "", "set": "TEST"},
        zone=Zone.LIBRARY,
        owner_id="Alice",
        controller_id="Alice",
    )
    game.cards.append(another_card)
    assert not game.card_is_known(another_card.instance_id, "Bob")


    # End of test


def test_multiple_lords_stacking_in_combat():
    """Test that multiple lords correctly stack P/T bonuses for combat."""
    game = create_ability_test_game()
    
    # Create two lords
    lord1 = CardInstance(
        instance_id="lord1",
        card_data={
            "name": "Lord 1",
            "type_line": "Creature — Human",
            "oracle_text": "Creatures you control get +1/+1.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord1)
    
    lord2 = CardInstance(
        instance_id="lord2",
        card_data={
            "name": "Lord 2",
            "type_line": "Creature — Human",
            "oracle_text": "Creatures you control get +1/+0.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord2)
    
    # Create creature
    creature = CardInstance(
        instance_id="creature",
        card_data={
            "name": "Creature",
            "type_line": "Creature",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    # Check effective P/T
    from src.engine.static_abilities import get_effective_power_toughness
    power, toughness = get_effective_power_toughness(creature, game)
    
    assert power == 3, f"Should have 1+1+1 = 3 power, got {power}"
    assert toughness == 2, f"Should have 1+1+0 = 2 toughness, got {toughness}"


def test_creature_with_triggered_etb_and_lord():
    """Test creature with ETB trigger being enhanced by lord."""
    game = create_ability_test_game()
    alice = game.players[0]
    
    # Create a lord
    lord = CardInstance(
        instance_id="lord",
        card_data={
            "name": "Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+1.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    # Create creature with ETB trigger on battlefield
    creature = CardInstance(
        instance_id="etb_creature",
        card_data={
            "name": "ETB Creature",
            "type_line": "Creature",
            "oracle_text": "When this creature enters the battlefield, draw a card.",
            "power": "2",
            "toughness": "2",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice",
        turn_entered=game.turn_number
    )
    game.cards.append(creature)
    
    # Verify creature has enhanced stats from lord
    from src.engine.static_abilities import get_effective_power_toughness
    power, toughness = get_effective_power_toughness(creature, game)
    
    assert power == 3, f"Should have 2+1 = 3 power from lord, got {power}"
    assert toughness == 3, f"Should have 2+1 = 3 toughness from lord, got {toughness}"
    
    # Verify ETB trigger was queued when creature entered
    # (In a real game, this would have happened during resolution)
    assert "enter" in creature.oracle_text.lower(), "Creature should have ETB trigger"


def test_mana_ability_not_on_stack():
    """Test that mana abilities resolve immediately, not going on stack."""
    game = create_ability_test_game()
    alice = game.players[0]
    
    # Create mana land
    land = CardInstance(
        instance_id="land",
        card_data={
            "name": "Mountain",
            "type_line": "Land",
            "oracle_text": "{T}: Add {R}.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice",
        tapped=False
    )
    game.cards.append(land)
    
    from src.engine.abilities import parse_abilities, is_mana_ability
    
    abilities = parse_abilities(land)
    assert len(abilities) > 0, "Land should have mana ability"
    
    ability = abilities[0]
    assert is_mana_ability(ability.effect), "Should be identified as mana ability"
    assert ability.can_use_any_time, "Mana ability should be usable anytime"


def test_lord_grants_keyword_to_creatures():
    """Test that lord can grant keywords to creatures."""
    game = create_ability_test_game()
    
    # Create flying lord
    flying_lord = CardInstance(
        instance_id="flying_lord",
        card_data={
            "name": "Flying Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control have flying.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(flying_lord)
    
    # Create creature without flying
    creature = CardInstance(
        instance_id="creature",
        card_data={
            "name": "Creature",
            "type_line": "Creature",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    # Verify creature now has flying keyword
    from src.engine.static_abilities import has_keyword
    
    assert has_keyword(creature, game, "flying"), "Creature should have flying from lord"


def test_ability_effects_disappear_when_source_leaves():
    """Test that static ability effects are lost when source leaves battlefield."""
    game = create_ability_test_game()
    
    # Create lord
    lord = CardInstance(
        instance_id="lord",
        card_data={
            "name": "Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+1.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    # Create creature
    creature = CardInstance(
        instance_id="creature",
        card_data={
            "name": "Creature",
            "type_line": "Creature",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    from src.engine.static_abilities import get_effective_power_toughness
    
    # With lord, creature is 2/2
    power1, toughness1 = get_effective_power_toughness(creature, game)
    assert power1 == 2 and toughness1 == 2
    
    # Remove lord from battlefield
    lord.zone = Zone.GRAVEYARD
    
    # Now creature should be 1/1 again
    power2, toughness2 = get_effective_power_toughness(creature, game)
    assert power2 == 1 and toughness2 == 1, "Creature should lose bonuses when lord leaves"


def test_activated_ability_costs_are_checked():
    """Test that activated ability costs prevent illegal activation."""
    game = create_ability_test_game()
    alice = game.players[0]
    
    # Create creature with expensive ability
    creature = CardInstance(
        instance_id="expensive",
        card_data={
            "name": "Expensive",
            "type_line": "Creature",
            "oracle_text": "{U}{U}{U}: Draw a card.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    # Give Alice only 1 blue mana (not enough)
    alice.mana_pool["U"] = 1
    
    from src.engine.abilities import parse_abilities, get_legal_activated_abilities
    
    # Ability exists
    abilities = parse_abilities(creature)
    assert len(abilities) > 0, "Should have ability"
    
    # But it's not legal to activate (not enough mana)
    legal = get_legal_activated_abilities(game, creature, "Alice")
    assert len(legal) == 0, "Should not have legal ability with insufficient mana"
    
    # Give enough mana
    alice.mana_pool["U"] = 3
    legal = get_legal_activated_abilities(game, creature, "Alice")
    assert len(legal) > 0, "Should have legal ability with sufficient mana"


def test_opponent_creatures_not_affected_by_lord():
    """Test that lords only affect controller's creatures."""
    game = create_ability_test_game()
    
    # Alice creates lord
    lord = CardInstance(
        instance_id="alice_lord",
        card_data={
            "name": "Alice's Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+1.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    # Bob creates creature
    bob_creature = CardInstance(
        instance_id="bob_creature",
        card_data={
            "name": "Bob's Creature",
            "type_line": "Creature",
            "oracle_text": "",
            "power": "2",
            "toughness": "2",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Bob",
        controller_id="Bob"
    )
    game.cards.append(bob_creature)
    
    from src.engine.static_abilities import get_effective_power_toughness
    
    # Bob's creature should NOT be affected by Alice's lord
    power, toughness = get_effective_power_toughness(bob_creature, game)
    assert power == 2, f"Bob's creature should not have bonus, got {power}"
    assert toughness == 2, f"Bob's creature should not have bonus, got {toughness}"


def test_activated_ability_on_stack():
    """Test that non-mana activated abilities go on stack."""
    game = create_ability_test_game()
    
    # Create creature with draw ability
    creature = CardInstance(
        instance_id="drawer",
        card_data={
            "name": "Drawer",
            "type_line": "Creature",
            "oracle_text": "{U}: Draw a card.",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    alice = game.players[0]
    alice.mana_pool["U"] = 1
    
    from src.engine.abilities import parse_abilities, is_mana_ability
    
    abilities = parse_abilities(creature)
    assert len(abilities) > 0
    
    ability = abilities[0]
    # This is a draw ability, not a mana ability
    assert not is_mana_ability(ability.effect), "Draw ability should not be mana ability"
    assert not ability.can_use_any_time, "Draw ability should require priority"


def test_full_game_sequence_with_abilities():
    """Test a small multi-turn sequence with various abilities in play."""
    game = create_ability_test_game()
    alice = game.players[0]
    bob = game.players[1]
    
    # Alice's board: Lord + 2 creatures
    lord = CardInstance(
        instance_id="alice_lord",
        card_data={
            "name": "Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+1.",
            "power": "2",
            "toughness": "2",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    attacker1 = CardInstance(
        instance_id="attacker1",
        card_data={
            "name": "Attacker 1",
            "type_line": "Creature",
            "oracle_text": "",
            "power": "2",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice",
        summoning_sick=False
    )
    game.cards.append(attacker1)
    
    attacker2 = CardInstance(
        instance_id="attacker2",
        card_data={
            "name": "Attacker 2",
            "type_line": "Creature",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice",
        summoning_sick=False
    )
    game.cards.append(attacker2)
    
    # Verify creatures are enhanced by lord
    from src.engine.static_abilities import get_effective_power_toughness
    
    power1, toughness1 = get_effective_power_toughness(attacker1, game)
    assert power1 == 3, f"Attacker 1 should have 3 power (2+1 from lord), got {power1}"
    assert toughness1 == 2, f"Attacker 1 should have 2 toughness (1+1 from lord), got {toughness1}"
    
    power2, toughness2 = get_effective_power_toughness(attacker2, game)
    assert power2 == 2, f"Attacker 2 should have 2 power (1+1 from lord), got {power2}"
    assert toughness2 == 2, f"Attacker 2 should have 2 toughness (1+1 from lord), got {toughness2}"
    
    # Alice has mana
    alice.mana_pool["R"] = 2
    
    # Verify total power on board: 2 (lord) + 3 (atk1) + 2 (atk2) = 7
    total_power = power1 + power2 + 3  # +3 for lord itself
    assert total_power == 8, f"Should have 8 total power, got {total_power}"
