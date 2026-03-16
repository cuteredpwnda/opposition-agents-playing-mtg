"""
Tests for static abilities — continuous game state modifying effects

Tests for:
- Parsing static abilities from oracle text
- Detecting scope (creatures you control, all creatures, etc.)
- Applying power/toughness bonuses
- Applying keywords
- Querying effective stats and keywords
"""

import pytest
from src.engine.game_state import GameState, Zone, CardInstance, PlayerState, StaticAbility
from src.engine.static_abilities import (
    parse_static_abilities,
    get_static_abilities_affecting,
    get_effective_power_toughness,
    has_keyword,
    _parse_scope,
)


def create_game_with_lords():
    """Create game with lords and creatures to test static effects."""
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


def test_parse_power_toughness_bonus():
    """Test parsing +X/+Y bonuses from oracle text."""
    card = CardInstance(
        instance_id="lord_1",
        card_data={
            "name": "Intrepid Hero",
            "type_line": "Creature — Human",
            "oracle_text": "Creatures you control get +1/+1.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    abilities = parse_static_abilities(card)
    
    assert len(abilities) >= 1, f"Expected at least 1 static ability, got {len(abilities)}"
    lord_ability = abilities[0]
    assert lord_ability.effect_type == "power_toughness"
    assert lord_ability.power_mod == 1
    assert lord_ability.toughness_mod == 1
    assert lord_ability.scope == "creatures_you_control"


def test_parse_keyword_ability():
    """Test parsing keyword grant abilities."""
    card = CardInstance(
        instance_id="flying_lord",
        card_data={
            "name": "Watcher of the Spheres",
            "type_line": "Creature",
            "oracle_text": "Creatures you control have flying.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    abilities = parse_static_abilities(card)
    
    assert len(abilities) >= 1, f"Expected at least 1 ability, got {len(abilities)}"
    flying_ability = abilities[0]
    assert flying_ability.effect_type == "keyword"
    assert "flying" in [k.lower() for k in flying_ability.keywords]
    assert flying_ability.scope == "creatures_you_control"


def test_parse_all_creatures_effect():
    """Test parsing affects all creatures (not just yours)."""
    card = CardInstance(
        instance_id="glare",
        card_data={
            "name": "Glare of Subdual",
            "type_line": "Enchantment",
            "oracle_text": "All creatures get -1/-1.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    abilities = parse_static_abilities(card)
    
    assert len(abilities) >= 1
    ability = abilities[0]
    assert ability.scope == "all_creatures"
    assert ability.power_mod == -1
    assert ability.toughness_mod == -1


def test_parse_scope_variations():
    """Test that different scope texts are parsed correctly."""
    test_cases = [
        ("creatures you control", "creatures_you_control"),
        ("artifacts you control", "artifacts_you_control"),
        ("permanents you control", "permanents_you_control"),
        ("all creatures", "all_creatures"),
        ("all artifacts", "all_artifacts"),
    ]
    
    for scope_text, expected_scope in test_cases:
        result = _parse_scope(scope_text)
        assert result == expected_scope, f"Expected {expected_scope} from '{scope_text}', got {result}"


def test_get_static_abilities_affecting_own_creatures():
    """Test that lords affect the controller's creatures."""
    game = create_game_with_lords()
    
    # Create a lord
    lord = CardInstance(
        instance_id="lord",
        card_data={
            "name": "+1/+1 Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+1.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    # Create Alice's creature
    creature = CardInstance(
        instance_id="creature_1",
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
    
    # Get abilities affecting the creature
    affecting = get_static_abilities_affecting(game, creature)
    
    assert len(affecting) >= 1, f"Expected lord to affect creature, got {len(affecting)} abilities"
    assert affecting[0].source_card_id == lord.instance_id


def test_get_static_abilities_not_affecting_opponent_creatures():
    """Test that lords don't affect opponent's creatures."""
    game = create_game_with_lords()
    
    # Create Alice's lord
    lord = CardInstance(
        instance_id="lord",
        card_data={
            "name": "+1/+1 Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+1.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    # Create Bob's creature
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
    
    # Get abilities affecting Bob's creature
    affecting = get_static_abilities_affecting(game, bob_creature)
    
    assert len(affecting) == 0, f"Alice's lord should not affect Bob's creature, got {len(affecting)}"


def test_get_effective_power_toughness():
    """Test calculating effective P/T with static ability modifiers."""
    game = create_game_with_lords()
    
    # Create lord
    lord = CardInstance(
        instance_id="lord",
        card_data={
            "name": "+1/+2 Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+2.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    # Create creature with 2/3 base stats
    creature = CardInstance(
        instance_id="creature",
        card_data={
            "name": "Creature",
            "type_line": "Creature",
            "oracle_text": "",
            "power": "2",
            "toughness": "3",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    # Get effective P/T
    power, toughness = get_effective_power_toughness(creature, game)
    
    assert power == 3, f"Expected power 3 (2+1), got {power}"
    assert toughness == 5, f"Expected toughness 5 (3+2), got {toughness}"


def test_get_effective_power_toughness_multiple_lords():
    """Test effective P/T with multiple lords."""
    game = create_game_with_lords()
    
    # Create two lords
    lord1 = CardInstance(
        instance_id="lord1",
        card_data={
            "name": "+1/+1 Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+1.",
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
            "name": "+0/+2 Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +0/+2.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord2)
    
    # Create creature with 1/1
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
    
    # Get effective P/T
    power, toughness = get_effective_power_toughness(creature, game)
    
    assert power == 2, f"Expected power 2 (1+1+0), got {power}"
    assert toughness == 4, f"Expected toughness 4 (1+1+2), got {toughness}"


def test_has_keyword_native():
    """Test detecting native keywords on cards."""
    creature = CardInstance(
        instance_id="flyer",
        card_data={
            "name": "Flying Creature",
            "type_line": "Creature",
            "oracle_text": "Flying.",
            "power": "2",
            "toughness": "2",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game = create_game_with_lords()
    game.cards.append(creature)
    
    # Should have flying intrinsically
    assert has_keyword(creature, game, "flying")


def test_has_keyword_from_lord():
    """Test detecting keywords granted by static abilities."""
    game = create_game_with_lords()
    
    # Create flying lord
    flying_lord = CardInstance(
        instance_id="flying_lord",
        card_data={
            "name": "Flying Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control have flying.",
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
    
    # Should have flying from lord's static ability
    assert has_keyword(creature, game, "flying")


def test_static_ability_does_not_affect_other_zones():
    """Test that static abilities only affect permanent on battlefield."""
    game = create_game_with_lords()
    
    # Create lord on battlefield
    lord = CardInstance(
        instance_id="lord",
        card_data={
            "name": "+1/+1 Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control get +1/+1.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(lord)
    
    # Create creature in hand
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
        zone=Zone.HAND,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(creature)
    
    # Should not be affected by lord (not on battlefield)
    affecting = get_static_abilities_affecting(game, creature)
    
    assert len(affecting) == 0, f"Lord should not affect creature in hand"


def test_parse_negative_modifiers():
    """Test parsing negative P/T modifiers."""
    card = CardInstance(
        instance_id="curse",
        card_data={
            "name": "Debuff",
            "type_line": "Enchantment",
            "oracle_text": "All creatures get -1/-2.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    abilities = parse_static_abilities(card)
    
    assert len(abilities) >= 1
    ability = abilities[0]
    assert ability.power_mod == -1
    assert ability.toughness_mod == -2


def test_multiple_keywords():
    """Test abilities that grant multiple keywords."""
    card = CardInstance(
        instance_id="multi_lord",
        card_data={
            "name": "Mighty Lord",
            "type_line": "Creature",
            "oracle_text": "Creatures you control have haste and vigilance.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    abilities = parse_static_abilities(card)
    
    # Should parse at least one keyword ability
    assert len(abilities) >= 1
    # Should find haste or vigilance
    has_haste = any("haste" in [k.lower() for k in a.keywords] for a in abilities)
    has_vigilance = any("vigilance" in [k.lower() for k in a.keywords] for a in abilities)
    
    assert has_haste or has_vigilance


def test_creature_not_affected_without_matching_type():
    """Test that artifact lords don't affect creatures."""
    game = create_game_with_lords()
    
    # Create artifact lord
    artifact_lord = CardInstance(
        instance_id="artifact_lord",
        card_data={
            "name": "Artifact Lord",
            "type_line": "Creature",
            "oracle_text": "Artifacts you control get +1/+1.",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    game.cards.append(artifact_lord)
    
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
    
    # Should not be affected (creature, not artifact)
    affecting = get_static_abilities_affecting(game, creature)
    
    assert len(affecting) == 0, "Artifact lord should not affect creatures"


def test_card_with_no_static_abilities():
    """Test cards without static abilities."""
    vanilla = CardInstance(
        instance_id="vanilla",
        card_data={
            "name": "Grizzly Bears",
            "type_line": "Creature",
            "oracle_text": "",
            "power": "2",
            "toughness": "2",
            "set": "TEST"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    abilities = parse_static_abilities(vanilla)
    
    assert len(abilities) == 0, f"Vanilla card should have no static abilities, got {len(abilities)}"
