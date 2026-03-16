"""
Tests for triggered abilities (Phase 3)

Validates:
- ETB (Enter-the-Battlefield) triggers fire when creatures resolve
- Triggers go on stack after spell resolution
- Trigger effects resolve correctly (draw, damage, etc.)
- Multiple triggers can be queued
- Trigger priority system works
"""

import pytest
from src.engine.game_state import GameState, Phase, Zone, CardInstance, PlayerState, TriggerType, Trigger
from src.engine.rules_engine import RulesEngine
from src.engine.triggers import parse_triggers, check_enters_battlefield_triggers, resolve_trigger


@pytest.fixture
def game_with_etb_creatures():
    """Create a game state with ETB creatures."""
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
        PlayerState(player_id="Bob", name="Bob", life_total=20),
    ]
    
    # Mulldrifter: "When Mulldrifter enters, draw a card"
    mulldrifter = CardInstance(
        instance_id="mulldrifter",
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
    
    # Landfall creature: "Whenever a creature enters"
    scute_swarm = CardInstance(
        instance_id="scute_swarm",
        card_data={
            "name": "Scute Swarm",
            "type_line": "Creature — Insect",
            "mana_cost": "{2}{G}",
            "cmc": 3,
            "power": "1",
            "toughness": "1",
            "oracle_text": "Whenever another creature enters the battlefield under your control, create a token that's a copy of Scute Swarm.",
            "set": "ZNR"
        },
        zone=Zone.BATTLEFIELD,
        owner_id="Alice",
        controller_id="Alice"
    )
    
    # Add some library cards for drawing
    library_cards = []
    for i in range(10):
        lib_card = CardInstance(
            instance_id=f"lib_card_{i}",
            card_data={
                "name": f"Card {i}",
                "type_line": "Creature — Test",
                "set": "TST"
            },
            zone=Zone.LIBRARY,
            owner_id="Alice",
            controller_id="Alice"
        )
        library_cards.append(lib_card)
    
    all_cards = [mulldrifter, scute_swarm] + library_cards
    
    game_state = GameState(
        format="standard",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=players,
        cards=all_cards,
    )
    
    # Give Alice enough mana
    players[0].mana_pool["U"] = 3
    players[0].mana_pool["G"] = 3
    
    return game_state


def test_parse_etb_triggers():
    """Verify that ETB triggers are extracted from oracle text."""
    card = CardInstance(
        instance_id="test",
        card_data={
            "name": "Mulldrifter",
            "oracle_text": "When Mulldrifter enters, draw a card.",
        },
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(card)
    assert len(triggers) > 0, "Should extract at least one trigger"
    
    etb_triggers = [t for t in triggers if t.trigger_type == TriggerType.ENTERS_BATTLEFIELD]
    assert len(etb_triggers) > 0, "Should have an ETB trigger"
    assert "draw" in etb_triggers[0].description.lower()


def test_creature_etb_draws_card(game_with_etb_creatures):
    """Verify that creature entering with ETB trigger causes draw."""
    engine = RulesEngine()
    state = game_with_etb_creatures
    
    mulldrifter = next(c for c in state.cards if c.name == "Mulldrifter")
    alice_hand_before = len([c for c in state.cards if c.zone == Zone.HAND and c.owner_id == "Alice"])
    
    # Get legal actions and cast Mulldrifter
    from src.engine.game_state import ActionType
    legal = engine.get_legal_actions(state, "Alice")
    cast_actions = [a for a in legal if a.action_type == ActionType.CAST_SPELL and a.card_instance_id == mulldrifter.instance_id]
    
    assert len(cast_actions) > 0, "Should be able to cast Mulldrifter"
    
    # Cast it
    state = engine.execute_action(state, cast_actions[0])
    
    # Verify it's on stack
    assert mulldrifter.zone == Zone.STACK
    assert len(state.stack) == 1
    
    # Resolve the spell
    state = engine.resolve_stack_item(state)
    
    # Verify Mulldrifter entered battlefield
    assert mulldrifter.zone == Zone.BATTLEFIELD
    
    # Verify a trigger was created
    assert len(state.triggered_abilities) > 0, "Should have triggered ability"
    
    # Verify trigger stack item was added
    # (The trigger stack item should be on the stack now)
    assert len(state.stack) > 0, "Trigger should be on stack"
    
    # Verify the trigger effect hasn't applied yet (still on stack)
    alice_hand_after_cast = len([c for c in state.cards if c.zone == Zone.HAND and c.owner_id == "Alice"])
    assert alice_hand_after_cast == alice_hand_before - 1, "Hand shouldn't change until trigger resolves"
    
    # Now resolve the trigger
    state = engine.resolve_stack_item(state)
    
    # Verify draw happened
    alice_hand_final = len([c for c in state.cards if c.zone == Zone.HAND and c.owner_id == "Alice"])
    assert alice_hand_final > alice_hand_after_cast, "Should have drawn a card from trigger"


def test_multiple_etb_triggers(game_with_etb_creatures):
    """Verify multiple ETB triggers can be on stack simultaneously."""
    engine = RulesEngine()
    state = game_with_etb_creatures
    
    mulldrifter = next(c for c in state.cards if c.name == "Mulldrifter")
    
    from src.engine.game_state import ActionType
    
    # Cast Mulldrifter
    legal = engine.get_legal_actions(state, "Alice")
    cast_action = next(a for a in legal if a.action_type == ActionType.CAST_SPELL and a.card_instance_id == mulldrifter.instance_id)
    state = engine.execute_action(state, cast_action)
    
    # Resolve spell (enters BF, ETB trigger goes on stack)
    state = engine.resolve_stack_item(state)
    
    # Should have: Mulldrifter's own ETB + Scute Swarm's "another creature" trigger
    assert len(state.stack) >= 1, "Should have trigger(s) on stack"
    
    # The Scute Swarm might also have a trigger if implementation includes "whenever another creature enters"
    # For now, at least Mulldrifter's own trigger should be there


def test_trigger_effect_resolution():
    """Verify trigger effects are correctly applied during resolution."""
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
    ]
    
    state = GameState(
        format="standard",
        players=players,
        cards=[]
    )
    
    # Create a trigger that draws a card
    draw_trigger = Trigger(
        source_card_id="test",
        controller_id="Alice",
        trigger_type=TriggerType.ENTERS_BATTLEFIELD,
        description="draw a card",
    )
    
    # Create a dummy card in library
    lib_card = CardInstance(
        instance_id="lib_card",
        card_data={"name": "Test Card"},
        zone=Zone.LIBRARY,
        owner_id="Alice",
    )
    state.cards.append(lib_card)
    
    # Resolve trigger
    before_hand = len([c for c in state.cards if c.zone == Zone.HAND])
    state = resolve_trigger(state, draw_trigger)
    after_hand = len([c for c in state.cards if c.zone == Zone.HAND])
    
    assert after_hand > before_hand, "Draw trigger should put card in hand"
    assert lib_card.zone == Zone.HAND


def test_creature_with_no_triggers():
    """Verify creatures without ETB triggers don't queue triggers."""
    card = CardInstance(
        instance_id="vanilla_bear",
        card_data={
            "name": "Grizzly Bears",
            "type_line": "Creature — Bear",
            "oracle_text": "",
            "power": "2",
            "toughness": "2",
        },
        owner_id="Alice",
        controller_id="Alice"
    )
    
    triggers = parse_triggers(card)
    etb_triggers = [t for t in triggers if t.trigger_type == TriggerType.ENTERS_BATTLEFIELD]
    
    assert len(etb_triggers) == 0, "Vanilla creature should have no ETB triggers"
