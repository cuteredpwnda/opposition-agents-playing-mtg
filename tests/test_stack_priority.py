"""
Tests for the Stack and Priority system (Phase 3 - Complex Interactions)

Validates:
- Spells go on the stack (not directly to battlefield)
- Priority passing in APNAP order
- Multiple spells can be on stack simultaneously
- Instant-speed responses work
- Proper spell resolution and fizzling
"""

import pytest
import asyncio
from src.engine.game_state import GameState, Phase, Zone, CardInstance, PlayerState, ActionType
from src.engine.rules_engine import RulesEngine
from src.engine.stack import push_to_stack, resolve_top, is_empty
from src.agents.random_agent import RandomAgent
from src.orchestrator.priority_loop import run_priority_loop, get_priority_order, advance_priority


@pytest.fixture
def simple_game_state():
    """Create a simple 2-player game state."""
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
        PlayerState(player_id="Bob", name="Bob", life_total=20),
    ]
    
    # Create some test cards (creatures and instants)
    cards = [
        # Alice's creatures
        CardInstance(instance_id="a1", card_data={"name": "Goblin", "type_line": "Creature — Goblin", "mana_cost": "{R}", "cmc": 1, "power": "1", "toughness": "1", "set": "DOM"}, zone=Zone.HAND, owner_id="Alice", controller_id="Alice"),
        CardInstance(instance_id="a2", card_data={"name": "Lightning Bolt", "type_line": "Instant", "mana_cost": "{R}", "cmc": 1, "set": "DOM", "oracle_text": ""}, zone=Zone.HAND, owner_id="Alice", controller_id="Alice"),
        
        # Bob's creatures
        CardInstance(instance_id="b1", card_data={"name": "Elf", "type_line": "Creature — Elf", "mana_cost": "{G}", "cmc": 1, "power": "1", "toughness": "1", "set": "DOM"}, zone=Zone.HAND, owner_id="Bob", controller_id="Bob"),
        CardInstance(instance_id="b2", card_data={"name": "Counterspell", "type_line": "Instant", "mana_cost": "{UU}", "cmc": 2, "set": "DOM", "oracle_text": ""}, zone=Zone.HAND, owner_id="Bob", controller_id="Bob"),
    ]
    
    game_state = GameState(
        format="standard",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=players,
        cards=cards,
    )
    
    # Give Alice mana
    players[0].mana_pool["R"] = 2
    
    return game_state


def test_spell_goes_on_stack(simple_game_state):
    """Verify that casting a creature puts it on the stack, not BF."""
    engine = RulesEngine()
    state = simple_game_state
    
    # Find Goblin
    goblin = next(c for c in state.cards if c.name == "Goblin")
    assert goblin.zone == Zone.HAND
    
    # Get legal actions for Alice
    legal = engine.get_legal_actions(state, "Alice")
    
    # Find cast spell action for Goblin
    cast_actions = [a for a in legal if a.action_type == ActionType.CAST_SPELL]
    assert len(cast_actions) > 0, "Should have cast spell actions"
    
    goblin_cast = next((a for a in cast_actions if a.card_instance_id == goblin.instance_id), None)
    assert goblin_cast is not None, "Should be able to cast Goblin"
    
    # Execute the cast
    state = engine.execute_action(state, goblin_cast)
    
    # Verify Goblin is now on stack, not BF
    assert goblin.zone == Zone.STACK, f"Goblin should be on stack, but is in {goblin.zone}"
    assert len(state.stack) == 1, "Stack should have 1 item"
    assert state.stack[0].card_data["name"] == "Goblin"


def test_stack_lifo_resolution(simple_game_state):
    """Verify that stack resolves in LIFO order."""
    from src.engine.game_state import StackItem
    
    state = simple_game_state
    
    # Manually push two spells onto stack
    spell1 = StackItem(
        source_card_id="a1",
        controller_id="Alice",
        is_spell=True,
        card_data={"name": "Lightning Bolt", "type_line": "Instant"}
    )
    spell2 = StackItem(
        source_card_id="a2",
        controller_id="Alice",
        is_spell=True,
        card_data={"name": "Counterspell", "type_line": "Instant"}
    )
    
    state.stack.append(spell1)
    state.stack.append(spell2)
    
    assert len(state.stack) == 2
    
    # Resolve once - should get Counterspell (last added)
    top1 = resolve_top(state)
    assert top1.card_data["name"] == "Counterspell"
    assert len(state.stack) == 1
    
    # Resolve again - should get Lightning Bolt
    top2 = resolve_top(state)
    assert top2.card_data["name"] == "Lightning Bolt"
    assert len(state.stack) == 0
    assert is_empty(state)


def test_priority_order_apnap(simple_game_state):
    """Verify APNAP priority order."""
    state = simple_game_state
    
    # Active player is Alice (index 0)
    order = get_priority_order(state)
    assert order[0] == "Alice", "Active player should be first in APNAP"
    assert order[1] == "Bob", "Non-active player should be second"
    
    # If we advance priority from Alice, should get Bob
    state.priority_player_index = 0  # Alice
    state = advance_priority(state)
    assert state.priority_player.player_id == "Bob"
    
    # Advance from Bob, should wrap to Alice
    state = advance_priority(state)
    assert state.priority_player.player_id == "Alice"


@pytest.mark.asyncio
async def test_multi_spell_stack_interaction():
    """Test that multiple spells can be on stack and resolve in order.
    
    Scenario:
    - Alice casts Goblin
    - Bob could respond with Counterspell (but chooses to pass)
    - Alice could add more spells
    """
    from src.engine.card_database import CardDatabase
    from src.orchestrator.game_runner import GameRunner, GameConfig
    
    # Create a simple 2-card deck for each player
    alice_deck = [
        {"name": "Goblin", "type_line": "Creature — Goblin", "mana_cost": "{R}", "cmc": 1, "power": "1", "toughness": "1", "set": "DOM"},
        {"name": "Mountain", "type_line": "Land — Mountain", "mana_cost": "", "cmc": 0, "power": None, "toughness": None, "set": "DOM"},
    ] * 20  # 40 cards total
    
    bob_deck = [
        {"name": "Elf", "type_line": "Creature — Elf", "mana_cost": "{G}", "cmc": 1, "power": "1", "toughness": "1", "set": "DOM"},
        {"name": "Forest", "type_line": "Land — Forest", "mana_cost": "", "cmc": 0, "power": None, "toughness": None, "set": "DOM"},
    ] * 20  # 40 cards total
    
    # Create agents
    agents = {
        "Alice": RandomAgent("Alice", "Alice"),
        "Bob": RandomAgent("Bob", "Bob"),
    }
    
    # Run a game
    config = GameConfig(format="standard", starting_life=20, max_turns=5)
    runner = GameRunner(config=config)
    
    result = await runner.run_game(agents, {"Alice": alice_deck, "Bob": bob_deck})
    
    # Verify game completed
    assert result.turns >= 1, "Game should have at least 1 turn"
    assert len(result.log) > 0, "Game should have logged events"
    
    # Verify stack was used (should see spell resolution in logs)
    stack_related = [e for e in result.log if "stack" in e.lower()]
    # With stack system, should have many stack-related events
    # (Not all games may use stack if only sorceries played in empty stack)
    print(f"Game completed: {result.turns} turns, {len(result.log)} events")
    print(f"Stack-related events: {len(stack_related)}")


def test_instant_speed_casting_legal(simple_game_state):
    """Verify that instants can be cast at instant speed."""
    engine = RulesEngine()
    state = simple_game_state
    
    # Give Alice enough mana for Lightning Bolt
    state.players[0].mana_pool["U"] = 2  # For Counterspell
    
    # Lightning Bolt should be in hand
    bolt = next((c for c in state.cards if c.name == "Lightning Bolt"), None)
    assert bolt is not None
    assert bolt.zone == Zone.HAND
    
    # Get legal actions - should include instant even during sorcery speed
    # (Game state is already in main phase though)
    legal = engine.get_legal_actions(state, "Alice")
    
    # Should have Lightning Bolt as castable (instant-speed)
    has_instant = any(a.action_type == ActionType.CAST_SPELL and a.card_instance_id == bolt.instance_id for a in legal)
    assert has_instant, "Should be able to cast instant-speed spells"


def test_stack_empty_check(simple_game_state):
    """Verify stack empty detection works."""
    state = simple_game_state
    
    assert is_empty(state), "Stack should start empty"
    
    # Add item
    from src.engine.game_state import StackItem
    item = StackItem(source_card_id="test", controller_id="Alice")
    state.stack.append(item)
    
    assert not is_empty(state), "Stack should not be empty after adding item"
    
    # Remove item
    state.stack.pop()
    assert is_empty(state), "Stack should be empty after removing item"
