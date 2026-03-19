"""Simple game test without LLM to check hand logging."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game_simulator import GameSimulator
from src.engine.game_execution import AgentGamePlayer
from src.engine.agent_strategies import Strategy
from src.engine.game_state import GameState, PlayerState, CardInstance, Zone


def play_simple_game():
    """Play a game without LLM to test hand logging."""
    print("\n" + "="*70)
    print("SIMPLE GAME TEST - WITHOUT LLM")
    print("="*70 + "\n")
    
    # Create two agents (NO LLM)
    print("Initializing agents...")
    aggro_agent = AgentGamePlayer(
        "UR Aggro",
        Strategy.AGGRESSIVE,
        knowledge_graph=None,
        use_llm=False  # NO LLM
    )
    
    control_agent = AgentGamePlayer(
        "Izzet Control",
        Strategy.CONTROL,
        knowledge_graph=None,
        use_llm=False  # NO LLM
    )
    
    print(f"✓ {aggro_agent.player_id} ({aggro_agent.strategy.value})")
    print(f"✓ {control_agent.player_id} ({control_agent.strategy.value})\n")
    
    # Create simulator
    sim = GameSimulator(aggro_agent, control_agent, max_turns=5)
    
    # Setup game
    sim.setup_game()
    
    # Add some test cards to hands
    print("Adding test cards to hands...")
    cards_to_add = [
        {
            "name": "Snapcaster Mage",
            "mana_cost": "{1}{U}",
            "type_line": "Creature — Wizard",
            "cmc": 2,
            "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn.",
            "power": "2",
            "toughness": "1"
        },
        {
            "name": "Lightning Bolt",
            "mana_cost": "{R}",
            "type_line": "Instant",
            "cmc": 1,
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
        },
        {
            "name": "Counterspell",
            "mana_cost": "{U}{U}",
            "type_line": "Instant",
            "cmc": 2,
            "oracle_text": "Counter target spell.",
        }
    ]
    
    for i, card_data in enumerate(cards_to_add):
        card = CardInstance(
            instance_id=f"aggro_card_{i}",
            card_data=card_data,
            controller_id="UR Aggro",
            zone=Zone.HAND,
            owner_id="UR Aggro",
            tapped=False
        )
        sim.game.cards.append(card)
    
    for i, card_data in enumerate(cards_to_add):
        card = CardInstance(
            instance_id=f"control_card_{i}",
            card_data=card_data,
            controller_id="Izzet Control",
            zone=Zone.HAND,
            owner_id="Izzet Control",
            tapped=False
        )
        sim.game.cards.append(card)
    
    print(f"✓ Added {len(cards_to_add)} cards to each player's hand")
    print(f"  Total cards in game.cards: {len(sim.game.cards)}")
    
    # Debug: Check what's in the hand
    aggro_hand = [c for c in sim.game.cards if c.zone == Zone.HAND and c.controller_id == "UR Aggro"]
    control_hand = [c for c in sim.game.cards if c.zone == Zone.HAND and c.controller_id == "Izzet Control"]
    print(f"  UR Aggro hand: {len(aggro_hand)} cards - {[c.name for c in aggro_hand]}")
    print(f"  Izzet Control hand: {len(control_hand)} cards - {[c.name for c in control_hand]}\n")
    
    print("-"*70)
    print("GAME START")
    print("-"*70 + "\n")
    
    # Run the game
    result = sim.run_game()
    
    # Display results
    print("\n" + "="*70)
    print("GAME RESULT")
    print("="*70)
    print(f"\nWinner: {result.value}")
    print(f"Final Turn: {sim.game.turn_number}")
    print(f"{sim.game.players[0].name}: {sim.game.players[0].life_total} HP")
    print(f"{sim.game.players[1].name}: {sim.game.players[1].life_total} HP")
    
    # Show game summary
    if sim.logger:
        print(sim.logger.get_summary())


if __name__ == "__main__":
    play_simple_game()
