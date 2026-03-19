"""Full Standard Game with Comprehensive Logging.

Plays a complete MTG game between agents with detailed logging and hand visibility.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game_simulator import GameSimulator
from src.engine.game_execution import AgentGamePlayer
from src.engine.agent_strategies import Strategy
from src.engine.game_state import CardInstance, Zone


def play_logged_game():
    """Play a single full game with comprehensive logging."""
    print("\n" + "="*70)
    print("MAGIC: THE GATHERING - FULL LOGGED GAME")
    print("="*70)
    print("\nPhase 6c: Game with Logging and Hand Visibility")
    print("="*70 + "\n")
    
    # Create two agents (without LLM - heuristic play is faster)
    print("Initializing agents...")
    aggro_agent = AgentGamePlayer(
        "UR Aggro",
        Strategy.AGGRESSIVE,
        knowledge_graph=None,
        use_llm=False  # Heuristic play for speed
    )
    
    control_agent = AgentGamePlayer(
        "Izzet Control",
        Strategy.CONTROL,
        knowledge_graph=None,
        use_llm=False  # Heuristic play for speed
    )
    
    print(f"✓ {aggro_agent.player_id} ({aggro_agent.strategy.value})")
    print(f"✓ {control_agent.player_id} ({control_agent.strategy.value})\n")
    
    # Create simulator
    sim = GameSimulator(aggro_agent, control_agent, max_turns=10)
    
    # Setup game
    sim.setup_game()
    
    # Add cards to hands
    print("Adding cards to hands...")
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
            "name": "Island",
            "mana_cost": "",
            "type_line": "Basic Land — Island",
            "cmc": 0,
            "oracle_text": "{T}: Add {U}.",
        },
        {
            "name": "Mountain",
            "mana_cost": "",
            "type_line": "Basic Land — Mountain",
            "cmc": 0,
            "oracle_text": "{T}: Add {R}.",
        },
        {
            "name": "Counterspell",
            "mana_cost": "{U}{U}",
            "type_line": "Instant",
            "cmc": 2,
            "oracle_text": "Counter target spell.",
        },
    ]
    
    # Add cards to both players
    for player_id in [aggro_agent.player_id, control_agent.player_id]:
        for i, card_data in enumerate(cards_to_add):
            card = CardInstance(
                instance_id=f"{player_id}_card_{i}",
                card_data=card_data,
                controller_id=player_id,
                zone=Zone.HAND,
                owner_id=player_id,
                tapped=False
            )
            sim.game.cards.append(card)
    
    print(f"✓ Added {len(cards_to_add)} cards to each player\n")
    
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
    play_logged_game()
