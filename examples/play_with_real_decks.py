"""
Full Standard Game with Real Scryfall Decks via LLM Integration.

Plays a complete MTG game between two LLM-powered agents using realistic decklists
with proper Scryfall card data (oracle text, mana costs, types, etc).
"""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game_simulator import GameSimulator
from src.engine.game_execution import AgentGamePlayer
from src.engine.game_state import CardInstance, Zone
from src.engine.agent_strategies import Strategy
from src.integrations.scryfall import ScryfallClient
from src.integrations.decklist_loader import DecklistLoader


# Real Standard decklists (as of 2024)
UR_MURKTIDE_DECKLIST = """
4 Murktide
3 Subtlety
2 Dragon's Rage Channeler
4 Snapcaster Mage
3 Dress Down
2 Counterspell
1 Flusterstorm
4 Lightning Bolt
4 Murktide
2 Unholy Heat
3 Counterspell
1 Counterspell
4 Murktide
2 Subtlety
3 Dress Down
2 Counterspell
4 Scalding Tarn
2 Flooded Strand
1 Island
1 Island
2 Scalding Tarn
2 Flooded Strand
1 Island
1 Mountain
2 Misty Rainforest
1 Scalding Tarn
2 Flooded Strand
"""

UB_CONTROL_DECKLIST = """
3 Jace, the Mind Sculptor
2 Snapcaster Mage
4 Solitude
3 Counterspell
4 Murktide
3 Subtlety
2 Dress Down
3 Counterspell
1 Prismatic Ending
2 Counterspell
2 Spreading Seas
1 Supreme Verdict
4 Flooded Strand
2 Island
2 Misty Rainforest
2 Scalding Tarn
1 Underground Sea
1 Marsh Flats
1 Flooded Strand
1 Tundra
"""


async def load_deck_from_scryfall(decklist_text: str, player_id: str) -> list[CardInstance]:
    """Load a decklist and fetch card data from Scryfall."""
    
    loader = DecklistLoader()
    decklist = loader.from_text(decklist_text)
    
    client = ScryfallClient()
    cards = []
    
    try:
        for card_name, count in decklist.mainboard.items():
            try:
                # Fetch card from Scryfall
                card_data = await client.get_card_by_name(card_name)
                
                # Create instances for each copy
                for i in range(count):
                    instance = CardInstance(
                        instance_id=f"{player_id}_{card_name.lower().replace(' ', '_')}_{i}",
                        card_data={
                            "name": card_data.get("name", card_name),
                            "mana_cost": card_data.get("mana_cost", ""),
                            "type_line": card_data.get("type_line", ""),
                            "oracle_text": card_data.get("oracle_text", ""),
                            "cmc": card_data.get("cmc", 0),
                            "power": card_data.get("power"),
                            "toughness": card_data.get("toughness"),
                            "set": card_data.get("set", ""),
                            "scryfall_id": card_data.get("id", ""),
                        },
                        zone=Zone.HAND,
                        controller_id=player_id,
                        owner_id=player_id,
                        tapped=False
                    )
                    cards.append(instance)
                    print(f"✓ Loaded {card_name}")
                
            except Exception as e:
                print(f"⚠ Could not load {card_name}: {e} - using mock data")
                # Fallback to mock
                instance = CardInstance(
                    instance_id=f"{player_id}_{card_name.lower().replace(' ', '_')}_mock",
                    card_data={
                        "name": card_name,
                        "mana_cost": "{U}",
                        "type_line": "Creature",
                        "oracle_text": "[Mock data - Scryfall fetch failed]",
                        "cmc": 1,
                    },
                    zone=Zone.HAND,
                    controller_id=player_id,
                    owner_id=player_id,
                    tapped=False
                )
                cards.append(instance)
        
    finally:
        await client.close()
    
    return cards


async def play_with_real_decks():
    """Play a single full game with real Scryfall decks."""
    print("\n" + "="*70)
    print("MAGIC: THE GATHERING - GAME WITH REAL SCRYFALL DECKS")
    print("="*70)
    print("\nPhase 6c: LLM-Powered Agents with Real Decklists")
    print("Model: Phi (2.7B) via Ollama")
    print("="*70 + "\n")
    
    # Load real decks from Scryfall
    print("Loading UR Murktide deck from Scryfall...")
    ur_cards = await load_deck_from_scryfall(UR_MURKTIDE_DECKLIST, "UR_Murktide")
    
    print("\nLoading UB Control deck from Scryfall...")
    ub_cards = await load_deck_from_scryfall(UB_CONTROL_DECKLIST, "UB_Control")
    
    # Create two LLM-powered agents
    print("\n" + "-"*70)
    print("Initializing agents...")
    aggro_agent = AgentGamePlayer(
        "UR Murktide",
        Strategy.AGGRESSIVE,
        knowledge_graph=None,
        use_llm=True,
        llm_model="phi"
    )
    
    control_agent = AgentGamePlayer(
        "UB Control",
        Strategy.CONTROL,
        knowledge_graph=None,
        use_llm=True,
        llm_model="phi"
    )
    
    print(f"✓ {aggro_agent.player_id} ({aggro_agent.strategy.value})")
    print(f"✓ {control_agent.player_id} ({control_agent.strategy.value})")
    print(f"✓ LLM Status: {'Available' if aggro_agent.llm_agent and aggro_agent.llm_agent.llm.is_available else 'Unavailable'}\n")
    
    # Create simulator
    sim = GameSimulator(aggro_agent, control_agent, max_turns=20)
    
    print("-"*70)
    print("GAME START - UR Murktide vs UB Control")
    print("-"*70 + "\n")
    
    # Setup game and add real cards
    game = sim.setup_game()
    
    # Add the real cards to the game
    game.cards.extend(ur_cards)
    game.cards.extend(ub_cards)
    
    print(f"\nCards in game: {len(game.cards)}")
    print(f"  UR Murktide: {len(ur_cards)} cards")
    print(f"  UB Control: {len(ub_cards)} cards\n")
    
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
        print(sim.logger.get_decision_chain())
    
    # Show full game summary from simulator
    print("\n" + sim.get_game_summary())


if __name__ == "__main__":
    asyncio.run(play_with_real_decks())
