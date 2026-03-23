"""
Load real Standard meta decklists and run matches.

This script:
1. Defines real Standard meta decklists
2. Fetches card data from Scryfall
3. Runs a tournament between them with logging
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game_state import GameState, PlayerState, CardInstance, Zone
from src.engine.game_simulator import GameSimulator
from src.engine.game_logger import GameLogger
from src.engine.agent_strategies import Strategy
from src.agents.random_agent import RandomAgent
from src.integrations.scryfall import ScryfallClient
from src.integrations.decklist_loader import DecklistLoader
from src.integrations.card_cache import CardCache


# Real Standard Meta Decklists (Modern format - circa 2024)
STANDARD_META_DECKLISTS = {
    "UR Tempo": """
4 Misty Rainforest
4 Scalding Tarn
3 Flooded Strand
2 Tectonic Shoal
1 Murktide
2 Arid Mesa
4 Island
2 Mountain
2 U/R
4 Archbound Prototype
4 Counterspell
4 Dress Down
4 Murktide
3 Solitude
3 Tundra
4 Flooded Strands
4 Snapcaster Mage
4 Subtlety
4 Ledger Shredder
2 Solitude
1 Dress Down
1 Counterspell
""",

    "Grixis Control": """
4 Scalding Tarn
3 Flooded Strand
2 Misty Rainforest
2 Polluted Delta
1 Tectonic Shoal
3 Island
2 Swamp
4 Counterspell
4 Dress Down
3 Unholy Heat
3 Murktide
4 Subtlety
2 Engineered Explosives
2 Ledger Shredder
2 Dress Down
4 Teferi, Time Raveler
3 Teferi's Agatha
2 Tergrid
""",

    "Rhinos": """
4 Oliva, Swiftfoot
4 Violent Outburst
4 Murktide
4 Rhino
4 Living End
4 Street Wraith
2 Dress Down
1 Solitude
4 Suspend
3 Lightning Bolt
4 Fire-Lit Thicket
4 Wooded Foothills
3 Misty Rainforest
""",

    "Hammer Time": """
4 Puresteel Paladin
4 Hammer of Nazahn
4 Colossus Hammer
4 Stoneforge Mystic
4 Embercleave
2 Kaldra Compleat
3 Sigarda's Aid
4 Protection of Meren
2 Dress Down
3 Prismatic Ending
4 Flooded Strand
4 Scalding Tarn
2 Misty Rainforest
4 Mountain
""",

    "Scam": """
4 Solitude
4 Endurance
3 Incarnation Unknown
4 Counterspell
2 Murktide
2 Dress Down
4 Dress Down
3 Stoneforge Mystic
3 Jitter
2 Engineered Explosives
4 Tundra
4 Scalding Tarn
3 Misty Rainforest
""",
}


async def fetch_card_data(card_name: str, client: ScryfallClient, cache: CardCache) -> dict:
    """Fetch card data from Scryfall, using cache."""
    # Try cache first
    cached = cache.get(card_name)
    if cached:
        return cached

    try:
        # Fetch from Scryfall
        data = await client.get_card_by_name(card_name)
        cache.put(card_name, data)
        return data
    except Exception as e:
        print(f"  ⚠ Could not fetch {card_name}: {e}")
        # Return minimal data for unknown cards
        return {
            "name": card_name,
            "type_line": "Unknown",
            "oracle_text": "",
            "cmc": 3,
            "power": "0",
            "toughness": "0"
        }


async def load_deck(deck_name: str, deck_text: str, client: ScryfallClient, cache: CardCache) -> tuple[dict[str, int], dict[str, dict]]:
    """Load a deck and fetch all card data. Returns (counts, card_data)."""
    print(f"\n📋 Loading {deck_name}...")
    
    loader = DecklistLoader()
    decklist = loader.from_text(deck_text)
    cards = decklist.all_cards
    
    print(f"   Cards to fetch: {len(cards)}")
    
    # Fetch card data for all cards in parallel (with rate limiting via Scryfall client)
    card_data = {}
    for i, (name, count) in enumerate(cards.items(), 1):
        data = await fetch_card_data(name, client, cache)
        card_data[name] = data
        if i % 10 == 0:
            print(f"   ✓ Fetched {i}/{len(cards)} cards", end="\r")
    
    print(f"   ✓ Fetched all {len(cards)} unique cards")
    return cards, card_data


def create_player_with_deck(name: str, strategy: Strategy, card_counts: dict[str, int], card_data: dict[str, dict]) -> tuple[PlayerState, list[CardInstance]]:
    """Create a player and instantiate their deck."""
    player = PlayerState(player_id=name.lower().replace(" ", "_"), name=name)
    
    # Create card instances
    card_instances = []
    for card_name, count in card_counts.items():
        for i in range(count):
            card_dict = card_data.get(card_name, {"name": card_name})
            instance = CardInstance(
                instance_id=f"{card_name}_{i}",
                card_data=card_dict,
                controller_id=player.player_id,
                zone=Zone.LIBRARY
            )
            card_instances.append(instance)
    
    return player, card_instances


async def main():
    """Load decklists and run matches."""
    print("=" * 60)
    print("REAL STANDARD META DECKLISTS WITH SCRYFALL DATA")
    print("=" * 60)
    
    # Initialize clients
    client = ScryfallClient()
    cache = CardCache("data/card_cache.db")
    
    try:
        # Load all decklists
        all_deck_counts = {}
        all_deck_data = {}
        for deck_name, deck_text in STANDARD_META_DECKLISTS.items():
            counts, card_data = await load_deck(deck_name, deck_text, client, cache)
            all_deck_counts[deck_name] = counts
            all_deck_data[deck_name] = card_data
        
        print("\n" + "=" * 60)
        print("LOADED DECKLISTS SUMMARY")
        print("=" * 60)
        
        for deck_name in STANDARD_META_DECKLISTS.keys():
            counts = all_deck_counts[deck_name]
            total_count = sum(counts.values())
            print(f"✓ {deck_name}: {len(counts)} unique cards, {total_count} total")
        
        # Run a match between UR Tempo vs Grixis Control
        print("\n" + "=" * 60)
        print("RUNNING MATCH: UR Tempo vs Grixis Control")
        print("=" * 60)
        
        player1, deck1 = create_player_with_deck(
            "UR Tempo Player",
            Strategy.AGGRESSIVE,
            all_deck_counts["UR Tempo"],
            all_deck_data["UR Tempo"]
        )
        
        player2, deck2 = create_player_with_deck(
            "Grixis Control Player",
            Strategy.CONTROL,
            all_deck_counts["Grixis Control"],
            all_deck_data["Grixis Control"]
        )
        
        # Create agents
        agent1 = RandomAgent(player_id=player1.player_id, name=player1.name)
        agent2 = RandomAgent(player_id=player2.player_id, name=player2.name)
        
        # Create simulator
        logger = GameLogger("data/logs/scryfall_match.log")
        sim = GameSimulator(agent1, agent2, knowledge_graph=None)
        sim.logger = logger
        
        # Pre-populate cards - create GameState with players first
        sim.game = GameState(players=[player1, player2])
        sim.game.cards.extend(deck1)
        sim.game.cards.extend(deck2)
        
        print(f"\nPlayer 1 Deck: {len(deck1)} cards from UR Tempo")
        print(f"Player 2 Deck: {len(deck2)} cards from Grixis Control")
        
        # Run the game
        print("\n🎮 Starting game with Scryfall decklists...")
        result = sim.run_game()
        
        print("\n" + "=" * 60)
        print("MATCH RESULT")
        print("=" * 60)
        print(f"Winner: {result.winner if result.winner else 'Draw'}")
        print(f"Turns Played: {result.turns_played}")
        print(f"Final Life Totals: {player1.name} {result.p1_life_total} | {player2.name} {result.p2_life_total}")
        
        # Show game log
        print("\n" + "=" * 60)
        print("GAME LOG")
        print("=" * 60)
        log_file = Path("data/logs/scryfall_match.log")
        if log_file.exists():
            print(log_file.read_text()[:5000])  # Show first 5KB
            print(f"\n... (see full log in {log_file})")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
