"""
Standard Meta Game Runner - March 2026

Runs games between top standard meta decks with LLM-powered agents.

Current meta (March 2026):
- UR Aggro (21% of meta)
- Izzet Control (17% of meta)
- Mono Green Aggro (16% of meta)
- Simic Aggro (7% of meta)
- Reanimator (7% of combo)
"""

import sys
from pathlib import Path

# Add parent directory (workspace root) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game_simulator import GameSimulator
from src.engine.game_execution import AgentGamePlayer
from src.engine.agent_strategies import Strategy
from src.engine.game_state import Zone


def build_ur_aggro_deck() -> list[dict]:
    """Build UR Aggro deck list (March 2026 meta).
    
    Standard aggressive blue-red tempo deck with creatures and burn spells.
    """
    deck = []
    
    # Creatures (24)
    creatures = [
        # 4x Snapcaster Mage (flash creature, 2/1)
        {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature - Human Wizard", 
         "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to its mana cost.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": ["flash"], "set": "DOM"},
        {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature - Human Wizard", 
         "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to its mana cost.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": ["flash"], "set": "DOM"},
        {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature - Human Wizard", 
         "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to its mana cost.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": ["flash"], "set": "DOM"},
        {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature - Human Wizard", 
         "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to its mana cost.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": ["flash"], "set": "DOM"},
        
        # 4x Dragon's Rage Channeler (surveil creature, 1/1)
        {"name": "Dragon's Rage Channeler", "mana_cost": "{R}", "type_line": "Creature - Human Shaman",
         "oracle_text": "Whenever Dragon's Rage Channeler attacks, Mill 1.\nWhenever an instant or sorcery spell you control is Counterspelled, put a +1/+1 counter on Dragon's Rage Channeler.",
         "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "MH2"},
        {"name": "Dragon's Rage Channeler", "mana_cost": "{R}", "type_line": "Creature - Human Shaman",
         "oracle_text": "Whenever Dragon's Rage Channeler attacks, Mill 1.\nWhenever an instant or sorcery spell you control is Counterspelled, put a +1/+1 counter on Dragon's Rage Channeler.",
         "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "MH2"},
        {"name": "Dragon's Rage Channeler", "mana_cost": "{R}", "type_line": "Creature - Human Shaman",
         "oracle_text": "Whenever Dragon's Rage Channeler attacks, Mill 1.\nWhenever an instant or sorcery spell you control is Counterspelled, put a +1/+1 counter on Dragon's Rage Channeler.",
         "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "MH2"},
        {"name": "Dragon's Rage Channeler", "mana_cost": "{R}", "type_line": "Creature - Human Shaman",
         "oracle_text": "Whenever Dragon's Rage Channeler attacks, Mill 1.\nWhenever an instant or sorcery spell you control is Counterspelled, put a +1/+1 counter on Dragon's Rage Channeler.",
         "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "MH2"},
        
        # 4x Tarmogoyf (growing creature)
        {"name": "Tarmogoyf", "mana_cost": "{1}{G}", "type_line": "Creature - Lhurgoyf",
         "oracle_text": "Tarmogoyf's power is equal to the number of card types among cards in all graveyards and its toughness is equal to that number plus 1.",
         "power": "*", "toughness": "*", "cmc": 2, "keywords": [], "set": "FUT"},
        {"name": "Tarmogoyf", "mana_cost": "{1}{G}", "type_line": "Creature - Lhurgoyf",
         "oracle_text": "Tarmogoyf's power is equal to the number of card types among cards in all graveyards and its toughness is equal to that number plus 1.",
         "power": "*", "toughness": "*", "cmc": 2, "keywords": [], "set": "FUT"},
        {"name": "Tarmogoyf", "mana_cost": "{1}{G}", "type_line": "Creature - Lhurgoyf",
         "oracle_text": "Tarmogoyf's power is equal to the number of card types among cards in all graveyards and its toughness is equal to that number plus 1.",
         "power": "*", "toughness": "*", "cmc": 2, "keywords": [], "set": "FUT"},
        {"name": "Tarmogoyf", "mana_cost": "{1}{G}", "type_line": "Creature - Lhurgoyf",
         "oracle_text": "Tarmogoyf's power is equal to the number of card types among cards in all graveyards and its toughness is equal to that number plus 1.",
         "power": "*", "toughness": "*", "cmc": 2, "keywords": [], "set": "FUT"},
        
        # 4x Murktide, Twin of Death (recursive creature)
        {"name": "Murktide, Twin of Death", "mana_cost": "{1}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nWhen Murktide, Twin of Death enters the battlefield, it becomes a copy of target creature you control except it has haste and flying.",
         "power": "3", "toughness": "2", "cmc": 3, "keywords": ["flying", "delve"], "set": "MH3"},
        {"name": "Murktide, Twin of Death", "mana_cost": "{1}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nWhen Murktide, Twin of Death enters the battlefield, it becomes a copy of target creature you control except it has haste and flying.",
         "power": "3", "toughness": "2", "cmc": 3, "keywords": ["flying", "delve"], "set": "MH3"},
        {"name": "Murktide, Twin of Death", "mana_cost": "{1}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nWhen Murktide, Twin of Death enters the battlefield, it becomes a copy of target creature you control except it has haste and flying.",
         "power": "3", "toughness": "2", "cmc": 3, "keywords": ["flying", "delve"], "set": "MH3"},
        {"name": "Murktide, Twin of Death", "mana_cost": "{1}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nWhen Murktide, Twin of Death enters the battlefield, it becomes a copy of target creature you control except it has haste and flying.",
         "power": "3", "toughness": "2", "cmc": 3, "keywords": ["flying", "delve"], "set": "MH3"},
        
        # 4x Murktide (original)
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
    ]
    deck.extend(creatures)
    
    # Removal/Burn (12)
    removal = [
        # 4x Lightning Bolt
        {"name": "Lightning Bolt", "mana_cost": "{R}", "type_line": "Instant", 
         "oracle_text": "Lightning Bolt deals 3 damage to any target.", 
         "power": None, "toughness": None, "cmc": 1, "keywords": [], "set": "A25"},
        {"name": "Lightning Bolt", "mana_cost": "{R}", "type_line": "Instant", 
         "oracle_text": "Lightning Bolt deals 3 damage to any target.", 
         "power": None, "toughness": None, "cmc": 1, "keywords": [], "set": "A25"},
        {"name": "Lightning Bolt", "mana_cost": "{R}", "type_line": "Instant", 
         "oracle_text": "Lightning Bolt deals 3 damage to any target.", 
         "power": None, "toughness": None, "cmc": 1, "keywords": [], "set": "A25"},
        {"name": "Lightning Bolt", "mana_cost": "{R}", "type_line": "Instant", 
         "oracle_text": "Lightning Bolt deals 3 damage to any target.", 
         "power": None, "toughness": None, "cmc": 1, "keywords": [], "set": "A25"},
        
        # 4x Counterspell
        {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "A25"},
        {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "A25"},
        {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "A25"},
        {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "A25"},
        
        # 4x Unholy Heat
        {"name": "Unholy Heat", "mana_cost": "{B}{R}", "type_line": "Instant",
         "oracle_text": "Unholy Heat deals X damage to any target, where X is 2 plus the number of card types in your graveyard.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "MH2"},
        {"name": "Unholy Heat", "mana_cost": "{B}{R}", "type_line": "Instant",
         "oracle_text": "Unholy Heat deals X damage to any target, where X is 2 plus the number of card types in your graveyard.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "MH2"},
        {"name": "Unholy Heat", "mana_cost": "{B}{R}", "type_line": "Instant",
         "oracle_text": "Unholy Heat deals X damage to any target, where X is 2 plus the number of card types in your graveyard.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "MH2"},
        {"name": "Unholy Heat", "mana_cost": "{B}{R}", "type_line": "Instant",
         "oracle_text": "Unholy Heat deals X damage to any target, where X is 2 plus the number of card types in your graveyard.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "MH2"},
    ]
    deck.extend(removal)
    
    # Lands (24)
    lands = [
        # Mana base
        {"name": "Scalding Tarn", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Scalding Tarn: Search your library for an Island or Mountain card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZEN"},
        {"name": "Scalding Tarn", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Scalding Tarn: Search your library for an Island or Mountain card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZEN"},
        {"name": "Scalding Tarn", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Scalding Tarn: Search your library for an Island or Mountain card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZEN"},
        {"name": "Scalding Tarn", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Scalding Tarn: Search your library for an Island or Mountain card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZEN"},
        
        # 4x Steam Vents
        {"name": "Steam Vents", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "({T}: Add {U} or {R}.)\nWhenever Steam Vents enters the battlefield, you gain 1 life.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RNA"},
        {"name": "Steam Vents", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "({T}: Add {U} or {R}.)\nWhenever Steam Vents enters the battlefield, you gain 1 life.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RNA"},
        {"name": "Steam Vents", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "({T}: Add {U} or {R}.)\nWhenever Steam Vents enters the battlefield, you gain 1 life.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RNA"},
        {"name": "Steam Vents", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "({T}: Add {U} or {R}.)\nWhenever Steam Vents enters the battlefield, you gain 1 life.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RNA"},
        
        # 4x Island
        {"name": "Island", "mana_cost": "", "type_line": "Basic Land — Island", 
         "oracle_text": "{T}: Add {U}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Island", "mana_cost": "", "type_line": "Basic Land — Island", 
         "oracle_text": "{T}: Add {U}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Island", "mana_cost": "", "type_line": "Basic Land — Island", 
         "oracle_text": "{T}: Add {U}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Island", "mana_cost": "", "type_line": "Basic Land — Island", 
         "oracle_text": "{T}: Add {U}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        
        # 4x Mountain
        {"name": "Mountain", "mana_cost": "", "type_line": "Basic Land — Mountain", 
         "oracle_text": "{T}: Add {R}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Mountain", "mana_cost": "", "type_line": "Basic Land — Mountain", 
         "oracle_text": "{T}: Add {R}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Mountain", "mana_cost": "", "type_line": "Basic Land — Mountain", 
         "oracle_text": "{T}: Add {R}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Mountain", "mana_cost": "", "type_line": "Basic Land — Mountain", 
         "oracle_text": "{T}: Add {R}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        
        # 4x Other duals
        {"name": "Flooded Strand", "mana_cost": "", "type_line": "Land", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for a Plains or Island card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
        {"name": "Flooded Strand", "mana_cost": "", "type_line": "Land", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for a Plains or Island card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
        {"name": "Flooded Strand", "mana_cost": "", "type_line": "Land", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for a Plains or Island card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
        {"name": "Flooded Strand", "mana_cost": "", "type_line": "Land", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for a Plains or Island card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
    ]
    deck.extend(lands)
    
    return deck


def build_izzet_control_deck() -> list[dict]:
    """Build Izzet Control deck list (March 2026 meta).
    
    Standard blue-red control deck with counterspells and board wipes.
    """
    deck = []
    
    # Creatures (8) - mainly for utility
    creatures = [
        # 4x Murktide
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        
        # 4x Snapcaster Mage
        {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature - Human Wizard", 
         "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to its mana cost.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": ["flash"], "set": "DOM"},
        {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature - Human Wizard", 
         "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to its mana cost.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": ["flash"], "set": "DOM"},
        {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature - Human Wizard", 
         "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to its mana cost.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": ["flash"], "set": "DOM"},
        {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature - Human Wizard", 
         "oracle_text": "Flash\nWhen Snapcaster Mage enters the battlefield, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to its mana cost.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": ["flash"], "set": "DOM"},
    ]
    deck.extend(creatures)
    
    # Counterspells (28) - heavy control
    control = [
        # 4x Counterspell
        {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "A25"},
        {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "A25"},
        {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "A25"},
        {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "A25"},
        
        # 4x Mana Leak
        {"name": "Mana Leak", "mana_cost": "{1}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell unless its controller pays {3}.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "CMD"},
        {"name": "Mana Leak", "mana_cost": "{1}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell unless its controller pays {3}.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "CMD"},
        {"name": "Mana Leak", "mana_cost": "{1}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell unless its controller pays {3}.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "CMD"},
        {"name": "Mana Leak", "mana_cost": "{1}{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell unless its controller pays {3}.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "CMD"},
        
        # 4x Spell Pierce
        {"name": "Spell Pierce", "mana_cost": "{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell unless its controller pays {2}.",
         "power": None, "toughness": None, "cmc": 1, "keywords": [], "set": "ZNE"},
        {"name": "Spell Pierce", "mana_cost": "{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell unless its controller pays {2}.",
         "power": None, "toughness": None, "cmc": 1, "keywords": [], "set": "ZNE"},
        {"name": "Spell Pierce", "mana_cost": "{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell unless its controller pays {2}.",
         "power": None, "toughness": None, "cmc": 1, "keywords": [], "set": "ZNE"},
        {"name": "Spell Pierce", "mana_cost": "{U}", "type_line": "Instant",
         "oracle_text": "Counter target spell unless its controller pays {2}.",
         "power": None, "toughness": None, "cmc": 1, "keywords": [], "set": "ZNE"},
        
        # 4x Dress Down (removal)
        {"name": "Dress Down", "mana_cost": "{1}{W}", "type_line": "Instant",
         "oracle_text": "Choose one —\n• Exile target creature with mana value 2 or less.\n• Exile target creature if it's white or black.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "SNC"},
        {"name": "Dress Down", "mana_cost": "{1}{W}", "type_line": "Instant",
         "oracle_text": "Choose one —\n• Exile target creature with mana value 2 or less.\n• Exile target creature if it's white or black.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "SNC"},
        {"name": "Dress Down", "mana_cost": "{1}{W}", "type_line": "Instant",
         "oracle_text": "Choose one —\n• Exile target creature with mana value 2 or less.\n• Exile target creature if it's white or black.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "SNC"},
        {"name": "Dress Down", "mana_cost": "{1}{W}", "type_line": "Instant",
         "oracle_text": "Choose one —\n• Exile target creature with mana value 2 or less.\n• Exile target creature if it's white or black.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "SNC"},
        
        # 4x Murktide
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        {"name": "Murktide", "mana_cost": "{5}{U}{R}", "type_line": "Creature - Dragon",
         "oracle_text": "Flying\nDelve\nMurktide enters the battlefield with a number of +1/+1 counters equal to the number of instant and sorcery cards in your graveyard.",
         "power": "2", "toughness": "1", "cmc": 7, "keywords": ["flying", "delve"], "set": "MH2"},
        
        # 4x Unholy Heat (removal)
        {"name": "Unholy Heat", "mana_cost": "{B}{R}", "type_line": "Instant",
         "oracle_text": "Unholy Heat deals X damage to any target, where X is 2 plus the number of card types in your graveyard.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "MH2"},
        {"name": "Unholy Heat", "mana_cost": "{B}{R}", "type_line": "Instant",
         "oracle_text": "Unholy Heat deals X damage to any target, where X is 2 plus the number of card types in your graveyard.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "MH2"},
        {"name": "Unholy Heat", "mana_cost": "{B}{R}", "type_line": "Instant",
         "oracle_text": "Unholy Heat deals X damage to any target, where X is 2 plus the number of card types in your graveyard.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "MH2"},
        {"name": "Unholy Heat", "mana_cost": "{B}{R}", "type_line": "Instant",
         "oracle_text": "Unholy Heat deals X damage to any target, where X is 2 plus the number of card types in your graveyard.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "MH2"},
    ]
    deck.extend(control)
    
    # Lands (24)
    lands = [
        {"name": "Scalding Tarn", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Scalding Tarn: Search your library for an Island or Mountain card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZEN"},
        {"name": "Scalding Tarn", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Scalding Tarn: Search your library for an Island or Mountain card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZEN"},
        {"name": "Scalding Tarn", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Scalding Tarn: Search your library for an Island or Mountain card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZEN"},
        {"name": "Scalding Tarn", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Scalding Tarn: Search your library for an Island or Mountain card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZEN"},
        
        {"name": "Steam Vents", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "({T}: Add {U} or {R}.)\nWhenever Steam Vents enters the battlefield, you gain 1 life.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RNA"},
        {"name": "Steam Vents", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "({T}: Add {U} or {R}.)\nWhenever Steam Vents enters the battlefield, you gain 1 life.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RNA"},
        {"name": "Steam Vents", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "({T}: Add {U} or {R}.)\nWhenever Steam Vents enters the battlefield, you gain 1 life.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RNA"},
        {"name": "Steam Vents", "mana_cost": "", "type_line": "Land — Island Mountain", 
         "oracle_text": "({T}: Add {U} or {R}.)\nWhenever Steam Vents enters the battlefield, you gain 1 life.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RNA"},
        
        {"name": "Island", "mana_cost": "", "type_line": "Basic Land — Island", 
         "oracle_text": "{T}: Add {U}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Island", "mana_cost": "", "type_line": "Basic Land — Island", 
         "oracle_text": "{T}: Add {U}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Island", "mana_cost": "", "type_line": "Basic Land — Island", 
         "oracle_text": "{T}: Add {U}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Island", "mana_cost": "", "type_line": "Basic Land — Island", 
         "oracle_text": "{T}: Add {U}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        
        {"name": "Mountain", "mana_cost": "", "type_line": "Basic Land — Mountain", 
         "oracle_text": "{T}: Add {R}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Mountain", "mana_cost": "", "type_line": "Basic Land — Mountain", 
         "oracle_text": "{T}: Add {R}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Mountain", "mana_cost": "", "type_line": "Basic Land — Mountain", 
         "oracle_text": "{T}: Add {R}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Mountain", "mana_cost": "", "type_line": "Basic Land — Mountain", 
         "oracle_text": "{T}: Add {R}.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        
        {"name": "Flooded Strand", "mana_cost": "", "type_line": "Land", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for a Plains or Island card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
        {"name": "Flooded Strand", "mana_cost": "", "type_line": "Land", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for a Plains or Island card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
        {"name": "Flooded Strand", "mana_cost": "", "type_line": "Land", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for a Plains or Island card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
        {"name": "Flooded Strand", "mana_cost": "", "type_line": "Land", 
         "oracle_text": "{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for a Plains or Island card and put it onto the battlefield. Then shuffle.", 
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
    ]
    deck.extend(lands)
    
    return deck


def build_mono_green_aggro_deck() -> list[dict]:
    """Build Mono Green Aggro deck list (March 2026 meta).
    
    Standard green ramp and beatdown deck with creatures and mana ramp.
    """
    deck = []
    
    # Creatures (28)
    creatures = [
        # 4x Llanowar Elves (green 1-drop)
        {"name": "Llanowar Elves", "mana_cost": "{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "{T}: Add {G}.",
         "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "M19"},
        {"name": "Llanowar Elves", "mana_cost": "{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "{T}: Add {G}.",
         "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "M19"},
        {"name": "Llanowar Elves", "mana_cost": "{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "{T}: Add {G}.",
         "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "M19"},
        {"name": "Llanowar Elves", "mana_cost": "{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "{T}: Add {G}.",
         "power": "1", "toughness": "1", "cmc": 1, "keywords": [], "set": "M19"},
        
        # 4x Beast Whisperer (green 2-drop)
        {"name": "Beast Whisperer", "mana_cost": "{1}{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "Whenever you cast a creature spell, draw a card.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": [], "set": "GRN"},
        {"name": "Beast Whisperer", "mana_cost": "{1}{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "Whenever you cast a creature spell, draw a card.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": [], "set": "GRN"},
        {"name": "Beast Whisperer", "mana_cost": "{1}{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "Whenever you cast a creature spell, draw a card.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": [], "set": "GRN"},
        {"name": "Beast Whisperer", "mana_cost": "{1}{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "Whenever you cast a creature spell, draw a card.",
         "power": "2", "toughness": "1", "cmc": 2, "keywords": [], "set": "GRN"},
        
        # 4x Elvish Archdruid (lord)
        {"name": "Elvish Archdruid", "mana_cost": "{1}{G}{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "Elvish Archdruid gets +1/+1 for each other Elf you control.\n{G}: Target creature you control gets +1/+2 until end of turn.",
         "power": "2", "toughness": "2", "cmc": 3, "keywords": [], "set": "M13"},
        {"name": "Elvish Archdruid", "mana_cost": "{1}{G}{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "Elvish Archdruid gets +1/+1 for each other Elf you control.\n{G}: Target creature you control gets +1/+2 until end of turn.",
         "power": "2", "toughness": "2", "cmc": 3, "keywords": [], "set": "M13"},
        {"name": "Elvish Archdruid", "mana_cost": "{1}{G}{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "Elvish Archdruid gets +1/+1 for each other Elf you control.\n{G}: Target creature you control gets +1/+2 until end of turn.",
         "power": "2", "toughness": "2", "cmc": 3, "keywords": [], "set": "M13"},
        {"name": "Elvish Archdruid", "mana_cost": "{1}{G}{G}", "type_line": "Creature - Elf Druid",
         "oracle_text": "Elvish Archdruid gets +1/+1 for each other Elf you control.\n{G}: Target creature you control gets +1/+2 until end of turn.",
         "power": "2", "toughness": "2", "cmc": 3, "keywords": [], "set": "M13"},
        
        # 4x Imperious Perfect (elf lord)
        {"name": "Imperious Perfect", "mana_cost": "{1}{G}{G}", "type_line": "Creature - Elf Warrior",
         "oracle_text": "Other Elf creatures get +1/+1.\n{G}{G}: Create a 1/1 green Elf Warrior token.",
         "power": "2", "toughness": "2", "cmc": 3, "keywords": [], "set": "LOR"},
        {"name": "Imperious Perfect", "mana_cost": "{1}{G}{G}", "type_line": "Creature - Elf Warrior",
         "oracle_text": "Other Elf creatures get +1/+1.\n{G}{G}: Create a 1/1 green Elf Warrior token.",
         "power": "2", "toughness": "2", "cmc": 3, "keywords": [], "set": "LOR"},
        {"name": "Imperious Perfect", "mana_cost": "{1}{G}{G}", "type_line": "Creature - Elf Warrior",
         "oracle_text": "Other Elf creatures get +1/+1.\n{G}{G}: Create a 1/1 green Elf Warrior token.",
         "power": "2", "toughness": "2", "cmc": 3, "keywords": [], "set": "LOR"},
        {"name": "Imperious Perfect", "mana_cost": "{1}{G}{G}", "type_line": "Creature - Elf Warrior",
         "oracle_text": "Other Elf creatures get +1/+1.\n{G}{G}: Create a 1/1 green Elf Warrior token.",
         "power": "2", "toughness": "2", "cmc": 3, "keywords": [], "set": "LOR"},
        
        # 4x Craterhoof Behemoth (big finisher)
        {"name": "Craterhoof Behemoth", "mana_cost": "{5}{G}{G}", "type_line": "Creature - Beast",
         "oracle_text": "When Craterhoof Behemoth enters the battlefield, creatures you control gain forestwalk and +X/+X until end of turn, where X is the number of creatures you control.",
         "power": "5", "toughness": "5", "cmc": 7, "keywords": [], "set": "AVR"},
        {"name": "Craterhoof Behemoth", "mana_cost": "{5}{G}{G}", "type_line": "Creature - Beast",
         "oracle_text": "When Craterhoof Behemoth enters the battlefield, creatures you control gain forestwalk and +X/+X until end of turn, where X is the number of creatures you control.",
         "power": "5", "toughness": "5", "cmc": 7, "keywords": [], "set": "AVR"},
        {"name": "Craterhoof Behemoth", "mana_cost": "{5}{G}{G}", "type_line": "Creature - Beast",
         "oracle_text": "When Craterhoof Behemoth enters the battlefield, creatures you control gain forestwalk and +X/+X until end of turn, where X is the number of creatures you control.",
         "power": "5", "toughness": "5", "cmc": 7, "keywords": [], "set": "AVR"},
        {"name": "Craterhoof Behemoth", "mana_cost": "{5}{G}{G}", "type_line": "Creature - Beast",
         "oracle_text": "When Craterhoof Behemoth enters the battlefield, creatures you control gain forestwalk and +X/+X until end of turn, where X is the number of creatures you control.",
         "power": "5", "toughness": "5", "cmc": 7, "keywords": [], "set": "AVR"},
        
        # 4x Tishana, Voice of Thunder (finisher)
        {"name": "Tishana, Voice of Thunder", "mana_cost": "{4}{G}{U}", "type_line": "Legendary Creature - Merfolk Shaman",
         "oracle_text": "Tishana, Voice of Thunder has power and toughness each equal to the number of creatures you control.\nWhen Tishana enters the battlefield, draw a card for each creature you control.",
         "power": "*", "toughness": "*", "cmc": 6, "keywords": [], "set": "XLN"},
        {"name": "Tishana, Voice of Thunder", "mana_cost": "{4}{G}{U}", "type_line": "Legendary Creature - Merfolk Shaman",
         "oracle_text": "Tishana, Voice of Thunder has power and toughness each equal to the number of creatures you control.\nWhen Tishana enters the battlefield, draw a card for each creature you control.",
         "power": "*", "toughness": "*", "cmc": 6, "keywords": [], "set": "XLN"},
    ]
    deck.extend(creatures)
    
    # Spells (8)
    spells = [
        {"name": "Overrun", "mana_cost": "{2}{G}{G}", "type_line": "Sorcery",
         "oracle_text": "Creatures you control get +1/+2 and gain trample until end of turn.",
         "power": None, "toughness": None, "cmc": 4, "keywords": [], "set": "PLC"},
        {"name": "Overrun", "mana_cost": "{2}{G}{G}", "type_line": "Sorcery",
         "oracle_text": "Creatures you control get +1/+2 and gain trample until end of turn.",
         "power": None, "toughness": None, "cmc": 4, "keywords": [], "set": "PLC"},
        {"name": "Overrun", "mana_cost": "{2}{G}{G}", "type_line": "Sorcery",
         "oracle_text": "Creatures you control get +1/+2 and gain trample until end of turn.",
         "power": None, "toughness": None, "cmc": 4, "keywords": [], "set": "PLC"},
        {"name": "Overrun", "mana_cost": "{2}{G}{G}", "type_line": "Sorcery",
         "oracle_text": "Creatures you control get +1/+2 and gain trample until end of turn.",
         "power": None, "toughness": None, "cmc": 4, "keywords": [], "set": "PLC"},
        
        {"name": "Ramp", "mana_cost": "{1}{G}", "type_line": "Sorcery",
         "oracle_text": "Search your library for a land card and put it onto the battlefield tapped. Then shuffle.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "M20"},
        {"name": "Ramp", "mana_cost": "{1}{G}", "type_line": "Sorcery",
         "oracle_text": "Search your library for a land card and put it onto the battlefield tapped. Then shuffle.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "M20"},
        {"name": "Ramp", "mana_cost": "{1}{G}", "type_line": "Sorcery",
         "oracle_text": "Search your library for a land card and put it onto the battlefield tapped. Then shuffle.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "M20"},
        {"name": "Ramp", "mana_cost": "{1}{G}", "type_line": "Sorcery",
         "oracle_text": "Search your library for a land card and put it onto the battlefield tapped. Then shuffle.",
         "power": None, "toughness": None, "cmc": 2, "keywords": [], "set": "M20"},
    ]
    deck.extend(spells)
    
    # Lands (24)
    lands = [
        {"name": "Forest", "mana_cost": "", "type_line": "Basic Land — Forest",
         "oracle_text": "{T}: Add {G}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Forest", "mana_cost": "", "type_line": "Basic Land — Forest",
         "oracle_text": "{T}: Add {G}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Forest", "mana_cost": "", "type_line": "Basic Land — Forest",
         "oracle_text": "{T}: Add {G}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Forest", "mana_cost": "", "type_line": "Basic Land — Forest",
         "oracle_text": "{T}: Add {G}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Forest", "mana_cost": "", "type_line": "Basic Land — Forest",
         "oracle_text": "{T}: Add {G}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Forest", "mana_cost": "", "type_line": "Basic Land — Forest",
         "oracle_text": "{T}: Add {G}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Forest", "mana_cost": "", "type_line": "Basic Land — Forest",
         "oracle_text": "{T}: Add {G}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        {"name": "Forest", "mana_cost": "", "type_line": "Basic Land — Forest",
         "oracle_text": "{T}: Add {G}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "ZNE"},
        
        {"name": "Botanical Sanctum", "mana_cost": "", "type_line": "Land",
         "oracle_text": "{T}: Add {G} or {U}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KLD"},
        {"name": "Botanical Sanctum", "mana_cost": "", "type_line": "Land",
         "oracle_text": "{T}: Add {G} or {U}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KLD"},
        {"name": "Botanical Sanctum", "mana_cost": "", "type_line": "Land",
         "oracle_text": "{T}: Add {G} or {U}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KLD"},
        {"name": "Botanical Sanctum", "mana_cost": "", "type_line": "Land",
         "oracle_text": "{T}: Add {G} or {U}.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KLD"},
        
        {"name": "Overgrown Tomb", "mana_cost": "", "type_line": "Land — Swamp Forest",
         "oracle_text": "({T}: Add {B} or {G}.)\nAs Overgrown Tomb enters the battlefield, you may pay 2 life. If you don't, Overgrown Tomb enters the battlefield tapped.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RTR"},
        {"name": "Overgrown Tomb", "mana_cost": "", "type_line": "Land — Swamp Forest",
         "oracle_text": "({T}: Add {B} or {G}.)\nAs Overgrown Tomb enters the battlefield, you may pay 2 life. If you don't, Overgrown Tomb enters the battlefield tapped.",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "RTR"},
        
        {"name": "Godless Shrine", "mana_cost": "", "type_line": "Land — Plains Swamp",
         "oracle_text": "({T}: Add {W} or {B}.)",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
        {"name": "Godless Shrine", "mana_cost": "", "type_line": "Land — Plains Swamp",
         "oracle_text": "({T}: Add {W} or {B}.)",
         "power": None, "toughness": None, "cmc": 0, "keywords": [], "set": "KTK"},
    ]
    deck.extend(lands)
    
    return deck


def run_meta_game():
    """Run a game between two meta decks."""
    print("\n" + "="*70)
    print("STANDARD META GAME - UR Aggro vs Izzet Control")
    print("="*70)
    print("\nBuilding meta decks from March 2026 standard...\n")
    
    # Create agents with appropriate strategies
    aggro_agent = AgentGamePlayer("ur_aggro_deck", Strategy.AGGRESSIVE, knowledge_graph=None)
    control_agent = AgentGamePlayer("izzet_control_deck", Strategy.CONTROL, knowledge_graph=None)
    
    print(f"  {aggro_agent.player_id} ({aggro_agent.strategy.value})")
    print(f"  {control_agent.player_id} ({control_agent.strategy.value})")
    
    # Create and run game
    sim = GameSimulator(aggro_agent, control_agent, max_turns=20)
    sim.setup_game("meta_game_1")
    
    print("\nStarting game...\n")
    
    result = sim.run_game()
    
    # Display results
    print("\n" + "="*70)
    print("GAME RESULT")
    print("="*70)
    print(f"\nWinner: {result.value}")
    print(f"Final Turn: {sim.game.turn_number}")
    print(f"UR Aggro Life Total: {sim.game.players[0].life_total}")
    print(f"Izzet Control Life Total: {sim.game.players[1].life_total}")
    
    print("\n" + sim.get_game_summary())


def run_meta_tournament():
    """Run multiple meta games and show statistics with LLM-powered agents."""
    print("\n" + "="*70)
    print("STANDARD META TOURNAMENT - March 2026")
    print("LLM-POWERED AGENTS (Ollama - Phi Model)")
    print("="*70)
    
    # Check if Ollama is available with phi model
    from src.engine.llm_orchestration import OllamaConnector
    ollama = OllamaConnector(model="gemma4:2b")  # Smallest Gemma 4 for limited hardware
    has_llm = ollama.is_available
    llm_status = f"[OLLAMA ENABLED - phi]" if has_llm else "[OLLAMA DISABLED - Using Heuristic Play]"
    print(f"\nAgent Mode: {llm_status}")
    if has_llm:
        print(f"Ollama URL: {ollama.base_url}")
        print(f"Model: phi (2.7B parameters - fast & lightweight)")
    print()
    
    # Track matchup results
    matchup_results = {}
    
    matchups = [
        ("UR Aggro", Strategy.AGGRESSIVE, "Izzet Control", Strategy.CONTROL, 2),
        ("UR Aggro", Strategy.AGGRESSIVE, "Mono Green Aggro", Strategy.AGGRESSIVE, 2),
        ("Izzet Control", Strategy.CONTROL, "Mono Green Aggro", Strategy.AGGRESSIVE, 1),
    ]
    
    total_games = 0
    
    for deck1_name, strategy1, deck2_name, strategy2, num_games in matchups:
        print(f"\n{'='*70}")
        print(f"MATCHUP: {deck1_name} vs {deck2_name}")
        print(f"{'='*70}")
        print(f"Playing {num_games} games...\n")
        
        deck1_wins = 0
        deck2_wins = 0
        draws = 0
        
        for game_num in range(1, num_games + 1):
            # Create LLM-powered agents
            player1 = AgentGamePlayer(deck1_name, strategy1, knowledge_graph=None, use_llm=has_llm)
            player2 = AgentGamePlayer(deck2_name, strategy2, knowledge_graph=None, use_llm=has_llm)
            
            sim = GameSimulator(player1, player2, max_turns=20)
            sim.setup_game(f"{deck1_name.replace(' ', '_')}_vs_{deck2_name.replace(' ', '_')}_game{game_num}")
            
            result = sim.run_game()
            total_games += 1
            
            # Parse result
            if result.value == "draw":
                draws += 1
                print(f"  Game {game_num}: DRAW (Turn {sim.game.turn_number})")
            elif sim.game.players[0].life_total <= 0:
                deck2_wins += 1
                print(f"  Game {game_num}: {deck2_name} wins (Turn {sim.game.turn_number})")
            else:
                deck1_wins += 1
                print(f"  Game {game_num}: {deck1_name} wins (Turn {sim.game.turn_number})")
        
        matchup_key = f"{deck1_name} vs {deck2_name}"
        matchup_results[matchup_key] = {
            "deck1": deck1_name,
            "deck2": deck2_name,
            "deck1_wins": deck1_wins,
            "deck2_wins": deck2_wins,
            "draws": draws,
            "total": num_games
        }
        
        # Display matchup summary
        print(f"\n  Result: {deck1_name} {deck1_wins}W-{deck2_wins}L-{draws}D")
        if deck1_wins + deck2_wins > 0:
            deck1_wr = (deck1_wins / (deck1_wins + deck2_wins)) * 100
            print(f"  Win Rate: {deck1_name} {deck1_wr:.1f}% vs {deck2_name} {100-deck1_wr:.1f}%")
    
    # Display tournament summary
    print("\n" + "="*70)
    print("TOURNAMENT SUMMARY")
    print("="*70)
    print(f"\nTotal Games Played: {total_games}")
    print(f"\nMatchup Results:")
    
    for matchup_key, stats in matchup_results.items():
        print(f"\n  {matchup_key}:")
        print(f"    {stats['deck1']}: {stats['deck1_wins']}W - {stats['deck2_wins']}L - {stats['draws']}D")
        print(f"    {stats['deck2']}: {stats['deck2_wins']}W - {stats['deck1_wins']}L - {stats['draws']}D")


if __name__ == "__main__":
    run_meta_tournament()
