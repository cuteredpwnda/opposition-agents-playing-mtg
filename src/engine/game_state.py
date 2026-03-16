"""
Core game state dataclasses.

References:
- mtg-player (MIT): Pydantic v2 game models — adapted structure
  https://github.com/theRealMarkCastillo/mtg-player
- open-mtg (MIT): Phase enum with index, deep-copyable state
  https://github.com/hlynurd/open-mtg
- Argentum Engine: Immutable state pattern (reference)
  https://github.com/wingedsheep/argentum-engine
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums (mapped to OWL individuals in mtg-ontology-v1.0.owl)
# ---------------------------------------------------------------------------


class Zone(str, Enum):
    """MTG game zones — mirrors mtg:Zone in the ontology."""

    LIBRARY = "library"
    HAND = "hand"
    BATTLEFIELD = "battlefield"
    GRAVEYARD = "graveyard"
    EXILE = "exile"
    STACK = "stack"
    COMMAND_ZONE = "command_zone"


class Phase(str, Enum):
    """Turn phases/steps — mirrors mtg:Phase in the ontology."""

    UNTAP = "untap"
    UPKEEP = "upkeep"
    DRAW = "draw"
    MAIN_1 = "main_1"
    COMBAT_BEGIN = "combat_begin"
    COMBAT_ATTACKERS = "combat_attackers"
    COMBAT_BLOCKERS = "combat_blockers"
    COMBAT_DAMAGE = "combat_damage"
    COMBAT_END = "combat_end"
    MAIN_2 = "main_2"
    END_STEP = "end_step"
    CLEANUP = "cleanup"


class Color(str, Enum):
    """Mana colors — mirrors mtg:Color in the ontology."""

    WHITE = "W"
    BLUE = "U"
    BLACK = "B"
    RED = "R"
    GREEN = "G"


class ActionType(str, Enum):
    """Types of game actions a player can take."""

    PLAY_LAND = auto()
    CAST_SPELL = auto()
    ACTIVATE_ABILITY = auto()
    DECLARE_ATTACKERS = auto()
    DECLARE_BLOCKERS = auto()
    PASS_PRIORITY = auto()
    CONCEDE = auto()
    SPECIAL_ACTION = auto()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CardInstance:
    """A specific instance of a card in a game (may differ from the oracle card)."""

    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    card_data: dict[str, Any] = field(default_factory=dict)  # Scryfall card object
    zone: Zone = Zone.LIBRARY
    owner_id: str = ""
    controller_id: str = ""
    tapped: bool = False
    counters: dict[str, int] = field(default_factory=dict)
    attached_to: Optional[str] = None
    damage_marked: int = 0
    summoning_sick: bool = True
    turn_entered: int = 0  # Track which turn creature entered (0 = pre-game)
    face_down: bool = False

    @property
    def name(self) -> str:
        return self.card_data.get("name", "Unknown")

    @property
    def oracle_text(self) -> str:
        return self.card_data.get("oracle_text", "")

    @property
    def type_line(self) -> str:
        return self.card_data.get("type_line", "")

    @property
    def mana_cost(self) -> str:
        return self.card_data.get("mana_cost", "")

    @property
    def cmc(self) -> float:
        return self.card_data.get("cmc", 0.0)

    def is_creature(self) -> bool:
        return "Creature" in self.type_line

    def is_land(self) -> bool:
        return "Land" in self.type_line

    def is_instant(self) -> bool:
        return "Instant" in self.type_line

    @property
    def power(self) -> Optional[str]:
        return self.card_data.get("power")

    @property
    def toughness(self) -> Optional[str]:
        return self.card_data.get("toughness")

    @property
    def is_token(self) -> bool:
        """Check if this card is a token (lacks a card set, o/w has token=true)."""
        return self.card_data.get("token", False) or self.card_data.get("set") is None

    @property
    def keywords(self) -> list[str]:
        """Extract keywords from oracle text and keywords field."""
        oracle_keywords = self.card_data.get("keywords", [])
        if isinstance(oracle_keywords, str):
            oracle_keywords = [oracle_keywords]
        # Also parse from oracle text
        text_keywords = []
        oracle = self.oracle_text.lower()
        keyword_list = ["flying", "flash", "lifelink", "deathtouch", "trample", 
                        "menace", "shadow", "unblockable", "vigilance", "reach"]
        for kw in keyword_list:
            if kw in oracle:
                text_keywords.append(kw)
        return list(set(oracle_keywords + text_keywords))


@dataclass
class PlayerState:
    """State of a single player in the game."""

    player_id: str
    name: str
    life_total: int = 20  # 40 for Commander
    mana_pool: dict[str, int] = field(
        default_factory=lambda: {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
    )
    commander_tax: int = 0
    commander_damage_received: dict[str, int] = field(default_factory=dict)
    has_drawn_for_turn: bool = False
    land_plays_remaining: int = 1
    passed_priority: bool = False
    is_human: bool = False


@dataclass
class Action:
    """A game action chosen by a player."""

    action_type: ActionType
    player_id: str
    card_instance_id: Optional[str] = None
    targets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StackItem:
    """An item on the stack (spell or ability)."""

    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_card_id: Optional[str] = None
    controller_id: str = ""
    targets: list[str] = field(default_factory=list)
    is_spell: bool = True  # False = ability
    card_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CombatState:
    """Tracks declared attackers & blockers during combat."""

    attackers: dict[str, str] = field(default_factory=dict)  # attacker_id -> defending_player_id
    blockers: dict[str, list[str]] = field(default_factory=dict)  # attacker_id -> [blocker_ids]
    damage_assignment_order: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class GameState:
    """Complete state of a game — the single source of truth."""

    game_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    format: str = "standard"  # "standard" | "commander"
    turn_number: int = 0
    active_player_index: int = 0
    priority_player_index: int = 0
    phase: Phase = Phase.UNTAP
    players: list[PlayerState] = field(default_factory=list)
    cards: list[CardInstance] = field(default_factory=list)
    stack: list[StackItem] = field(default_factory=list)
    combat: Optional[CombatState] = None
    game_log: list[str] = field(default_factory=list)
    game_over: bool = False
    winner: Optional[str] = None

    @property
    def active_player(self) -> PlayerState:
        return self.players[self.active_player_index]

    @property
    def priority_player(self) -> PlayerState:
        return self.players[self.priority_player_index]

    def cards_in_zone(self, player_id: str, zone: Zone) -> list[CardInstance]:
        return [
            c for c in self.cards if c.zone == zone and c.controller_id == player_id
        ]

    def log(self, message: str) -> None:
        self.game_log.append(message)
