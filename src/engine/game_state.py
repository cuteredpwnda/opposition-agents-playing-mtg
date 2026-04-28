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


class TriggerType(str, Enum):
    """Types of triggers (when X happens, ability triggers)."""
    
    ENTERS_BATTLEFIELD = "enters_battlefield"
    LEAVES_BATTLEFIELD = "leaves_battlefield"
    ATTACKS = "attacks"
    BLOCKS = "blocks"
    CAST = "cast"
    DEALT_DAMAGE = "dealt_damage"
    CREATURE_DIES = "creature_dies"
    LIFE_GAIN = "life_gain"
    COMBAT_DAMAGE = "combat_damage"
    LANDFALL = "landfall"
    UPKEEP = "upkeep"
    END_STEP = "end_step"
    BEGINNING_OF_COMBAT = "beginning_of_combat"


@dataclass
class Trigger:
    """A triggered ability awaiting resolution."""
    
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_card_id: str = ""  # Which card is triggering
    controller_id: str = ""  # Who controls the trigger
    trigger_type: TriggerType = TriggerType.ENTERS_BATTLEFIELD
    description: str = ""  # "draw a card", "deal 1 damage", etc.
    on_stack: bool = False  # Whether this trigger has been put on stack


@dataclass
class Ability:
    """An activated ability on a permanent that can be controlled by player."""
    
    ability_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_card_id: str = ""  # Which card has this ability
    controller_id: str = ""  # Who controls the ability
    cost: str = ""  # Mana cost: "{T}", "{2}{U}", "{1}{B}", etc.
    effect: str = ""  # "Add {R}", "Draw a card", "Deal 1 damage to target", etc.
    can_use_any_time: bool = False  # True for mana abilities (can use anytime)
    description: str = ""  # Full ability text for logging


@dataclass
class StaticAbility:
    """A static ability that continuously modifies game state (not activated or triggered)."""
    
    static_ability_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_card_id: str = ""  # Which card has this static ability
    controller_id: str = ""  # Who controls the static ability
    scope: str = ""  # "creatures_you_control", "all_creatures", "artifacts_you_control", etc.
    effect_type: str = ""  # "power_toughness", "keyword", "protection", "indestructible", etc.
    power_mod: int = 0  # Power modifier (e.g., +1)
    toughness_mod: int = 0  # Toughness modifier (e.g., +1)
    keywords: list[str] = field(default_factory=list)  # Keywords to add (flying, deathtouched, etc.)
    subtype_filter: str = ""  # Optional subtype restriction (e.g., "goblin", "elf")
    description: str = ""  # Full ability text for logging


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
    known_to: set[str] = field(default_factory=set)  # Which players know the identity of this card

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

    def is_artifact(self) -> bool:
        return "Artifact" in self.type_line

    @property
    def power(self) -> Optional[str]:
        return self.card_data.get("power")

    @property
    def toughness(self) -> Optional[str]:
        return self.card_data.get("toughness")

    @property
    def is_token(self) -> bool:
        """Whether this card instance is a token (CR 111.1).

        Tokens must be flagged explicitly via ``card_data['is_token']`` or
        ``card_data['token']``. We intentionally do *not* infer token-ness
        from a missing ``set`` field, because user-supplied / synthetic deck
        dictionaries (e.g., the archetype benchmark decks) routinely omit
        ``set`` and would otherwise be wiped out by CR 704.5d on the first
        state-based-action check.
        """
        return bool(
            self.card_data.get("is_token", False)
            or self.card_data.get("token", False)
        )

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

    player_id: str = ""
    name: str = ""
    life_total: int = 20  # 40 for Commander
    mana_pool: dict[str, int] = field(
        default_factory=lambda: {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
    )
    commander_tax: int = 0
    commander_damage_received: dict[str, int] = field(default_factory=dict)
    has_drawn_for_turn: bool = False
    land_plays_remaining: int = 1
    max_hand_size: int = 7
    mulligans_taken: int = 0
    passed_priority: bool = False
    is_human: bool = False
    # Player-level counters (CR 122)
    poison_counters: int = 0
    energy_counters: int = 0
    experience_counters: int = 0


@dataclass
class Action:
    """A game action chosen by a player."""

    action_type: ActionType
    player_id: str = ""
    player: str = ""
    card_instance_id: Optional[str] = None
    targets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.player_id and not self.player:
            self.player = self.player_id
        if self.player and not self.player_id:
            self.player_id = self.player


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


@dataclass(init=False)
class GameState:
    """Complete state of a game — the single source of truth."""

    game_id: str
    format: str
    turn_number: int
    active_player_index: int
    priority_player_index: int
    phase: Phase
    players: list[PlayerState]
    cards: list[CardInstance]
    stack: list[StackItem]
    triggered_abilities: list[Trigger]
    combat: Optional[CombatState]
    game_log: list[str]
    game_over: bool
    winner: Optional[str]
    commanders: dict[str, str]  # player_id -> commander card instance id

    def __init__(
        self,
        game_id: str = None,
        format: str = "standard",
        turn_number: int = 0,
        active_player_index: int = 0,
        priority_player_index: int = 0,
        phase: Phase = Phase.UNTAP,
        players=None,
        cards=None,
        stack=None,
        triggered_abilities=None,
        combat=None,
        game_log=None,
        game_over: bool = False,
        winner: Optional[str] = None,
        active_player: Optional[str] = None,
        priority_player: Optional[str] = None,
    ):
        self.game_id = game_id or str(uuid.uuid4())
        self.format = format
        self.turn_number = turn_number
        self.phase = phase
        self.players = players or []
        self.cards = cards or []
        self.stack = stack or []
        self.triggered_abilities = triggered_abilities or []
        self.combat = combat
        self.game_log = game_log or []
        self.game_over = game_over
        self.winner = winner
        self.commanders = {}

        if isinstance(self.players, dict):
            self.players = list(self.players.values())

        if active_player is not None:
            self.active_player_index = next(
                (i for i, p in enumerate(self.players) if p.player_id == active_player),
                0,
            )
        else:
            self.active_player_index = active_player_index

        if priority_player is not None:
            self.priority_player_index = next(
                (i for i, p in enumerate(self.players) if p.player_id == priority_player),
                0,
            )
        else:
            self.priority_player_index = priority_player_index

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

    def mark_card_known(self, card_instance_id: str, player_id: str) -> None:
        """Mark a specific card instance as known to a player."""
        card = next((c for c in self.cards if c.instance_id == card_instance_id), None)
        if not card:
            return
        card.known_to.add(player_id)

    def card_is_known(self, card_instance_id: str, player_id: str) -> bool:
        """Check if a player knows a given card instance."""
        card = next((c for c in self.cards if c.instance_id == card_instance_id), None)
        return bool(card and player_id in card.known_to)

    def log(self, message: str) -> None:
        self.game_log.append(message)
