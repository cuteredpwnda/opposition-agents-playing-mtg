"""
Triggered abilities system — "when", "whenever" effects that detect state changes.

Design patterns from mtg-python-engine (MIT) https://github.com/wanqizhu/mtg-python-engine
Reference: MTG rules 603 (Triggered Abilities)

Current Phase 1: Basic ETB (enters the battlefield) triggers.
Future: Attack triggers, death triggers, turn-begin triggers, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from src.engine.game_state import GameState, CardInstance, Zone


class TriggerEvent(Enum):
    """Standard trigger events."""
    ENTERS_BATTLEFIELD = auto()
    CREATURE_ATTACKS = auto()
    CREATURE_BLOCKS = auto()
    CREATURE_DIES = auto()
    SPELL_CAST = auto()
    TURN_BEGINS = auto()
    TURN_ENDS = auto()
    DAMAGE_DEALT = auto()


@dataclass
class TriggerDetail:
    """A triggered ability on a permanent (e.g. "When ~ enters, draw a card")."""
    source_card_id: str
    event: TriggerEvent
    description: str  # Human-readable: "When ~ enters, draw a card"
    effect_fn: callable  # Function to execute when triggered


def detect_enters_battlefield(prev_state: GameState, new_state: GameState) -> list[TriggerDetail]:
    """Detect creatures/permanents that entered the battlefield this turn."""
    triggers = []
    
    # Build set of card IDs that were on BF before
    old_bf_ids = {c.instance_id for c in prev_state.cards if c.zone == Zone.BATTLEFIELD}
    new_bf_ids = {c.instance_id for c in new_state.cards if c.zone == Zone.BATTLEFIELD}
    
    # Cards that entered = new cards in BF that weren't before
    entered_ids = new_bf_ids - old_bf_ids
    
    for card in new_state.cards:
        if card.instance_id not in entered_ids:
            continue
        
        # Check for ETB abilities (hardcoded for Phase 1)
        if "Mulldrifter" in card.name:
            triggers.append(
                TriggerDetail(
                    source_card_id=card.instance_id,
                    event=TriggerEvent.ENTERS_BATTLEFIELD,
                    description=f"When {card.name} enters the battlefield, draw a card",
                    effect_fn=lambda gs, cid: _draw_card(gs, cid),
                )
            )
    
    return triggers


def resolve_triggers(game_state: GameState, triggers: list[TriggerDetail]) -> GameState:
    """Execute triggered abilities in APNAP order (all triggers at once for Phase 1)."""
    for trigger in triggers:
        game_state.log(f"[TRIGGER] {trigger.description}")
        game_state = trigger.effect_fn(game_state, trigger.source_card_id)
    
    return game_state


# Standard effects
def _draw_card(game_state: GameState, card_instance_id: str) -> GameState:
    """Draw a card for the owner of source card."""
    source_card = next(
        (c for c in game_state.cards if c.instance_id == card_instance_id), None
    )
    if not source_card:
        return game_state
    
    library = game_state.library.get(source_card.owner_id, [])
    if not library:
        return game_state
    
    drawn = library.pop(0)
    drawn.zone = Zone.HAND
    game_state.log(f"  → {source_card.owner_id} draws {drawn.name}")
    
    return game_state
