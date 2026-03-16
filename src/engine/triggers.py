"""
Triggered abilities system — handles "when X happens, do Y" mechanics.

Triggered abilities are fundamentally different from activated abilities:
- Activated: Controlled by player, can be used any time you have priority
- Triggered: Automatic, happen when a condition is met, go on the stack

This module handles parsing, checking, and creating triggered abilities.

Reference: MTG CR 603 — Handling Triggered Abilities
"""

from __future__ import annotations

import re
from src.engine.game_state import Trigger, TriggerType, CardInstance, GameState, Zone


def parse_triggers(card: CardInstance) -> list[Trigger]:
    """Extract triggered abilities from a card's oracle text.
    
    Returns list of potential triggers this card can have.
    Some triggers may not fire in a given game (e.g., never attack).
    """
    oracle = card.oracle_text.lower()
    if not oracle:
        return []
    
    triggers = []
    
    # Enter the Battlefield / When X enters (most common early trigger)
    # Matches: "When X enters", "When X enters the battlefield", "enters with", etc.
    if ("enter" in oracle and ("when" in oracle or "whenever" in oracle)):
        # Or more lenient: "whenever this creature enters the battlefield"
        if "when" in oracle and "enter" in oracle:
            effect = _extract_effect_from_etb(card.oracle_text)
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.ENTERS_BATTLEFIELD,
                description=effect,
            ))
    
    # Whenever this creature attacks
    if "attack" in oracle and ("whenever" in oracle or card.is_creature()):
        effect = _extract_effect_attack(card.oracle_text)
        if effect:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.ATTACKS,
                description=effect,
            ))
    
    # When this creature dies
    if "die" in oracle or "death" in oracle:
        effect = _extract_effect_death(card.oracle_text)
        if effect:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.CREATURE_DIES,
                description=effect,
            ))
    
    # Whenever you gain life
    if "gain" in oracle and "life" in oracle:
        triggers.append(Trigger(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            trigger_type=TriggerType.LIFE_GAIN,
            description="gain life trigger",
        ))
    
    return triggers


def check_enters_battlefield_triggers(state: GameState, entering_card: CardInstance) -> list[Trigger]:
    """Check which cards on battlefield have ETB triggers.
    
    When a creature enters, check all permanents (including the new one)
    for "enters the battlefield" triggers.
    """
    triggered = []
    
    # Check all permanents for ETB abilities
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        
        # Skip the card that just entered (it has its own checks)
        if card.instance_id == entering_card.instance_id:
            continue
        
        # Check if this card has an ETB trigger that cares about others entering
        # (e.g., "whenever another creature enters")
        if "another creature enter" in card.oracle_text.lower():
            effect = _extract_effect_from_etb(card.oracle_text)
            triggered.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.ENTERS_BATTLEFIELD,
                description=effect or "whenever another creature enters",
            ))
        
        # Check for "whenever a creature enters" (triggers on all creatures)
        if "whenever a creature enter" in card.oracle_text.lower():
            effect = _extract_effect_from_etb(card.oracle_text)
            triggered.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.ENTERS_BATTLEFIELD,
                description=effect or "whenever a creature enters",
            ))
    
    # Also check the entering card itself for ETB triggers
    own_triggers = parse_triggers(entering_card)
    etb_triggers = [t for t in own_triggers if t.trigger_type == TriggerType.ENTERS_BATTLEFIELD]
    triggered.extend(etb_triggers)
    
    return triggered


def _extract_effect_from_etb(oracle_text: str) -> str:
    """Extract the effect part from an ETB trigger.
    
    E.g., "When ~ enters, draw a card" → "draw a card"
    """
    # Simple heuristic: everything after "enter" and ","
    match = re.search(r"enter[^,]*,\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: just return the whole text
    return oracle_text


def _extract_effect_attack(oracle_text: str) -> str | None:
    """Extract attack trigger effect."""
    if "whenever" not in oracle_text.lower():
        return None
    
    match = re.search(r"whenever[^,]*attack[^,]*,\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None


def _extract_effect_death(oracle_text: str) -> str | None:
    """Extract death trigger effect."""
    match = re.search(r"when[^,]*die[^,]*,\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None


def resolve_trigger(state: GameState, trigger: Trigger) -> GameState:
    """Resolve a triggered ability's effect.
    
    Most common effects:
    - Draw a card
    - Deal damage
    - Gain life
    - Create tokens
    - Destroy/tap a permanent
    
    For Phase 3, we'll implement the most common ones.
    """
    description = trigger.description.lower()
    controller = next((p for p in state.players if p.player_id == trigger.controller_id), None)
    
    if not controller:
        return state
    
    # Draw a card
    if "draw" in description and "card" in description:
        library = [c for c in state.cards if c.zone == Zone.LIBRARY and c.owner_id == controller.player_id]
        if library:
            from src.engine.zones import move_card
            card_to_draw = library[0]
            state = move_card(state, card_to_draw.instance_id, Zone.LIBRARY, Zone.HAND, controller.player_id)
            state.log(f"Trigger: {controller.name} draws a card")
    
    # Gain life
    if "gain" in description and "life" in description:
        # Parse amount: "gain 1 life", "gain 5 life", etc.
        match = re.search(r"gain (\d+) life", description)
        if match:
            amount = int(match.group(1))
            controller.life_total += amount
            state.log(f"Trigger: {controller.name} gained {amount} life")
        else:
            # Default to 1 if not specified
            controller.life_total += 1
            state.log(f"Trigger: {controller.name} gained 1 life")
    
    # Opponent loses life / deal damage
    if "deal" in description and "damage" in description:
        # Simple: "deal 1 damage"
        match = re.search(r"deal (\d+) damage", description)
        if match:
            amount = int(match.group(1))
            opponent = next((p for p in state.players if p.player_id != controller.player_id), None)
            if opponent:
                opponent.life_total -= amount
                state.log(f"Trigger: Opponent takes {amount} damage")
    
    return state
