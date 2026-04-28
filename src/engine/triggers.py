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
    if "attack" in oracle and ("whenever" in oracle):
        effect = _extract_effect_attack(card.oracle_text)
        if effect:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.ATTACKS,
                description=effect,
            ))
    
    # When/Whenever this creature dies
    if ("die" in oracle or "death" in oracle) and ("when" in oracle or "whenever" in oracle):
        effect = _extract_effect_death(card.oracle_text)
        if effect:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.CREATURE_DIES,
                description=effect,
            ))
    
    # Whenever you cast a spell
    if "cast" in oracle and ("when" in oracle or "whenever" in oracle):
        effect = _extract_effect_cast(card.oracle_text)
        if effect:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.CAST,
                description=effect,
            ))
    
    # Whenever you gain life
    if ("gain" in oracle and "life" in oracle) and ("when" in oracle or "whenever" in oracle):
        triggers.append(Trigger(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            trigger_type=TriggerType.LIFE_GAIN,
            description="whenever you gain life",
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


def check_attack_triggers(state: GameState, attacking_card: CardInstance) -> list[Trigger]:
    """Check which creatures have attack triggers.
    
    When a creature attacks, check if it has any "whenever attacks" triggers.
    """
    triggered = []
    
    # Check the attacking creature itself
    own_triggers = parse_triggers(attacking_card)
    attack_triggers = [t for t in own_triggers if t.trigger_type == TriggerType.ATTACKS]
    triggered.extend(attack_triggers)
    
    # Check other permanents for "whenever a creature attacks" triggers
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        if card.instance_id == attacking_card.instance_id:
            continue
        
        if "whenever a creature attack" in card.oracle_text.lower():
            effect = _extract_effect_attack(card.oracle_text)
            if not effect:
                effect = "whenever a creature attacks"
            triggered.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.ATTACKS,
                description=effect,
            ))
    
    return triggered


def check_death_triggers(state: GameState, dying_card: CardInstance) -> list[Trigger]:
    """Check which creatures have death triggers.
    
    When a creature dies, check if it has any "when dies" triggers,
    and check other permanents for "whenever a creature dies" triggers.
    """
    triggered = []
    
    # Check the dying creature itself
    own_triggers = parse_triggers(dying_card)
    death_triggers = [t for t in own_triggers if t.trigger_type == TriggerType.CREATURE_DIES]
    triggered.extend(death_triggers)
    
    # Check other permanents for "whenever a creature dies" triggers
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        if card.instance_id == dying_card.instance_id:
            continue
        
        oracle_lower = card.oracle_text.lower()
        if ("whenever a creature die" in oracle_lower or 
            "whenever another creature die" in oracle_lower):
            effect = _extract_effect_death(card.oracle_text)
            if not effect:
                effect = "whenever a creature dies"
            triggered.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.CREATURE_DIES,
                description=effect,
            ))
    
    return triggered


def check_cast_triggers(state: GameState, casting_player_id: str, spell_card: CardInstance) -> list[Trigger]:
    """Check which permanents have cast triggers.
    
    When a spell is cast, check all permanents (controlled by the casting player)
    for "whenever you cast a spell" triggers.
    """
    triggered = []
    
    # Check permanents controlled by the casting player
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        if card.controller_id != casting_player_id:
            continue
        
        oracle_lower = card.oracle_text.lower()
        if "whenever you cast" in oracle_lower:
            # Also check for spell type restrictions (instant, spell, creature, etc.)
            effect = _extract_effect_cast(card.oracle_text)
            if not effect:
                effect = "whenever you cast a spell"
            triggered.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.CAST,
                description=effect,
            ))
    
    return triggered


def check_landfall_triggers(state: GameState, controller_id: str, land_card: CardInstance) -> list[Trigger]:
    """Check for landfall triggers when a land enters the battlefield.

    CR 702.124: "Landfall — Whenever a land enters the battlefield under
    your control, ..."  We detect any permanent the casting player controls
    whose oracle text mentions ``landfall`` or the explicit phrase
    "whenever a land enters the battlefield under your control".
    """
    triggered: list[Trigger] = []
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        if card.controller_id != controller_id:
            continue
        oracle_lower = card.oracle_text.lower()
        if "landfall" not in oracle_lower and \
                "whenever a land enters the battlefield under your control" not in oracle_lower:
            continue
        # Effect = text after the landfall keyword colon, or after the comma
        effect = _extract_effect_landfall(card.oracle_text) or "landfall trigger"
        triggered.append(Trigger(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            trigger_type=TriggerType.LANDFALL,
            description=f"{card.name}: {effect}",
        ))
    return triggered


def _extract_effect_landfall(oracle_text: str) -> str | None:
    """Pull the effect after 'Landfall —' or after the trigger comma."""
    m = re.search(r"landfall\s*[—\-:]\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(
        r"whenever a land enters the battlefield under your control,?\s*(.+?)(?:\.|$)",
        oracle_text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


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
    """Extract attack trigger effect.
    
    E.g., "Whenever ~ attacks, you gain 1 life" → "you gain 1 life"
    """
    if "whenever" not in oracle_text.lower():
        return None
    
    match = re.search(r"whenever[^,]*attack[^,]*,\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None


def _extract_effect_death(oracle_text: str) -> str | None:
    """Extract death trigger effect.
    
    E.g., "When ~ dies, draw a card" → "draw a card"
    """
    match = re.search(r"when[^,]*die[^,]*,\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Also check "whenever" format
    match = re.search(r"whenever[^,]*die[^,]*,\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None


def _extract_effect_cast(oracle_text: str) -> str | None:
    """Extract cast trigger effect.
    
    E.g., "Whenever you cast a spell, draw a card" → "draw a card"
    """
    match = re.search(r"whenever[^,]*cast[^,]*,\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Also check "when" format
    match = re.search(r"when[^,]*cast[^,]*,\s*(.+?)(?:\.|$)", oracle_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None



def resolve_trigger(state: GameState, trigger: Trigger) -> GameState:
    """Resolve a triggered ability's effect.
    
    Supporting effects:
    - Draw a card / cards
    - Deal damage (to opponent or player)
    - Gain life
    - Create tokens (referenced as "create X tokens")
    - Destroy / Tap permanents
    
    Trigger types:
    - ENTERS_BATTLEFIELD: Creature/permanent enter
    - ATTACKS: Creature attacks
    - CREATURE_DIES: Creature dies  
    - CAST: Spell cast
    - LIFE_GAIN: Player gains life
    - COMBAT_DAMAGE: Combat damage dealt
    """
    description = trigger.description.lower()
    controller = next((p for p in state.players if p.player_id == trigger.controller_id), None)
    
    if not controller:
        return state
    
    opponent = next((p for p in state.players if p.player_id != trigger.controller_id), None)
    
    # Draw effect: "draw a card", "draw 2 cards", etc.
    if "draw" in description and "card" in description:
        match = re.search(r"draw (\d+) cards?", description)
        count = int(match.group(1)) if match else 1
        
        library = [c for c in state.cards if c.zone == Zone.LIBRARY and c.owner_id == controller.player_id]
        for i in range(count):
            if library:
                from src.engine.zones import move_card
                card_to_draw = library.pop(0)
                state = move_card(state, card_to_draw.instance_id, Zone.LIBRARY, Zone.HAND, controller.player_id)
        
        state.log(f"[Trigger] {controller.name} draws {count} card{'s' if count != 1 else ''}")
    
    # Gain life: "gain 1 life", "gain 5 life", etc.
    if "gain" in description and "life" in description:
        match = re.search(r"gain (\d+) life", description)
        amount = int(match.group(1)) if match else 1
        controller.life_total += amount
        state.log(f"[Trigger] {controller.name} gains {amount} life")
    
    # Opponent loses life / Deal damage: "deal 1 damage", "opponent loses 1 life", etc.
    if "deal" in description and "damage" in description:
        match = re.search(r"deal (\d+) damage", description)
        amount = int(match.group(1)) if match else 1
        if opponent:
            opponent.life_total -= amount
            state.log(f"[Trigger] {opponent.name} takes {amount} damage")
    
    if "opponent lose" in description and "life" in description:
        match = re.search(r"lose (\d+) life", description)
        amount = int(match.group(1)) if match else 1
        if opponent:
            opponent.life_total -= amount
            state.log(f"[Trigger] {opponent.name} loses {amount} life")
    
    # Discard effect: "discard a card", "discard 2 cards", etc.
    if "discard" in description and "card" in description:
        match = re.search(r"discard (\d+) cards?", description)
        count = int(match.group(1)) if match else 1
        
        hand = [c for c in state.cards if c.zone == Zone.HAND and c.owner_id == controller.player_id]
        for i in range(count):
            if hand:
                from src.engine.zones import move_card
                # Discard first card in hand
                card_to_discard = hand.pop(0)
                state = move_card(state, card_to_discard.instance_id, Zone.HAND, Zone.GRAVEYARD, controller.player_id)
        
        state.log(f"[Trigger] {controller.name} discards {count} card{'s' if count != 1 else ''}")
    
    # Create tokens: "create a 1/1 token", "create 2 2/2 tokens", etc.
    if "create" in description and "token" in description:
        from src.engine import tokens as _tok
        desc_l = description.lower()
        # Count: "a"/"an" → 1, otherwise digit.
        cm = re.search(r"create (a|an|\d+)", desc_l)
        count = 1
        if cm:
            tok = cm.group(1)
            count = 1 if tok in ("a", "an") else int(tok)
        # Predefined token types first.
        spawned = False
        if "treasure" in desc_l:
            for _ in range(count):
                _tok.create_treasure_token(state, controller.player_id)
            spawned = True
        elif "food" in desc_l:
            for _ in range(count):
                _tok.create_food_token(state, controller.player_id)
            spawned = True
        elif "clue" in desc_l:
            for _ in range(count):
                _tok.create_clue_token(state, controller.player_id)
            spawned = True
        elif "blood" in desc_l:
            for _ in range(count):
                _tok.create_blood_token(state, controller.player_id)
            spawned = True
        if not spawned:
            # Generic creature token: parse "X/Y <colors> <subtypes> creature token".
            ptm = re.search(r"(\d+)/(\d+)\s+([\w\s]*?)\s*(?:creature\s+)?token", desc_l)
            if ptm:
                p = int(ptm.group(1))
                t = int(ptm.group(2))
                middle = ptm.group(3).strip()
                # Extract color words and remaining as subtypes.
                color_map = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G"}
                colors = [color_map[w] for w in middle.split() if w in color_map]
                subtypes = [w.title() for w in middle.split() if w not in color_map and w]
                if not subtypes:
                    subtypes = ["Spirit"]  # fallback generic
                for _ in range(count):
                    _tok.create_creature_token(
                        state, controller.player_id,
                        power=p, toughness=t,
                        subtypes=subtypes, colors=colors or None,
                    )
                spawned = True
        if spawned:
            state.log(f"[Trigger] {controller.name} creates {count} token(s)")    
    # Tap target permanent: "tap a creature", "tap target land", etc.
    if "tap" in description and "target" in description:
        # Find target permanent on battlefield
        for card in state.cards:
            if card.zone == Zone.BATTLEFIELD and card.controller_id != controller.player_id:
                card.tapped = True
                state.log(f"[Trigger] {card.name} is tapped")
                break
    
    return state
