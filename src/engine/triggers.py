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
    # Matches: "When [this/CARDNAME] enters the battlefield". We must NOT
    # treat "Whenever ANOTHER ... enters" as a self-ETB — that's a
    # cross-permanent trigger handled in check_enters_battlefield_triggers.
    # We split the oracle into clauses and look for a self-ETB clause.
    self_etb_clause = None
    for clause in re.split(r"[.\n]", card.oracle_text):
        cl = clause.lower().strip()
        if not cl or "enter" not in cl:
            continue
        if not (cl.startswith("when") or cl.startswith("whenever")):
            continue
        # Skip cross-permanent triggers ("whenever another ... enters",
        # "whenever a creature enters", etc.).
        if "another" in cl:
            continue
        if re.search(r"whenever (a|an|each|one or more) [\w\s/+\-]*?enter", cl):
            continue
        self_etb_clause = clause
        break
    if self_etb_clause is not None:
        effect = _extract_effect_from_etb(self_etb_clause)
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
        m = re.search(r"whenever (?:you|a player|an opponent)[^,]*gain[s]? (?:\d+ )?life,?\s*(.+?)(?:\.|$)",
                      card.oracle_text, re.IGNORECASE)
        if m:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.LIFE_GAIN,
                description=m.group(1).strip(),
            ))

    # At the beginning of (your) upkeep
    if "upkeep" in oracle and "beginning" in oracle:
        m = re.search(
            r"at the beginning of (?:your|each|the) [^,]*upkeep,?\s*(.+?)(?:\.|$)",
            card.oracle_text, re.IGNORECASE,
        )
        if m:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.UPKEEP,
                description=m.group(1).strip(),
            ))

    # At the beginning of (your/each) end step
    if "end step" in oracle and "beginning" in oracle:
        m = re.search(
            r"at the beginning of (?:your|each|the) (?:next |precombat )?end step,?\s*(.+?)(?:\.|$)",
            card.oracle_text, re.IGNORECASE,
        )
        if m:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.END_STEP,
                description=m.group(1).strip(),
            ))

    # Whenever ~ deals (combat) damage to a player / creature
    if ("deals" in oracle and "damage" in oracle) and "whenever" in oracle:
        m = re.search(
            r"whenever (?:this creature|[\w\s,'\-]*?) deals (?:combat )?damage to "
            r"(?:a player|an opponent|a creature|any target),?\s*(.+?)(?:\.|$)",
            card.oracle_text, re.IGNORECASE,
        )
        if m:
            triggers.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.DEALT_DAMAGE,
                description=m.group(1).strip(),
            ))

    return triggers


def check_phase_triggers(state: GameState, trigger_type: TriggerType, active_player_id: str) -> list[Trigger]:
    """Collect upkeep / end-step / beginning-of-combat triggers from all
    permanents on the battlefield. Triggers that say "your" only fire for
    the active player; "each" / "the" fire for everyone."""
    out: list[Trigger] = []
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        oracle = card.oracle_text or ""
        if "beginning" not in oracle.lower():
            continue
        own = parse_triggers(card)
        for t in own:
            if t.trigger_type != trigger_type:
                continue
            ol = oracle.lower()
            scope_match = re.search(
                r"at the beginning of (your|each|the) [^,]*"
                + ("upkeep" if trigger_type == TriggerType.UPKEEP else "end step"),
                ol,
            )
            if scope_match:
                scope = scope_match.group(1)
                if scope == "your" and card.controller_id != active_player_id:
                    continue
            out.append(t)
    return out


def check_damage_triggers(
    state: GameState, source: CardInstance, target_kind: str
) -> list[Trigger]:
    """Fire ``whenever <source> deals damage to <target_kind>`` triggers.

    target_kind is one of: 'player', 'creature', 'any'.
    """
    out: list[Trigger] = []
    own = parse_triggers(source)
    for t in own:
        if t.trigger_type != TriggerType.DEALT_DAMAGE:
            continue
        ol = source.oracle_text.lower()
        if target_kind == "player" and "to a player" not in ol \
                and "to an opponent" not in ol and "to any target" not in ol:
            continue
        if target_kind == "creature" and "to a creature" not in ol \
                and "to any target" not in ol:
            continue
        out.append(t)
    return out


def check_lifegain_triggers(state: GameState, gaining_player_id: str) -> list[Trigger]:
    """Whenever you/an opponent gain life — fire matching triggers."""
    out: list[Trigger] = []
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        ol = (card.oracle_text or "").lower()
        if "gain" not in ol or "life" not in ol or "whenever" not in ol:
            continue
        own = [t for t in parse_triggers(card) if t.trigger_type == TriggerType.LIFE_GAIN]
        if not own:
            continue
        # 'whenever you gain life' fires only for controller; 'whenever an
        # opponent gains life' fires when a different player gained.
        if "whenever you gain" in ol and card.controller_id != gaining_player_id:
            continue
        if "whenever an opponent gain" in ol and card.controller_id == gaining_player_id:
            continue
        out.extend(own)
    return out


# Cross-permanent ETB pattern. Captures:
#   group(1) = "another" (optional) — restricts to other permanents
#   group(2) = optional modifier string ("red", "artifact", "red artifact",
#              "non-token", "+1/+1 counter" filler etc.)
#   group(3) = noun ("creature", "artifact", "permanent", "land", ...)
# Followed by "enter" and optionally "under your control".
_ETB_OTHER_RE = re.compile(
    r"whenever\s+(another\s+|a\s+|an\s+)"
    r"([\w\-/+\s]*?)"
    r"\b(creature|permanent|artifact|enchantment|land|planeswalker|token)s?\s+"
    r"(?:you control\s+)?enters?"
    r"(?:\s+the battlefield)?"
    r"(\s+under your control)?",
    re.IGNORECASE,
)

_COLOR_WORDS = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G"}


def _entering_matches_filter(entering: CardInstance, modifier: str, noun: str) -> bool:
    """Does ``entering`` satisfy the filter '<modifier> <noun>'?"""
    type_line = entering.type_line.lower()
    noun_l = noun.lower()
    if noun_l == "permanent":
        if "land" not in type_line and "creature" not in type_line and \
           "artifact" not in type_line and "enchantment" not in type_line and \
           "planeswalker" not in type_line and "battle" not in type_line:
            return False
    elif noun_l == "token":
        if not entering.card_data.get("is_token"):
            return False
    else:
        if noun_l not in type_line:
            return False
    # Color / type modifiers (best-effort).
    mods = [m for m in re.split(r"[\s\-]+", modifier.lower()) if m]
    color_id = [c.lower() for c in (entering.card_data.get("color_identity") or [])]
    for m in mods:
        if m in ("non", "a", "an", "the", "of", "or", "and", ""):
            continue
        if m == "nontoken":
            if entering.card_data.get("is_token"):
                return False
            continue
        if m in _COLOR_WORDS:
            if _COLOR_WORDS[m].lower() not in color_id:
                return False
            continue
        # Treat as a subtype/type token (e.g. "artifact", "goblin").
        if m not in type_line:
            return False
    return True


def check_enters_battlefield_triggers(state: GameState, entering_card: CardInstance) -> list[Trigger]:
    """Return triggers that fire because ``entering_card`` just ETBed.

    Handles the entering card's own ETB triggers AND every other
    permanent's "whenever (another) [<modifier>] <type> enters [under your
    control]" trigger, with controller / color / type filtering.
    """
    triggered: list[Trigger] = []

    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        if card.instance_id == entering_card.instance_id:
            continue
        oracle = card.oracle_text or ""
        if "enter" not in oracle.lower() or "whenever" not in oracle.lower():
            continue
        for clause in re.split(r"[.\n]", oracle):
            m = _ETB_OTHER_RE.search(clause)
            if not m:
                continue
            quantifier = (m.group(1) or "").strip().lower()
            modifier = (m.group(2) or "").strip()
            noun = m.group(3)
            under_your_control = bool(m.group(4))
            # "another" requires same controller as the watcher AND not self.
            if quantifier == "another":
                if entering_card.controller_id != card.controller_id:
                    continue
            elif under_your_control:
                if entering_card.controller_id != card.controller_id:
                    continue
            if not _entering_matches_filter(entering_card, modifier, noun):
                continue
            effect = _extract_effect_from_etb(clause)
            triggered.append(Trigger(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                trigger_type=TriggerType.ENTERS_BATTLEFIELD,
                description=effect or clause.strip(),
            ))
            break  # one match per clause

    # The entering card's own ETB triggers (already filter "another" out).
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
    
    if "opponent lose" in description and "life" in description and "each" not in description:
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

    # Self-pump: "this creature gets +X/+Y until end of turn" — applies to the
    # source of the trigger (Foundry Street Denizen, etc.). detect_effect_kind
    # below would return "noop" because it expects "target creature".
    self_pump = re.search(
        r"(?:this (?:creature|permanent)|it)\s+gets\s+\+(\d+)/\+(\d+)",
        description,
    )
    if self_pump:
        dp, dt = int(self_pump.group(1)), int(self_pump.group(2))
        source = next((c for c in state.cards if c.instance_id == trigger.source_card_id), None)
        if source is not None and source.zone == Zone.BATTLEFIELD:
            source.eot_power_bonus = int(getattr(source, "eot_power_bonus", 0) or 0) + dp
            source.eot_toughness_bonus = int(getattr(source, "eot_toughness_bonus", 0) or 0) + dt
            state.log(f"[Trigger] {source.name} gets +{dp}/+{dt} until end of turn")
        return state

    # Echo (CR 702.50): "At the beginning of your upkeep, if this came under
    # your control since the beginning of your last upkeep, sacrifice it
    # unless you pay its echo cost." We must actually pay or sacrifice on
    # resolution.
    if "echo cost" in description or ("sacrifice it unless you pay" in description
                                       and "echo" in description):
        source = next((c for c in state.cards if c.instance_id == trigger.source_card_id), None)
        if source is not None and source.zone == Zone.BATTLEFIELD:
            from src.engine.mana import auto_tap_for_cost, parse_mana_cost
            from src.engine.zones import move_card
            # Parse the echo cost from the source's oracle: "Echo {1}{R}".
            ec_match = re.search(r"echo\s+((?:\{[^}]+\})+)",
                                 (source.oracle_text or ""), re.IGNORECASE)
            paid = False
            if ec_match:
                cost = parse_mana_cost(ec_match.group(1))
                # Heuristic: pay echo only if it's cheap (cmc <= 3) AND we
                # can afford it. Otherwise sacrifice — naive but safe.
                cmc = sum(cost.values())
                if cmc <= 3 and auto_tap_for_cost(state, controller, cost):
                    from src.engine.mana import pay_cost
                    pay_cost(controller, cost)
                    paid = True
                    state.log(f"[Trigger] {controller.name} pays echo cost for {source.name}")
            if not paid:
                move_card(
                    state, source.instance_id, Zone.BATTLEFIELD,
                    Zone.GRAVEYARD, source.owner_id,
                )
                state.log(f"[Trigger] {source.name} is sacrificed (echo unpaid)")
        return state

    # Generic fallback — let spell_effects.apply_spell_effect handle any
    # effect kind we haven't already covered above (destroy / exile / fight /
    # bounce / scry / mill / ramp / tutor / proliferate / +1+1 counter /
    # anthem-pump / pump / drain_each / damage_each / discard / surveil ...).
    try:
        from src.engine import spell_effects
        from src.engine.game_state import StackItem
        kind = spell_effects.detect_effect_kind(description)
        already_handled = {"draw", "lifegain", "damage", "discard", "token", "tap"}
        if kind != "noop" and kind not in already_handled:
            source = next((c for c in state.cards if c.instance_id == trigger.source_card_id), None)
            synthetic = StackItem(
                source_card_id=trigger.source_card_id,
                controller_id=controller.player_id,
                is_spell=False,
                card_data={
                    "name": (source.name if source else "Trigger"),
                    "oracle_text": description,
                },
            )
            # Auto-pick targets using the trigger description as oracle text.
            if source is not None:
                # Temporarily swap oracle so auto_pick_targets sees the right text.
                original_oracle = source.card_data.get("oracle_text", "")
                source.card_data["oracle_text"] = description
                try:
                    synthetic.targets = spell_effects.auto_pick_targets(
                        state, source, controller.player_id
                    )
                finally:
                    source.card_data["oracle_text"] = original_oracle
            spell_effects.apply_spell_effect(state, synthetic)
    except Exception as exc:  # pragma: no cover - defensive
        state.log(f"[Trigger] fallback effect failed: {exc!r}")

    return state
