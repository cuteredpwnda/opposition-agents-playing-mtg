"""
Static abilities — continuous effects that modify game state.

Static abilities are not activated and not triggered; they apply continuously
to affected permanents. Examples:
- "Creatures you control get +1/+1"
- "All creatures have flying"
- "Artifacts you control are indestructible"
- "Each creature has deathtouched"

This module handles:
1. Parsing static abilities from oracle text
2. Determining which permanents are affected by each static ability
3. Applying effects (P/T modifications, keywords)
4. Querying effective stats and keywords after applying modifiers
"""

import re
from .game_state import GameState, StaticAbility, Zone, CardInstance


def parse_static_abilities(card: CardInstance) -> list[StaticAbility]:
    """Parse static abilities from a card's oracle text.
    
    Detects patterns like:
    - "Creatures you control get +1/+1"
    - "Creatures get flying"
    - "~ is indestructible"
    - "{T}: Add {R}" is an activated ability (skip)
    
    Returns list of StaticAbility objects.
    """
    if not card.card_data or not card.oracle_text:
        return []
    
    abilities = []
    lines = card.oracle_text.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip activated abilities (contain {T} or {mana})
        if "{" in line and ":" in line and not any(x in line for x in ["get", "gain", "have"]):
            continue
        
        ability = _parse_static_ability_line(card, line)
        if ability:
            abilities.append(ability)
    
    return abilities


def _parse_static_ability_line(card: CardInstance, line: str) -> StaticAbility | None:
    """Parse a single line of oracle text into a static ability."""
    lower_line = line.lower()

    # Pattern: "<Subtype> creatures you control get +X/+Y" e.g.
    # "Goblin creatures you control get +1/+1 and have haste."
    sub_match = re.search(
        r"\b([a-z]+)\s+creatures?\s+you\s+control\s+(?:get|have)\s+([+\-]?\d+)/([+\-]?\d+)",
        lower_line,
    )
    if sub_match and sub_match.group(1) not in ("non", "all", "other", "the", "your"):
        subtype = sub_match.group(1)
        # Filter out reserved type-line words that are NOT subtypes ("token"
        # is treated as a generic type, etc.). Whitelist anything else.
        if subtype not in ("a", "an"):
            return StaticAbility(
                source_card_id=card.instance_id,
                controller_id=card.controller_id,
                scope="subtype_creatures_you_control",
                effect_type="power_toughness",
                power_mod=int(sub_match.group(2)),
                toughness_mod=int(sub_match.group(3)),
                subtype_filter=subtype,
                description=line,
            )

    # Pattern: "<Subtype>(s) you control have <keyword>" — e.g.
    # "Goblins you control have haste." Granting a keyword to a creature
    # type without a P/T mod.
    keywords_to_match = ["flying", "haste", "deathtouch", "lifelink",
                         "vigilance", "indestructible", "hexproof", "shroud",
                         "menace", "trample", "reach", "first strike",
                         "double strike"]
    sub_kw_match = re.search(
        r"\b([a-z]+)s?\s+you\s+control\s+(?:have|get|gain)\s+("
        + "|".join(keywords_to_match) + r")\b",
        lower_line,
    )
    if sub_kw_match and sub_kw_match.group(1) not in (
        "non", "all", "other", "the", "your", "creature", "creatures",
        "permanent", "permanents", "artifact", "artifacts",
    ):
        # "Goblins" -> "goblin" so we match the singular form on type_line.
        subtype = sub_kw_match.group(1).rstrip("s")
        return StaticAbility(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            scope="subtype_creatures_you_control",
            effect_type="keyword",
            keywords=[sub_kw_match.group(2)],
            subtype_filter=subtype,
            description=line,
        )

    # Pattern: "Creatures you control get +X/+Y" or "All creatures get +X/+Y"
    match = re.search(
        r"(?:(?:(\w+\s+(?:you\s+)?control)|(?:all\s+)?(\w+))\s+)?(?:get|have)\s+([+\-]?\d+)/([+\-]?\d+)",
        lower_line
    )
    if match:
        scope_text = match.group(1) if match.group(1) else (match.group(2) or "creatures")
        power_mod = int(match.group(3))
        toughness_mod = int(match.group(4))
        
        scope = _parse_scope(scope_text)
        return StaticAbility(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            scope=scope,
            effect_type="power_toughness",
            power_mod=power_mod,
            toughness_mod=toughness_mod,
            description=line,
        )
    
    # Pattern: "Creatures you control get keyword"
    keywords_to_match = ["flying", "haste", "deathtouched", "lifelink", "vigilance", 
                        "indestructible", "hexproof", "shroud", "menace", "trample"]
    
    for keyword in keywords_to_match:
        if keyword in lower_line:
            # Check for "get" or "have" before the keyword
            pattern = rf"(?:(\w+\s+(?:you\s+)?control)|(?:all\s+)?(\w+))\s+(?:get|have|gain)\s+(?:.*\s+)?{keyword}"
            match = re.search(pattern, lower_line)
            if match:
                scope_text = match.group(1) if match.group(1) else (match.group(2) or "creatures")
                scope = _parse_scope(scope_text)
                return StaticAbility(
                    source_card_id=card.instance_id,
                    controller_id=card.controller_id,
                    scope=scope,
                    effect_type="keyword",
                    keywords=[keyword],
                    description=line,
                )
    
    # Pattern: "~ is indestructible" (the card itself)
    if "indestructible" in lower_line and ("this creature" in lower_line or "~" in lower_line):
        return StaticAbility(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            scope="source_only",
            effect_type="indestructible",
            keywords=["indestructible"],
            description=line,
        )
    
    # Pattern: "All creatures have flying"
    pattern = r"(?:all\s+)?(?:permanents|creatures|artifacts)\s+(?:have|get)\s+(" + "|".join(keywords_to_match) + ")"
    match = re.search(pattern, lower_line)
    if match:
        keyword = match.group(1)
        return StaticAbility(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            scope="all",
            effect_type="keyword",
            keywords=[keyword],
            description=line,
        )
    
    return None


def _parse_scope(scope_text: str) -> str:
    """Convert scope text to a standard scope string."""
    scope = scope_text.lower()
    
    if "you control" in scope:
        if "creature" in scope:
            return "creatures_you_control"
        elif "artifact" in scope:
            return "artifacts_you_control"
        elif "permanent" in scope:
            return "permanents_you_control"
        else:
            return "permanents_you_control"
    
    elif "opponent" in scope:
        if "creature" in scope:
            return "opponent_creatures"
        else:
            return "opponent_permanents"
    
    elif "all" in scope or "each" in scope:
        if "creature" in scope:
            return "all_creatures"
        elif "artifact" in scope:
            return "all_artifacts"
        elif "permanent" in scope:
            return "all_permanents"
        else:
            return "all_permanents"
    
    # Handle single word types (e.g., "creatures", "artifacts")
    elif "creature" in scope:
        return "all_creatures"
    elif "artifact" in scope:
        return "all_artifacts"
    elif "permanent" in scope:
        return "all_permanents"
    
    return "all_permanents"


def get_static_abilities_affecting(state: GameState, target_card: CardInstance) -> list[StaticAbility]:
    """Get all static abilities that affect a target card.
    
    Static abilities can affect their own card or other permanents depending on scope.
    """
    affecting = []
    
    for card in state.cards:
        if card.zone != Zone.BATTLEFIELD:
            continue
        
        static_abilities = parse_static_abilities(card)
        for ability in static_abilities:
            if _ability_affects_card(state, ability, card, target_card):
                affecting.append(ability)
    
    return affecting


def _ability_affects_card(state: GameState, ability: StaticAbility, source_card: CardInstance, 
                         target_card: CardInstance) -> bool:
    """Check if a static ability affects a target card."""
    
    # Source only affects itself
    if ability.scope == "source_only":
        return source_card.instance_id == target_card.instance_id
    
    # Target must be on battlefield
    if target_card.zone != Zone.BATTLEFIELD:
        return False
    
    # Creatures you control
    if ability.scope == "creatures_you_control":
        return (target_card.controller_id == source_card.controller_id and 
                target_card.is_creature())

    # Subtype-restricted creatures you control (e.g. Goblin, Elf, Soldier).
    if ability.scope == "subtype_creatures_you_control":
        if target_card.controller_id != source_card.controller_id:
            return False
        if not target_card.is_creature():
            return False
        sub = (ability.subtype_filter or "").lower()
        if not sub:
            return False
        return sub in target_card.type_line.lower()
    
    # Artifacts you control
    if ability.scope == "artifacts_you_control":
        return (target_card.controller_id == source_card.controller_id and 
                target_card.is_artifact())
    
    # Permanents you control
    if ability.scope == "permanents_you_control":
        return target_card.controller_id == source_card.controller_id
    
    # All creatures
    if ability.scope == "all_creatures":
        return target_card.is_creature()
    
    # All artifacts
    if ability.scope == "all_artifacts":
        return target_card.is_artifact()
    
    # All permanents (default)
    if ability.scope == "all_permanents":
        return True
    
    return False


def get_effective_power_toughness(card: CardInstance, state: GameState) -> tuple[int, int]:
    """Calculate a card's effective power and toughness after applying static effects."""
    
    # Get base P/T
    base_power = _parse_int(card.power) or 0
    base_toughness = _parse_int(card.toughness) or 0
    
    # Apply all affecting static abilities
    power_mod = 0
    toughness_mod = 0
    
    affecting = get_static_abilities_affecting(state, card)
    for ability in affecting:
        if ability.effect_type == "power_toughness":
            power_mod += ability.power_mod
            toughness_mod += ability.toughness_mod

    # End-of-turn pump bonuses (set by spell_effects.pump / anthem_pump).
    power_mod += int(getattr(card, "eot_power_bonus", 0) or 0)
    toughness_mod += int(getattr(card, "eot_toughness_bonus", 0) or 0)
    # +1/+1 counters live in card.counters.
    plus = int(card.counters.get("+1/+1", 0) or 0)
    minus = int(card.counters.get("-1/-1", 0) or 0)
    power_mod += plus - minus
    toughness_mod += plus - minus

    # Equipment attached to this creature contributes "Equipped creature
    # gets +N/+M" to its P/T (CR 702.6).
    for other in state.cards:
        if other.zone != Zone.BATTLEFIELD:
            continue
        if other.attached_to != card.instance_id:
            continue
        m = re.search(
            r"equipped creature gets \+(\d+)/\+(\d+)",
            (other.oracle_text or "").lower(),
        )
        if m:
            power_mod += int(m.group(1))
            toughness_mod += int(m.group(2))

    return (base_power + power_mod, base_toughness + toughness_mod)


def has_keyword(card: CardInstance, state: GameState, keyword: str) -> bool:
    """Check if a card has a keyword, including from static abilities."""
    
    # Check card's intrinsic keywords
    if keyword.lower() in card.oracle_text.lower():
        return True
    
    # Check static abilities that grant keywords
    affecting = get_static_abilities_affecting(state, card)
    for ability in affecting:
        if ability.effect_type == "keyword" and keyword.lower() in [k.lower() for k in ability.keywords]:
            return True
    
    return False


def _parse_int(val: str | None) -> int | None:
    """Try to parse an integer from a string."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
