"""
Activated abilities system — handles permanent abilities controlled by players.

Activated abilities are fundamentally different from triggered abilities:
- Triggered: Automatic, when condition is met
- Activated: Controlled by player, any time they have priority

Most common activated abilities:
- Mana abilities: "{T}: Add {R}" (can be used anytime, don't use the stack)
- Other abilities: "{2}{U}: Draw a card" (require priority, go on stack)

Reference: MTG CR 602 — Activated Abilities
"""

from __future__ import annotations

import re
from src.engine.game_state import Ability, CardInstance, GameState, Zone


def parse_abilities(card: CardInstance) -> list[Ability]:
    """Extract activated abilities from a card's oracle text.
    
    This finds abilities of the form:
    - "{cost}: effect"
    - Cost can be: mana ({R}, {2}{U}, etc.), tapability ({T}), other ({Q}, etc.)
    
    Returns list of activated abilities this card has.
    """
    oracle = card.oracle_text.lower()
    if not oracle:
        return []
    
    abilities = []
    
    # Pattern: "{cost}: effect"
    # Matches patterns like:
    #   {T}: Add {R}
    #   {2}{U}: Draw a card
    #   {1}{B}, {T}: Create a token
    pattern = r"\{[^\}]+\}(?:\s*,\s*\{[^\}]+\})*\s*:\s*([^.\n]+)"
    
    matches = re.finditer(pattern, card.oracle_text)
    for match in matches:
        cost = extract_cost(match.group(0))
        effect = match.group(1).strip()
        
        # Determine if this can be used anytime
        # Mana abilities: simple, produce mana, can use anytime
        can_use_anytime = is_mana_ability(effect)
        
        abilities.append(Ability(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            cost=cost,
            effect=effect,
            can_use_any_time=can_use_anytime,
            description=f"{cost}: {effect}"
        ))
    
    return abilities


def extract_cost(cost_text: str) -> str:
    """Extract just the cost part from '{cost}: effect' pattern.
    
    E.g., "{T}: Add {R}" → "{T}"
    """
    # Find everything before the colon
    match = re.match(r"([^:]+):", cost_text)
    if match:
        return match.group(1).strip()
    return ""


def is_mana_ability(effect: str) -> bool:
    """Check if ability is a mana ability (can be used anytime).
    
    Mana abilities are simple:
    - Add mana
    - Don't have targets
    - Don't cause other abilities to trigger
    """
    effect_lower = effect.lower()
    
    # Simple heuristic: if it just adds mana, it's a mana ability
    if "add" in effect_lower and any(mana in effect_lower for mana in ["{w}", "{u}", "{b}", "{r}", "{g}", "{c}"]):
        # Check it doesn't have other effects
        if "," not in effect and "and" not in effect:
            return True
    
    return False


def get_legal_activated_abilities(state: GameState, card: CardInstance, player_id: str) -> list[Ability]:
    """Get all activated abilities on a card that a player can currently use.
    
    Restrictions:
    - Card must be on battlefield
    - Player must control the card
    - Player must be able to pay the cost (mana + tap)
    - For non-mana abilities: player must have priority
    
    Args:
        state: Current game state
        card: Card instance to check for abilities
        player_id: Player checking for legal abilities
    
    Returns:
        List of abilities the player can currently activate
    """
    if card.zone != Zone.BATTLEFIELD:
        return []
    
    if card.controller_id != player_id:
        return []
    
    # Get all abilities on this card
    all_abilities = parse_abilities(card)
    legal = []
    
    player = next((p for p in state.players if p.player_id == player_id), None)
    if not player:
        return []
    
    for ability in all_abilities:
        # Check if player can pay cost
        # Cost contains both mana requirements and tap requirement
        can_pay = can_pay_ability_cost(player, card, ability)
        
        if can_pay:
            legal.append(ability)
    
    return legal


def can_pay_ability_cost(player, card: CardInstance, ability: Ability) -> bool:
    """Check if player can pay an ability's cost.
    
    Cost includes:
    - Mana ({R}, {2}{U}, etc.)
    - Tap ({T}) - card must be untapped
    - Other ({Q}, etc.) - simplified, assume can pay
    
    Returns True if player has mana and card is not tapped (if needed).
    """
    cost = ability.cost.lower()
    
    # Check if tap is required
    if "{t}" in cost or "tap" in cost:
        if card.tapped:
            return False  # Can't pay tap cost if already tapped
    
    # Check mana cost
    # Parse mana symbols: {R}, {U}, {W}, {B}, {G}, {1}, {2}, etc.
    mana_cost = extract_mana_cost(cost)
    
    # Get available mana
    available_mana = player.mana_pool
    
    # Simple check: do we have enough total mana?
    # (Simplified - doesn't check color-specific requirements strictly)
    total_required = sum(1 for m in mana_cost if m in "WUBRG")
    total_generic = sum(int(m) for m in mana_cost if m.isdigit())
    total_available = sum(available_mana.values())
    
    return (total_required + total_generic) <= total_available


def extract_mana_cost(cost_text: str) -> list[str]:
    """Extract individual mana symbols from cost text.
    
    E.g., "{2}{U}{B}" → [2, U, B]
    """
    # Find all {X} patterns
    matches = re.findall(r"\{([^\}]+)\}", cost_text)
    result = []
    for match in matches:
        if match.isdigit():
            result.extend([c for c in match])  # Split digits
        elif match in "WUBRG":
            result.append(match)
    return result


def resolve_ability(state: GameState, ability: Ability, player_id: str) -> GameState:
    """Resolve an activated ability's effect.
    
    Effects supported:
    - Add mana: "{T}: Add {R}"
    - Draw card: "{U}: Draw a card"
    - Gain life: "{2}: Gain 1 life"
    - Deal damage: "{1}{R}: Deal 1 damage to target opponent"
    - Discard: "{B}: Discard a card"
    
    Returns updated game state.
    """
    player = next((p for p in state.players if p.player_id == player_id), None)
    card = next((c for c in state.cards if c.instance_id == ability.source_card_id), None)
    
    if not player or not card:
        return state
    
    effect = ability.effect.lower()
    
    # Pay the cost first
    # Tap the card if needed
    if "{t}" in ability.cost.lower():
        card.tapped = True
        state.log(f"{card.name} is tapped")
    
    # Pay mana cost (simplified - remove from pool)
    mana_symbols = extract_mana_cost(ability.cost)
    for symbol in mana_symbols:
        if symbol in "WUBRG":
            if player.mana_pool[symbol] > 0:
                player.mana_pool[symbol] -= 1
        elif symbol.isdigit():
            # Generic mana - remove from any color
            for color in "WUBRG":
                if player.mana_pool[color] > 0:
                    player.mana_pool[color] -= 1
                    break
    
    # Resolve the effect
    
    # Add mana effects: "add {R}", "add {2}", etc.
    if "add" in effect and "{" in effect:
        mana_adds = re.findall(r"\{([^\}]+)\}", effect)
        for mana in mana_adds:
            mana_upper = mana.upper()
            if mana_upper in "WUBRG":
                player.mana_pool[mana_upper] += 1
                state.log(f"[Ability] {player.name} adds {mana_upper} mana")
            elif mana_upper == "C":
                player.mana_pool["C"] += 1
                state.log(f"[Ability] {player.name} adds colorless mana")
    
    # Draw card
    if "draw" in effect and "card" in effect:
        from src.engine.zones import move_card
        library = [c for c in state.cards if c.zone == Zone.LIBRARY and c.owner_id == player.player_id]
        if library:
            card_to_draw = library[0]
            state = move_card(state, card_to_draw.instance_id, Zone.LIBRARY, Zone.HAND, player.player_id)
            state.log(f"[Ability] {player.name} draws a card")
    
    # Gain life
    if "gain" in effect and "life" in effect:
        match = re.search(r"gain (\d+) life", effect)
        amount = int(match.group(1)) if match else 1
        player.life_total += amount
        state.log(f"[Ability] {player.name} gains {amount} life")
    
    # Deal damage
    if "deal" in effect and "damage" in effect:
        opponent = next((p for p in state.players if p.player_id != player.player_id), None)
        if opponent:
            match = re.search(r"deal (\d+) damage", effect)
            amount = int(match.group(1)) if match else 1
            opponent.life_total -= amount
            state.log(f"[Ability] {opponent.name} takes {amount} damage")
    
    # Discard
    if "discard" in effect and "card" in effect:
        from src.engine.zones import move_card
        hand = [c for c in state.cards if c.zone == Zone.HAND and c.owner_id == player.player_id]
        if hand:
            card_to_discard = hand[0]
            state = move_card(state, card_to_discard.instance_id, Zone.HAND, Zone.GRAVEYARD, player.player_id)
            state.log(f"[Ability] {player.name} discards a card")
    
    return state
