"""
Keyword ability implementations — Flying, Haste, Lifelink, Menace, etc.

This module provides keyword ability checks and resolution logic.
"""

from __future__ import annotations

from src.engine.game_state import CardInstance, GameState


def has_keyword(card: CardInstance, keyword: str) -> bool:
    """Check if a card has a specific keyword."""
    return keyword.lower() in card.oracle_text.lower()


def can_block_with(attacker: CardInstance, defender: CardInstance) -> bool:
    """Check if defender can block attacker (considering evasion keywords)."""
    if not defender.is_creature:
        return False
    
    # Flying blockers can only block flying
    if has_keyword(attacker, "flying") and not has_keyword(defender, "flying"):
        # Flying creatures can be blocked by birds/other flying, or reach creatures
        if not has_keyword(defender, "reach"):
            return False
    
    # Menace requires 2+ blockers
    if has_keyword(attacker, "menace"):
        # Stub: would need to track multiple blockers
        pass
    
    # Shadow can't be blocked by non-shadow
    if has_keyword(attacker, "shadow") and not has_keyword(defender, "shadow"):
        return False
    
    # Unblockable by creatures with specific color/type
    unblockable = [
        "unblockable by blue creatures",
        "unblockable by red creatures",
        "unblockable by creatures",
    ]
    for u in unblockable:
        if u in attacker.oracle_text.lower():
            return False
    
    return True


def apply_lifelink(
    damage: int, attacker: CardInstance, defender_id: str, game_state: GameState
) -> GameState:
    """If attacker has lifelink, deal damage and gain that much life."""
    if has_keyword(attacker, "lifelink"):
        # Find attacker controller in players
        controller_id = attacker.controller
        if controller_id in game_state.players:
            game_state.players[controller_id].life += damage
    return game_state


def apply_deathtouch(damage: int, defender: CardInstance) -> int:
    """Deathtouch: 1 damage is lethal."""
    if hasattr(defender, "damage"):
        return 1  # Only 1 damage needed to kill
    return damage
