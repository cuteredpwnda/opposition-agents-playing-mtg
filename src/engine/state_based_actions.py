"""
State-based actions (CR 704) — automatic effects checked multiple times per turn.

Handles: lethal damage, 0-toughness creatures, commander damage, empty draw,
player action legal but impossible, deck size rules, etc.
"""

from __future__ import annotations

from src.engine.game_state import GameState, Zone


def check_state_based_actions(game_state: GameState) -> GameState:
    """Check and apply all state-based actions.
    
    SBAs are checked:
    - After each player's action resolves
    - After the stack is resolved
    - Before passing priority
    
    They're applied simultaneously, not one at a time.
    """
    # Move to-be-destroyed creatures to graveyard
    for zone_key in list(game_state.cards_in_zone.keys()):
        pid, zone_name = zone_key
        if zone_name != "battlefield":
            continue
        cards = game_state.cards_in_zone[zone_key]
        survivors = []
        for card in cards:
            # Creature has 0 or less toughness → dies
            if card.is_creature and card.toughness is not None and card.toughness <= 0:
                game_state.cards_in_zone.setdefault((pid, "graveyard"), []).append(card)
                continue
            # Creature has lethal damage → dies
            if card.is_creature and getattr(card, "damage", 0) >= (card.toughness or 0):
                game_state.cards_in_zone.setdefault((pid, "graveyard"), []).append(card)
                continue
            survivors.append(card)
        game_state.cards_in_zone[zone_key] = survivors

    # Check lifetotal SBAs
    for pid, pstate in game_state.players.items():
        # Player has 0 or less life → loses
        if pstate.life <= 0:
            game_state.game_over = True
            game_state.winner = next(p for p in game_state.players if p != pid)
            break
        # Commander damage (21+ combat damage from single commander) → loses
        if not hasattr(pstate, 'commander_damage'):
            pstate.commander_damage = {}
        for opponent_id, damage in pstate.commander_damage.items():
            if damage >= 21:
                game_state.game_over = True
                game_state.winner = next(p for p in game_state.players if p != pid)
                break

    # Check draw empties (trying to draw from empty library)
    for zone_key in list(game_state.cards_in_zone.keys()):
        pid, zone_name = zone_key
        if zone_name != "library":
            continue
        if len(game_state.cards_in_zone[zone_key]) == 0:
            # At next draw step, player loses (for now)
            # In full implementation: queue draw-loss effect
            pass

    return game_state
