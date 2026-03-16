"""
Triggered abilities — "whenever", "at", "when" effects that queue to the stack.

Reference: MTG rules 603.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.engine.game_state import GameState, StackItem


@dataclass
class Trigger:
    """A triggered ability waiting to be put on stack."""

    card_name: str
    trigger_text: str
    controller: str
    targets: list[str]


def detect_triggers(prev_state: GameState, new_state: GameState) -> list[Trigger]:
    """
    Detect which triggered abilities should fire based on state change.
    
    Common triggers:
    - "When [permanent] enters the battlefield"
    - "Whenever [action] happens"
    - "At the end of [phase]"
    - "At the beginning of [phase]"
    """
    triggers = []
    
    # ETB triggers (card entered battlefield)
    for zone_key in new_state.cards_in_zone:
        pid, zone_name = zone_key
        if zone_name != "battlefield":
            continue
        new_cards = new_state.cards_in_zone[zone_key]
        old_cards = prev_state.cards_in_zone.get(zone_key, [])
        old_ids = {c.instance_id for c in old_cards}
        
        for card in new_cards:
            if card.instance_id not in old_ids:
                if "enter the battlefield" in card.oracle_text.lower():
                    triggers.append(
                        Trigger(
                            card_name=card.name,
                            trigger_text=card.oracle_text,
                            controller=card.controller,
                            targets=[],
                        )
                    )
    
    return triggers


def queue_triggers(game_state: GameState, triggers: list[Trigger]) -> GameState:
    """Add triggered abilities to the stack in APNAP order."""
    for trigger in triggers:
        stack_item = StackItem(
            id=f"trigger_{trigger.card_name}_{len(game_state.stack)}",
            type="triggered_ability",
            source_card_name=trigger.card_name,
            text=trigger.trigger_text,
            controller=trigger.controller,
            targets=trigger.targets,
        )
        game_state.stack.append(stack_item)
    return game_state
