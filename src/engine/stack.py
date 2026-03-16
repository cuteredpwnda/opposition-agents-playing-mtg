"""
The Stack — LIFO resolution of spells and abilities.

Reference: mtg-python-engine (MIT) play.Play objects on the stack,
target legality checking before resolution, spell fizzling.
"""

from __future__ import annotations

from .game_state import GameState, StackItem


def push_to_stack(state: GameState, stack_item: StackItem) -> GameState:
    """Push a spell or ability onto the stack."""
    state.stack.append(stack_item)
    card_name = stack_item.card_data.get("name", "an ability")
    state.log(f"{card_name} added to the stack")
    return state


def resolve_top(state: GameState) -> StackItem | None:
    """Resolve the topmost item on the stack (LIFO)."""
    if not state.stack:
        return None

    item = state.stack.pop()
    card_name = item.card_data.get("name", "an ability")

    # Check if targets are still legal (fizzle if not)
    if item.targets and not _targets_still_legal(state, item):
        state.log(f"{card_name} fizzled — all targets are illegal")
        return item

    state.log(f"{card_name} resolves")
    # Actual resolution logic is handled by the rules engine
    return item


def is_empty(state: GameState) -> bool:
    """Check if the stack is empty."""
    return len(state.stack) == 0


def _targets_still_legal(state: GameState, item: StackItem) -> bool:
    """Check if at least one target is still legal."""
    for target_id in item.targets:
        card = next((c for c in state.cards if c.instance_id == target_id), None)
        if card is not None:
            return True
    return False
