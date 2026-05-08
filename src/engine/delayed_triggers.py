"""Delayed triggered abilities (H5 — CR 603.7).

A *delayed triggered ability* is a triggered ability that is created by a
spell or another ability that sets up something to happen at a later game
event — typically "at the beginning of the next end step", "at end of
combat", or "at the beginning of your next upkeep".

Common use cases:
* Myriad: "Exile those tokens at end of combat." (CR 702.116c)
* Encore tokens: "Exile those tokens at the beginning of your next end step."
* Flicker/bounce: "Return at the beginning of the next end step."
* "Until end of turn" effects that need cleanup beyond the layer system.
* Suspend: "At the beginning of your upkeep, remove a time counter…" (already
  handled ad-hoc in game_runner; can migrate here over time).

Design
------
``DelayedTrigger`` objects are scheduled via ``schedule(state, trigger)`` and
fired by the game-runner at the appropriate phase-boundary hook by calling
``fire(state, point)``.  Each trigger fires **at most once** (``once=True``)
or persists until explicitly cancelled.

Reference:
  XMage — DelayedTriggeredAbility.java /
  AtTheEndOfCombatDelayedTriggeredAbility.java (MIT, magefree/mage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .game_state import GameState


# ---------------------------------------------------------------------------
# Trigger-point enum
# ---------------------------------------------------------------------------

class TriggerPoint(str, Enum):
    END_OF_COMBAT = "end_of_combat"
    END_OF_TURN = "end_of_turn"
    NEXT_UPKEEP = "next_upkeep"
    NEXT_END_STEP = "next_end_step"
    BEFORE_NEXT_UNTAP = "before_next_untap"
    NEXT_DRAW = "next_draw"


# ---------------------------------------------------------------------------
# DelayedTrigger dataclass
# ---------------------------------------------------------------------------

@dataclass
class DelayedTrigger:
    """A one-shot triggered ability to be fired at a named future phase.

    Parameters
    ----------
    trigger_point:   When the effect fires (see ``TriggerPoint``).
    effect:          Callable executed when the trigger fires.  It receives
                     ``state`` and may mutate it freely.
    controller_id:   Player who controls the ability (APNAP ordering).
    description:     Human-readable description for the game log.
    only_this_turn:  If True (default), the trigger expires at cleanup if
                     the named point was not reached this turn.
    once:            If True (default), remove after first firing.
    condition:       Optional guard — effect fires only if this returns True.
    """

    trigger_point: TriggerPoint
    effect: Callable[[GameState], None]
    controller_id: str
    description: str = ""
    only_this_turn: bool = True
    once: bool = True
    condition: Callable[[GameState], bool] = field(
        default_factory=lambda: (lambda _s: True)
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class DelayedTriggerRegistry:
    """Stores pending delayed triggers and fires them at phase-boundary hooks."""

    def __init__(self) -> None:
        self._pending: list[DelayedTrigger] = []

    def schedule(self, trigger: DelayedTrigger) -> None:
        """Schedule a delayed trigger to fire at its designated point."""
        self._pending.append(trigger)

    def fire(self, state: GameState, point: TriggerPoint) -> None:
        """Fire (and remove) all one-shot triggers scheduled for ``point``."""
        fired: list[DelayedTrigger] = []
        kept: list[DelayedTrigger] = []
        for t in self._pending:
            if t.trigger_point != point:
                kept.append(t)
                continue
            try:
                if t.condition(state):
                    if t.description:
                        state.log(f"[DELAYED] {t.description}")
                    t.effect(state)
                    fired.append(t)
                    if not t.once:
                        kept.append(t)
                else:
                    # Condition not met — keep unless only_this_turn
                    if not t.only_this_turn:
                        kept.append(t)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug(
                    "Delayed trigger '%s' raised %s", t.description, exc
                )
        self._pending = kept

    def expire_end_of_turn(self) -> None:
        """Remove all ``only_this_turn=True`` triggers that were never fired."""
        self._pending = [t for t in self._pending if not t.only_this_turn]

    def cancel_for_card(self, card_id: str) -> None:
        """Cancel triggers whose description contains ``card_id`` (coarse)."""
        self._pending = [t for t in self._pending if card_id not in t.description]

    def pending(self) -> list[DelayedTrigger]:
        return list(self._pending)


# ---------------------------------------------------------------------------
# GameState helper
# ---------------------------------------------------------------------------

def get_delayed_registry(state: GameState) -> DelayedTriggerRegistry:
    """Lazily create and return the registry on ``state``."""
    reg: Optional[DelayedTriggerRegistry] = getattr(
        state, "_delayed_trigger_registry", None
    )
    if reg is None or not isinstance(reg, DelayedTriggerRegistry):
        reg = DelayedTriggerRegistry()
        state._delayed_trigger_registry = reg
    return reg


def schedule(
    state: GameState,
    point: TriggerPoint,
    effect: Callable[[GameState], None],
    controller_id: str,
    description: str = "",
    only_this_turn: bool = True,
    once: bool = True,
    condition: Callable[[GameState], bool] | None = None,
) -> None:
    """Convenience wrapper: schedule a delayed trigger on ``state``."""
    get_delayed_registry(state).schedule(
        DelayedTrigger(
            trigger_point=point,
            effect=effect,
            controller_id=controller_id,
            description=description,
            only_this_turn=only_this_turn,
            once=once,
            condition=condition or (lambda _s: True),
        )
    )


def fire(state: GameState, point: TriggerPoint) -> None:
    """Convenience: fire all pending delayed triggers for ``point``."""
    get_delayed_registry(state).fire(state, point)
