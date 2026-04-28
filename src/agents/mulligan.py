"""Strategy-aware mulligan policy.

Pure functions used by ``MTGAgent.decide_mulligan`` and
``MTGAgent.select_bottom_cards``.  Kept free of game-engine state so that
unit tests can call them directly with a hand of ``CardInstance`` and a
``Strategy`` enum value.

The heuristics here are deliberately simple; learned agents (LLM, world
model, active inference) override the corresponding ``MTGAgent`` methods
and ignore this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.engine.game_state import CardInstance


# Imported lazily to avoid a circular import with ``agent_strategies`` at
# module load time.  The enum only exposes ``str`` values so we accept
# either the enum or a raw string.
_DEFAULT_STRATEGY = "aggressive"


@dataclass(frozen=True)
class HandStats:
    """Cheap summary of an opening hand used by every heuristic."""

    lands: int
    creatures: int
    cheap_spells: int       # CMC <= 2 non-land
    expensive_spells: int   # CMC >= 5 non-land
    total: int


def summarise_hand(hand: Iterable[CardInstance]) -> HandStats:
    lands = 0
    creatures = 0
    cheap = 0
    expensive = 0
    total = 0
    for card in hand:
        total += 1
        if card.is_land():
            lands += 1
            continue
        cmc = float(card.cmc or 0.0)
        if card.is_creature():
            creatures += 1
        if cmc <= 2:
            cheap += 1
        if cmc >= 5:
            expensive += 1
    return HandStats(lands=lands, creatures=creatures, cheap_spells=cheap,
                     expensive_spells=expensive, total=total)


def _strategy_value(strategy) -> str:
    if strategy is None:
        return _DEFAULT_STRATEGY
    value = getattr(strategy, "value", strategy)
    return str(value).lower()


def should_keep(
    hand: list[CardInstance],
    strategy=None,
    mulligans_taken: int = 0,
    max_mulligans: int = 3,
) -> bool:
    """Return True if the agent should keep this opening hand.

    Always keeps once ``mulligans_taken >= max_mulligans``.

    Strategy heuristics:

    - aggressive: 1-3 lands and at least two cheap (CMC<=2) plays
    - control:    3-5 lands and at least one non-land
    - combo:      2-5 lands and at least 4 non-lands (cards to chain)
    - reactive:   2-5 lands and at least one cheap interaction piece
    """
    if mulligans_taken >= max_mulligans:
        return True

    stats = summarise_hand(hand)
    if stats.total == 0:
        return True

    s = _strategy_value(strategy)
    non_lands = stats.total - stats.lands

    if s == "aggressive":
        return 1 <= stats.lands <= 3 and stats.cheap_spells >= 2
    if s == "control":
        return 3 <= stats.lands <= 5 and non_lands >= 1
    if s == "combo":
        return 2 <= stats.lands <= 5 and non_lands >= 4
    if s == "reactive":
        return 2 <= stats.lands <= 5 and stats.cheap_spells >= 1

    # Fallback: classic "2-5 lands, at least one spell"
    return 2 <= stats.lands <= 5 and non_lands >= 1


def select_bottom_cards(
    hand: list[CardInstance],
    n: int,
    strategy=None,
) -> list[CardInstance]:
    """Choose ``n`` cards from ``hand`` to put on the bottom of the library.

    Strategy heuristics:

    - aggressive: bottom expensive non-lands first, then extra lands
    - control:    bottom cheap creatures first (we want late-game cards)
    - combo:      bottom non-enabler high-CMC cards first
    - reactive:   bottom expensive cards first, keep cheap interaction
    """
    if n <= 0 or not hand:
        return []
    n = min(n, len(hand))
    s = _strategy_value(strategy)

    def key(card: CardInstance) -> tuple:
        cmc = float(card.cmc or 0.0)
        is_land = card.is_land()
        is_creature = card.is_creature()
        if s == "aggressive":
            # bottom expensive non-lands, then lands beyond the third
            return (0 if is_land else 1, cmc)
        if s == "control":
            # bottom cheap creatures first
            return (1 if is_creature else 0, -cmc)
        if s == "combo":
            # bottom expensive non-lands first
            return (0 if is_land else 1, cmc)
        if s == "reactive":
            return (0 if is_land else 1, cmc)
        return (0 if is_land else 1, cmc)

    ranked = sorted(hand, key=key, reverse=True)
    return ranked[:n]
