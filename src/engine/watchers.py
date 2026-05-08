"""Game-event watchers — observer pattern for per-turn bookkeeping (H3).

XMage uses ``Watcher`` subclasses that subscribe to ``GameEvent`` types and
own the bookkeeping for things like "how many spells were cast this turn",
"how much life did this player lose this turn", etc.  This module mirrors
that design.

Watchers are *pure observers* — they **do not mutate game state**.  Their
read-only answers are queried by mechanic-checking functions (storm count,
spectacle condition, surge, raid, etc.).

The ``WatcherRegistry`` lives on ``GameState`` (lazily created via
``get_watcher_registry``).  Standard watchers are registered automatically
on first access.

Lifecycle:
* ``WatcherRegistry.fire(event)`` — call whenever a notable event occurs.
* ``WatcherRegistry.reset_turn()`` — call at cleanup to clear per-turn state.

Reference: XMage Watcher.java / CastSpellLastTurnWatcher.java (MIT, magefree/mage).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Type

if TYPE_CHECKING:
    from .game_state import GameState


# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------


class GameEventKind(str, Enum):
    SPELL_CAST = "spell_cast"
    ABILITY_ACTIVATED = "ability_activated"
    LIFE_LOST = "life_lost"
    LIFE_GAINED = "life_gained"
    DAMAGE_DEALT = "damage_dealt"
    CREATURE_ATTACKED = "creature_attacked"
    CARD_DRAWN = "card_drawn"
    PERMANENT_ETB = "permanent_etb"
    PERMANENT_LTB = "permanent_ltb"
    LAND_PLAYED = "land_played"
    CARD_DISCARDED = "card_discarded"
    COUNTER_ADDED = "counter_added"


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


@dataclass
class GameEvent:
    """A discrete game event broadcast to all watchers."""

    kind: GameEventKind
    player_id: str = ""
    source_id: str = ""          # card instance_id of event source
    target_id: str = ""          # card instance_id or player_id of target
    amount: int = 0              # life, damage, counter quantity, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract watcher
# ---------------------------------------------------------------------------


class Watcher(ABC):
    """Subscribe to a set of event kinds; accumulate state; reset at boundary."""

    reset_at: str = "turn"   # "turn" | "phase" | "game"

    @abstractmethod
    def watches(self) -> set[GameEventKind]: ...

    @abstractmethod
    def observe(self, event: GameEvent) -> None: ...

    def reset(self) -> None:
        """Called by WatcherRegistry.reset_turn() (or reset_phase/reset_game)."""


# ---------------------------------------------------------------------------
# Concrete watchers (replace per-turn GameState / PlayerState ad-hoc fields)
# ---------------------------------------------------------------------------


class SpellsCastThisTurnWatcher(Watcher):
    """Count spells cast per player this turn (storm, spectacle, magecraft gate)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}      # player_id -> count
        self.spells: list[GameEvent] = []     # ordered cast events

    def watches(self) -> set[GameEventKind]:
        return {GameEventKind.SPELL_CAST}

    def observe(self, event: GameEvent) -> None:
        self.counts[event.player_id] = self.counts.get(event.player_id, 0) + 1
        self.spells.append(event)

    def reset(self) -> None:
        self.counts.clear()
        self.spells.clear()

    def count_for(self, player_id: str) -> int:
        return self.counts.get(player_id, 0)

    def total(self) -> int:
        return sum(self.counts.values())


class LifeLostThisTurnWatcher(Watcher):
    """Track life lost per player this turn (spectacle condition)."""

    def __init__(self) -> None:
        self.totals: dict[str, int] = {}

    def watches(self) -> set[GameEventKind]:
        return {GameEventKind.LIFE_LOST}

    def observe(self, event: GameEvent) -> None:
        self.totals[event.player_id] = self.totals.get(event.player_id, 0) + event.amount

    def reset(self) -> None:
        self.totals.clear()

    def lost_by(self, player_id: str) -> int:
        return self.totals.get(player_id, 0)


class PlayerAttackedThisTurnWatcher(Watcher):
    """Track which players attacked this turn (raid condition)."""

    def __init__(self) -> None:
        self.player_attacked: set[str] = set()    # player_ids who attacked
        self.attacker_ids: set[str] = set()       # card instance_ids

    def watches(self) -> set[GameEventKind]:
        return {GameEventKind.CREATURE_ATTACKED}

    def observe(self, event: GameEvent) -> None:
        self.player_attacked.add(event.player_id)
        if event.source_id:
            self.attacker_ids.add(event.source_id)

    def reset(self) -> None:
        self.player_attacked.clear()
        self.attacker_ids.clear()

    def did_attack(self, player_id: str) -> bool:
        return player_id in self.player_attacked


class LandPlayedThisTurnWatcher(Watcher):
    """Track land plays this turn (landfall, max-land-play enforcement)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def watches(self) -> set[GameEventKind]:
        return {GameEventKind.LAND_PLAYED}

    def observe(self, event: GameEvent) -> None:
        self.counts[event.player_id] = self.counts.get(event.player_id, 0) + 1

    def reset(self) -> None:
        self.counts.clear()

    def count_for(self, player_id: str) -> int:
        return self.counts.get(player_id, 0)


class DamageThisTurnWatcher(Watcher):
    """Track damage dealt per source/target pair this turn."""

    def __init__(self) -> None:
        # source_id -> total damage dealt
        self.by_source: dict[str, int] = {}
        # player_id -> total damage received
        self.to_player: dict[str, int] = {}

    def watches(self) -> set[GameEventKind]:
        return {GameEventKind.DAMAGE_DEALT}

    def observe(self, event: GameEvent) -> None:
        self.by_source[event.source_id] = (
            self.by_source.get(event.source_id, 0) + event.amount
        )
        self.to_player[event.target_id] = (
            self.to_player.get(event.target_id, 0) + event.amount
        )

    def reset(self) -> None:
        self.by_source.clear()
        self.to_player.clear()


class LifeGainedThisTurnWatcher(Watcher):
    """Track life gained per player this turn."""

    def __init__(self) -> None:
        self.totals: dict[str, int] = {}

    def watches(self) -> set[GameEventKind]:
        return {GameEventKind.LIFE_GAINED}

    def observe(self, event: GameEvent) -> None:
        self.totals[event.player_id] = self.totals.get(event.player_id, 0) + event.amount

    def reset(self) -> None:
        self.totals.clear()

    def gained_by(self, player_id: str) -> int:
        return self.totals.get(player_id, 0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class WatcherRegistry:
    """Manages a collection of watchers and routes events to subscribers."""

    def __init__(self) -> None:
        self._watchers: list[Watcher] = []
        self._by_kind: dict[GameEventKind, list[Watcher]] = {}

    def register(self, watcher: Watcher) -> None:
        self._watchers.append(watcher)
        for kind in watcher.watches():
            self._by_kind.setdefault(kind, []).append(watcher)

    def fire(self, event: GameEvent) -> None:
        for w in self._by_kind.get(event.kind, []):
            try:
                w.observe(event)
            except Exception:
                pass

    def reset_turn(self) -> None:
        for w in self._watchers:
            if w.reset_at in ("turn", "phase"):
                w.reset()

    def reset_phase(self) -> None:
        for w in self._watchers:
            if w.reset_at == "phase":
                w.reset()

    def get(self, watcher_cls: Type[Watcher]) -> Optional[Watcher]:
        for w in self._watchers:
            if isinstance(w, watcher_cls):
                return w
        return None


# Default set of watchers installed on every registry
_DEFAULT_WATCHERS: list[Type[Watcher]] = [
    SpellsCastThisTurnWatcher,
    LifeLostThisTurnWatcher,
    PlayerAttackedThisTurnWatcher,
    LandPlayedThisTurnWatcher,
    DamageThisTurnWatcher,
    LifeGainedThisTurnWatcher,
]


# ---------------------------------------------------------------------------
# GameState helper
# ---------------------------------------------------------------------------


def get_watcher_registry(state: GameState) -> WatcherRegistry:
    """Lazily create and return the ``WatcherRegistry`` on ``state``."""
    reg: Optional[WatcherRegistry] = getattr(state, "_watcher_registry", None)
    if reg is None or not isinstance(reg, WatcherRegistry):
        reg = WatcherRegistry()
        for cls in _DEFAULT_WATCHERS:
            reg.register(cls())
        state._watcher_registry = reg
    return reg


def fire(state: GameState, event: GameEvent) -> None:
    """Convenience: fire an event through state's watcher registry."""
    get_watcher_registry(state).fire(event)


# ---------------------------------------------------------------------------
# Convenience fire-helpers (used by rules_engine / triggers)
# ---------------------------------------------------------------------------


def on_spell_cast(state: GameState, player_id: str, source_id: str = "") -> None:
    fire(state, GameEvent(GameEventKind.SPELL_CAST, player_id=player_id, source_id=source_id))


def on_life_lost(state: GameState, player_id: str, amount: int, source_id: str = "") -> None:
    if amount > 0:
        fire(state, GameEvent(GameEventKind.LIFE_LOST, player_id=player_id, amount=amount, source_id=source_id))


def on_life_gained(state: GameState, player_id: str, amount: int, source_id: str = "") -> None:
    if amount > 0:
        fire(state, GameEvent(GameEventKind.LIFE_GAINED, player_id=player_id, amount=amount, source_id=source_id))


def on_damage_dealt(
    state: GameState, source_id: str, target_id: str, amount: int, player_id: str = ""
) -> None:
    if amount > 0:
        fire(state, GameEvent(
            GameEventKind.DAMAGE_DEALT,
            player_id=player_id,
            source_id=source_id,
            target_id=target_id,
            amount=amount,
        ))


def on_creature_attacked(state: GameState, player_id: str, source_id: str = "") -> None:
    fire(state, GameEvent(GameEventKind.CREATURE_ATTACKED, player_id=player_id, source_id=source_id))


def on_land_played(state: GameState, player_id: str, source_id: str = "") -> None:
    fire(state, GameEvent(GameEventKind.LAND_PLAYED, player_id=player_id, source_id=source_id))


def on_card_drawn(state: GameState, player_id: str, source_id: str = "") -> None:
    fire(state, GameEvent(GameEventKind.CARD_DRAWN, player_id=player_id, source_id=source_id))
