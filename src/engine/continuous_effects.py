"""Continuous-effects engine (CR 613).

Every permanent has *characteristics* — name, mana cost, color, type/subtype,
text, power, and toughness.  Continuous effects can change those characteristics.
The Comprehensive Rules require those effects to be applied in a fixed order
(layers 1–7) every time a characteristic is needed.

This module implements that layer pipeline for the characteristics that matter
most to our engine:

  Layer 1  (COPY)       — copy effects (CR 613.1a)
  Layer 2  (CONTROL)    — control-changing (CR 613.1b)
  Layer 3  (TEXT)       — text-changing (CR 613.1c)
  Layer 4  (TYPE)       — type / subtype / supertype (CR 613.1d)
  Layer 5  (COLOR)      — color-changing (CR 613.1e)
  Layer 6  (ABILITY)    — adding / removing abilities (CR 613.1f)
  Layer 7a (PT_CDA)     — CDA-defined P/T (CR 613.4a)
  Layer 7b (PT_SET)     — P/T-setting effects (CR 613.4b)
  Layer 7c (PT_MOD)     — P/T modifiers / anthems (CR 613.4c)
  Layer 7d (PT_COUNTER) — +1/+1 and -1/-1 counter sums (CR 613.4d)
  Layer 7e (PT_SWITCH)  — switch P/T (CR 613.4e)

Usage
-----
The ``ContinuousEffectsRegistry`` lives on ``GameState`` (lazily created
via ``get_registry(state)``).  Call ``register(state, effect)`` when a
permanent with a continuous static ability enters the battlefield, and
``expire_for_card(state, card_id)`` when it leaves.

To read a permanent's current P/T call ``effective_power(card, state)`` /
``effective_toughness(card, state)`` from this module.

Reference: CR 613. XMage: ContinuousEffects.java (MIT, magefree/mage).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .game_state import CardInstance, GameState


# ---------------------------------------------------------------------------
# Layer enum (CR 613)
# ---------------------------------------------------------------------------

class Layer(IntEnum):
    COPY = 1
    CONTROL = 2
    TEXT = 3
    TYPE = 4
    COLOR = 5
    ABILITY = 6
    PT_CDA = 70       # characteristic-defining ability
    PT_SET = 71       # sets base P/T to a fixed value
    PT_MOD = 72       # modifies P/T by +N/+M  (anthems, pumps)
    PT_COUNTER = 73   # counter-based delta (+1/+1 / -1/-1)
    PT_SWITCH = 74    # switches P/T


# ---------------------------------------------------------------------------
# Effect dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContinuousEffect:
    """A single continuous effect (CR 611.3).

    Parameters
    ----------
    layer:           Which layer (see above).
    source_id:       ``CardInstance.instance_id`` of the source permanent/spell.
    controller_id:   Player who controls the source.
    timestamp:       Monotonically increasing integer (``next_timestamp(state)``).
    duration:        ``"permanent"``, ``"end_of_turn"``, or ``"until_leaves"``.
    apply:           Callable that *mutates* the snapshot in-place.
    target_filter:   Called before ``apply``; skip if returns False.
    description:     Human-readable label for debugging.
    """

    layer: Layer
    source_id: str
    controller_id: str
    timestamp: int
    apply: Callable[[CardInstance, GameState], None]
    duration: str = "permanent"
    target_filter: Callable[[CardInstance, GameState], bool] = field(
        default_factory=lambda: (lambda _c, _s: True)
    )
    description: str = ""

    def __hash__(self) -> int:
        return hash((self.source_id, self.layer, self.timestamp))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContinuousEffect):
            return NotImplemented
        return (self.source_id, self.layer, self.timestamp) == (
            other.source_id, other.layer, other.timestamp
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ContinuousEffectsRegistry:
    """Holds all active continuous effects and applies them in CR 613 order."""

    def __init__(self) -> None:
        self._effects: list[ContinuousEffect] = []

    # --- registration ---

    def register(self, effect: ContinuousEffect) -> None:
        self._effects.append(effect)

    def expire_for_card(self, source_id: str) -> None:
        """Remove all effects whose source is ``source_id`` (LTB / exile)."""
        self._effects = [e for e in self._effects if e.source_id != source_id]

    def expire_end_of_turn(self) -> None:
        """Remove effects with ``duration="end_of_turn"`` (called at cleanup)."""
        self._effects = [e for e in self._effects if e.duration != "end_of_turn"]

    # --- applying ---

    def _sorted(self) -> list[ContinuousEffect]:
        return sorted(self._effects, key=lambda e: (int(e.layer), e.timestamp))

    def apply_to(self, card: CardInstance, state: GameState) -> CardInstance:
        """Return a *characteristics snapshot* of ``card`` with all applicable
        continuous effects applied in CR 613 order.

        The snapshot is a shallow copy; do **not** write it back into
        ``state.cards`` — it is a read-only view of effective characteristics.
        """
        snap = _shallow_copy_card(card)
        for effect in self._sorted():
            try:
                if effect.target_filter(snap, state):
                    effect.apply(snap, state)
            except Exception:
                pass  # never crash the game over a layer misfire
        return snap

    def all_effects(self) -> list[ContinuousEffect]:
        return list(self._effects)


# ---------------------------------------------------------------------------
# GameState helpers
# ---------------------------------------------------------------------------

def get_registry(state: GameState) -> ContinuousEffectsRegistry:
    """Lazily create and return the registry on ``state``."""
    reg: Optional[ContinuousEffectsRegistry] = getattr(
        state, "_continuous_effects_registry", None
    )
    if reg is None or not isinstance(reg, ContinuousEffectsRegistry):
        reg = ContinuousEffectsRegistry()
        state._continuous_effects_registry = reg
    return reg


def register(state: GameState, effect: ContinuousEffect) -> None:
    get_registry(state).register(effect)


def expire_for_card(state: GameState, card_id: str) -> None:
    get_registry(state).expire_for_card(card_id)


def expire_end_of_turn(state: GameState) -> None:
    get_registry(state).expire_end_of_turn()


def apply_to(card: CardInstance, state: GameState) -> CardInstance:
    return get_registry(state).apply_to(card, state)


def next_timestamp(state: GameState) -> int:
    """Monotonically incrementing counter for ordering effects by timestamp."""
    ts: int = getattr(state, "_ce_timestamp", 0)
    state._ce_timestamp = ts + 1
    return ts


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------

def _shallow_copy_card(card: CardInstance) -> CardInstance:
    snap = copy.copy(card)
    snap.card_data = dict(card.card_data)
    return snap


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_anthem_effect(
    source_id: str,
    controller_id: str,
    timestamp: int,
    power_bonus: int,
    toughness_bonus: int,
    filter_fn: Callable[[CardInstance, GameState], bool],
    duration: str = "permanent",
    description: str = "",
) -> ContinuousEffect:
    """Layer-7c effect: +N/+M to all permanents matching ``filter_fn``."""

    def _apply(snap: CardInstance, state: GameState) -> None:
        snap.card_data["_ce_power_mod"] = (
            snap.card_data.get("_ce_power_mod", 0) + power_bonus
        )
        snap.card_data["_ce_toughness_mod"] = (
            snap.card_data.get("_ce_toughness_mod", 0) + toughness_bonus
        )

    return ContinuousEffect(
        layer=Layer.PT_MOD,
        source_id=source_id,
        controller_id=controller_id,
        timestamp=timestamp,
        apply=_apply,
        duration=duration,
        target_filter=filter_fn,
        description=description or f"+{power_bonus}/+{toughness_bonus} anthem",
    )


def make_keyword_grant_effect(
    source_id: str,
    controller_id: str,
    timestamp: int,
    keyword: str,
    filter_fn: Callable[[CardInstance, GameState], bool],
    duration: str = "permanent",
    description: str = "",
) -> ContinuousEffect:
    """Layer-6 effect: grant a keyword to all permanents matching ``filter_fn``."""

    def _apply(snap: CardInstance, state: GameState) -> None:
        existing: list = snap.card_data.get("keywords") or []
        if isinstance(existing, str):
            existing = [existing]
        else:
            existing = list(existing)
        if keyword not in existing:
            existing.append(keyword)
        snap.card_data["keywords"] = existing
        # Append to oracle_text so ``has()`` text-scan picks it up.
        snap.card_data["oracle_text"] = (
            (snap.card_data.get("oracle_text") or "") + f"\n{keyword}"
        )

    return ContinuousEffect(
        layer=Layer.ABILITY,
        source_id=source_id,
        controller_id=controller_id,
        timestamp=timestamp,
        apply=_apply,
        duration=duration,
        target_filter=filter_fn,
        description=description or f"grant {keyword}",
    )


def make_set_pt_effect(
    source_id: str,
    controller_id: str,
    timestamp: int,
    power: int,
    toughness: int,
    filter_fn: Callable[[CardInstance, GameState], bool],
    duration: str = "permanent",
    description: str = "",
) -> ContinuousEffect:
    """Layer-7b effect: set base P/T of matching permanents."""

    def _apply(snap: CardInstance, state: GameState) -> None:
        snap.card_data["power"] = str(power)
        snap.card_data["toughness"] = str(toughness)

    return ContinuousEffect(
        layer=Layer.PT_SET,
        source_id=source_id,
        controller_id=controller_id,
        timestamp=timestamp,
        apply=_apply,
        duration=duration,
        target_filter=filter_fn,
        description=description or f"set P/T to {power}/{toughness}",
    )


def make_eot_pump_effect(
    source_id: str,
    controller_id: str,
    timestamp: int,
    target_id: str,
    power_bonus: int,
    toughness_bonus: int,
    description: str = "",
) -> ContinuousEffect:
    """Layer-7c until-EOT pump on a single card (e.g. Giant Growth)."""

    def _apply(snap: CardInstance, state: GameState) -> None:
        snap.card_data["_ce_power_mod"] = snap.card_data.get("_ce_power_mod", 0) + power_bonus
        snap.card_data["_ce_toughness_mod"] = snap.card_data.get("_ce_toughness_mod", 0) + toughness_bonus

    return ContinuousEffect(
        layer=Layer.PT_MOD,
        source_id=source_id,
        controller_id=controller_id,
        timestamp=timestamp,
        apply=_apply,
        duration="end_of_turn",
        target_filter=lambda c, _s: c.instance_id == target_id,
        description=description or f"+{power_bonus}/+{toughness_bonus} until EOT",
    )


# ---------------------------------------------------------------------------
# Effective P/T — the canonical read path (delegates to layer engine)
# ---------------------------------------------------------------------------

def _parse_pt_int(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def effective_power(card: CardInstance, state: GameState | None = None) -> int:
    """Compute effective power after all continuous effects (CR 613)."""
    if state is not None:
        snap = apply_to(card, state)
    else:
        snap = card

    base = _parse_pt_int(snap.card_data.get("power") or snap.power)
    base += snap.card_data.get("_ce_power_mod", 0)
    plus = snap.counters.get("+1/+1", 0)
    minus = snap.counters.get("-1/-1", 0)
    # Legacy eot / equip fields (will migrate to layer effects over time)
    eot = int(getattr(card, "eot_power_bonus", 0) or 0)
    equip = int(card.counters.get("equip_pwr", 0) or 0)
    prowess = card.counters.get("prowess_eot", 0)
    return base + plus - minus + eot + equip + prowess


def effective_toughness(card: CardInstance, state: GameState | None = None) -> int:
    """Compute effective toughness after all continuous effects."""
    if state is not None:
        snap = apply_to(card, state)
    else:
        snap = card

    base = _parse_pt_int(snap.card_data.get("toughness") or snap.toughness)
    base += snap.card_data.get("_ce_toughness_mod", 0)
    plus = snap.counters.get("+1/+1", 0)
    minus = snap.counters.get("-1/-1", 0)
    eot = int(getattr(card, "eot_toughness_bonus", 0) or 0)
    equip = int(card.counters.get("equip_tou", 0) or 0)
    return base + plus - minus + eot + equip


# ---------------------------------------------------------------------------
# Auto-install static anthem / keyword-grant effects at ETB
# ---------------------------------------------------------------------------

_ANTHEM_RE = re.compile(
    r"(?:other\s+)?(?:(?P<subtype>[A-Z][a-z]+)s?\s+)?creatures you control"
    r"(?:\s+and\s+other\s+\S+)?\s+get\s+\+(?P<p>\d+)/\+(?P<t>\d+)",
    re.IGNORECASE,
)

_KW_GRANT_RE = re.compile(
    r"(?:other\s+)?(?:(?P<subtype>[A-Z][a-z]+)s?\s+)?creatures you control"
    r"(?:\s+and\s+other\s+\S+)?\s+have\s+(?P<kw>[\w ]+)",
    re.IGNORECASE,
)


def auto_install_effects(state: GameState, card: CardInstance) -> None:
    """Parse oracle text for common static abilities and register layer effects.

    Called after a permanent enters the battlefield.
    """
    text = card.card_data.get("oracle_text") or ""
    ts = next_timestamp(state)

    # Anthem: "creatures you control get +1/+1"
    for m in _ANTHEM_RE.finditer(text):
        p_bonus = int(m.group("p"))
        t_bonus = int(m.group("t"))
        subtype_filter = (m.group("subtype") or "").lower().rstrip("s")
        ctrl = card.controller_id
        src_id = card.instance_id

        def _f(c: CardInstance, s: GameState,
               _ctrl=ctrl, _sub=subtype_filter, _src=src_id) -> bool:
            if c.zone.value != "battlefield":
                return False
            if c.controller_id != _ctrl:
                return False
            if not c.is_creature():
                return False
            if c.instance_id == _src:
                return False
            if _sub and _sub not in (c.type_line or "").lower():
                return False
            return True

        register(state, make_anthem_effect(
            source_id=card.instance_id,
            controller_id=card.controller_id,
            timestamp=ts,
            power_bonus=p_bonus,
            toughness_bonus=t_bonus,
            filter_fn=_f,
            description=f"{card.name}: +{p_bonus}/+{t_bonus} to your creatures",
        ))

    # Keyword grant: "creatures you control have flying"
    for m in _KW_GRANT_RE.finditer(text):
        kw_raw = (m.group("kw") or "").strip().lower()
        kw = kw_raw.split(" and ")[0].strip()
        if not kw or len(kw) > 30:
            continue
        subtype_filter = (m.group("subtype") or "").lower().rstrip("s")
        ctrl = card.controller_id

        def _g(c: CardInstance, s: GameState,
               _ctrl=ctrl, _sub=subtype_filter) -> bool:
            if c.zone.value != "battlefield":
                return False
            if c.controller_id != _ctrl:
                return False
            if not c.is_creature():
                return False
            if _sub and _sub not in (c.type_line or "").lower():
                return False
            return True

        register(state, make_keyword_grant_effect(
            source_id=card.instance_id,
            controller_id=card.controller_id,
            timestamp=ts,
            keyword=kw,
            filter_fn=_g,
            description=f"{card.name}: grant {kw} to your creatures",
        ))
