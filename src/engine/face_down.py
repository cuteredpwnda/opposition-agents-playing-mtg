"""Face-down spells and permanents — full characteristic-replacement (CR 707).

This module owns the *characteristics* side of morph/megamorph/manifest:

* A face-down permanent has no name, no mana cost, no color, no subtypes,
  no rules text, no abilities, and is a 2/2 colorless creature (CR 707.2).
* Turning face up restores the original characteristics and fires
  "when ~ is turned face up" triggers (CR 702.36e).
* Megamorph adds a +1/+1 counter on flip (CR 702.99).
* Manifest (CR 701.32) turns ANY card face-down with no morph cost; can
  only be turned face up if it is a creature card (by paying its mana cost).

Implementation
--------------
1. ``cast_face_down`` / ``manifest`` store the original ``card_data`` as
   ``card._face_up_data`` and replace ``card_data`` with a 2/2-stub.
   A ``Layer.COPY`` effect is registered in the continuous-effects registry
   so that ``effective_power`` / ``effective_toughness`` always see 2/2
   while the card is face-down.

2. ``turn_face_up`` pays the morph cost (or creature mana cost for manifest),
   removes the Layer effect, restores ``_face_up_data``, and fires triggers.

3. All existing ``alternate_costs.cast_face_down`` / ``turn_face_up`` code
   continues to work — this module calls through to it and supplements with
   the layer engine.

Reference: CR 707 (face-down permanents), CR 702.36 (morph), CR 702.99
(megamorph), CR 701.32 (manifest).
XMage: BecomesFaceDownCreatureEffect.java (MIT, magefree/mage).
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Optional

from .game_state import CardInstance, GameState, Zone


# ---------------------------------------------------------------------------
# FaceDownMode enum
# ---------------------------------------------------------------------------

class FaceDownMode(str, Enum):
    MORPHED = "morphed"
    MEGAMORPHED = "megamorphed"
    MANIFESTED = "manifested"
    DISGUISED = "disguised"   # MKM cloak
    CLOAKED = "cloaked"       # WOE


# ---------------------------------------------------------------------------
# Internal characteristic stub for face-down objects (CR 707.2)
# ---------------------------------------------------------------------------

_FACE_DOWN_STUB: dict = {
    "name": "",
    "mana_cost": "",
    "cmc": 0.0,
    "colors": [],
    "color_identity": [],
    "type_line": "Creature",
    "subtypes": [],
    "supertypes": [],
    "oracle_text": "",
    "keywords": [],
    "power": "2",
    "toughness": "2",
    "_face_down": True,
}


def _apply_face_down_stub(card: CardInstance, mode: FaceDownMode) -> None:
    """Replace card.card_data with a 2/2-colorless-creature stub.

    The original ``card_data`` is preserved in ``card._face_up_data``.
    """
    card._face_up_data = copy.deepcopy(card.card_data)      # type: ignore[attr-defined]
    card._face_down_mode = mode                              # type: ignore[attr-defined]
    # Overwrite with the stub (keep instance-level fields like instance_id)
    card.card_data = dict(_FACE_DOWN_STUB)
    card.face_down = True


def _restore_face_up_data(card: CardInstance) -> None:
    """Restore original card_data from the saved face-up data."""
    saved: dict = getattr(card, "_face_up_data", None)  # type: ignore[attr-defined]
    if saved:
        card.card_data = saved
        del card._face_up_data       # type: ignore[attr-defined]
    card.face_down = False
    mode = getattr(card, "_face_down_mode", None)
    if mode:
        del card._face_down_mode     # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Register / expire Layer 1 (COPY) effect for the face-down object
# ---------------------------------------------------------------------------

def _register_face_down_layer(state: GameState, card: CardInstance) -> None:
    """Install a Layer 1 (COPY) effect so effective_* reads 2/2 even if some
    other effect would try to read the original characteristics."""
    try:
        from .continuous_effects import (
            ContinuousEffect, Layer, next_timestamp, register,
        )

        def _apply_layer(snap: CardInstance, _state: GameState) -> None:
            snap.card_data["power"] = "2"
            snap.card_data["toughness"] = "2"
            snap.card_data["oracle_text"] = ""
            snap.card_data["keywords"] = []

        eff = ContinuousEffect(
            layer=Layer.COPY,
            source_id=card.instance_id,
            controller_id=card.controller_id,
            timestamp=next_timestamp(state),
            apply=_apply_layer,
            duration="permanent",
            target_filter=lambda c, _s: c.instance_id == card.instance_id,
            description=f"Face-down 2/2 ({card.instance_id})",
        )
        register(state, eff)
    except Exception:
        pass  # Layer system not yet loaded; the stub in card_data still works


def _expire_face_down_layer(state: GameState, card: CardInstance) -> None:
    """Remove the face-down Layer 1 effect so the card's real characteristics show."""
    try:
        from .continuous_effects import expire_for_card
        expire_for_card(state, card.instance_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API — cast face-down (morph / megamorph)
# ---------------------------------------------------------------------------

def cast_face_down(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Cast a morph/megamorph card face-down for {3} (CR 702.36b).

    Delegates to ``alternate_costs.cast_face_down`` for the mana/zone
    handling, then installs the Layer 1 characteristic-replacement effect.
    """
    from .alternate_costs import (
        cast_face_down as _ac_cast_fd,
        has_megamorph,
    )
    mode = FaceDownMode.MEGAMORPHED if has_megamorph(card) else FaceDownMode.MORPHED
    ok = _ac_cast_fd(state, caster_id, card)
    if ok:
        card._face_up_data = {}  # alternate_costs already saved P/T keys; mark mode  # type: ignore
        card._face_down_mode = mode  # type: ignore
        card.face_down = True
        _register_face_down_layer(state, card)
    return ok


# ---------------------------------------------------------------------------
# Public API — manifest
# ---------------------------------------------------------------------------

def manifest(state: GameState, controller_id: str, card: CardInstance) -> None:
    """Manifest a card face-down onto the battlefield (CR 701.32).

    The card enters as a 2/2 colorless face-down creature with no abilities.
    It can only be turned face up by paying its mana cost IF it is a creature
    card (CR 701.32c).
    """
    from .zones import move_card
    _apply_face_down_stub(card, FaceDownMode.MANIFESTED)
    if card.zone != Zone.BATTLEFIELD:
        move_card(state, card.instance_id, card.zone, Zone.BATTLEFIELD, controller_id)
    card.summoning_sick = True
    _register_face_down_layer(state, card)
    state.log(f"Manifest: a card enters the battlefield face-down as a 2/2")


# ---------------------------------------------------------------------------
# Public API — turn face-up
# ---------------------------------------------------------------------------

def turn_face_up(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Turn a face-down permanent face-up.

    For morph/megamorph: pay the morph cost (delegates to
    ``alternate_costs.turn_face_up``).
    For manifest: pay the card's printed mana cost (only if it is a creature).

    Returns True on success.
    """
    mode: FaceDownMode = getattr(card, "_face_down_mode", FaceDownMode.MORPHED)  # type: ignore

    if mode == FaceDownMode.MANIFESTED:
        return _turn_up_manifested(state, caster_id, card)
    else:
        return _turn_up_morph(state, caster_id, card)


def _turn_up_morph(state: GameState, caster_id: str, card: CardInstance) -> bool:
    from .alternate_costs import turn_face_up as _ac_tfu
    mode: FaceDownMode = getattr(card, "_face_down_mode", FaceDownMode.MORPHED)  # type: ignore
    ok = _ac_tfu(state, caster_id, card)
    if ok:
        _expire_face_down_layer(state, card)
        card.face_down = False
        # alternate_costs already restored P/T keys; fire "face-up" triggers
        _fire_face_up_triggers(state, card)
    return ok


def _turn_up_manifested(state: GameState, caster_id: str, card: CardInstance) -> bool:
    """Turn up a manifested card by paying its mana cost (creatures only)."""
    from .mana import auto_tap_for_cost, pay_cost
    saved: dict = getattr(card, "_face_up_data", None)  # type: ignore
    if not saved:
        return False
    # Must be a creature card to turn face-up via manifest (CR 701.32c)
    if "Creature" not in (saved.get("type_line") or ""):
        state.log("Cannot turn manifest face-up: not a creature card")
        return False
    player = next((p for p in state.players if p.player_id == caster_id), None)
    if player is None:
        return False
    # Build cost from mana_cost string
    from .mana import parse_mana_cost
    mana_cost_str = saved.get("mana_cost") or ""
    cost = parse_mana_cost(mana_cost_str)
    if not auto_tap_for_cost(state, player, cost):
        state.log(f"Cannot turn manifest face-up: insufficient mana for {mana_cost_str}")
        return False
    pay_cost(player, cost)
    _restore_face_up_data(card)
    _expire_face_down_layer(state, card)
    _fire_face_up_triggers(state, card)
    state.log(f"Manifest: {card.name} turned face-up")
    return True


def _fire_face_up_triggers(state: GameState, card: CardInstance) -> None:
    """Fire 'when CARDNAME is turned face up' triggered abilities."""
    text = (card.oracle_text or "").lower()
    if "turned face up" in text:
        from .game_state import StackItem, Trigger, TriggerType
        trigger = Trigger(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            trigger_type=TriggerType.ENTERS_BATTLEFIELD,
            description=f"{card.name}: turned face up trigger",
        )
        item = StackItem(
            source_card_id=card.instance_id,
            controller_id=card.controller_id,
            is_spell=False,
            card_data={"name": f"[Trigger] {card.name} turned face up", "type_line": "Ability"},
        )
        state.stack.append(item)
        state.triggered_abilities.append(trigger)
        state.log(f"[TRIGGER] {card.name} turned face up")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def is_face_down(card: CardInstance) -> bool:
    """True if the card is currently face-down on the battlefield."""
    return bool(card.face_down or card.card_data.get("_face_down"))


def face_down_mode(card: CardInstance) -> Optional[FaceDownMode]:
    mode = getattr(card, "_face_down_mode", None)  # type: ignore
    return FaceDownMode(mode) if mode else None


def can_be_turned_face_up(card: CardInstance) -> bool:
    """True if the card has a valid turn-face-up method available."""
    mode = face_down_mode(card)
    if mode is None:
        return False
    if mode in (FaceDownMode.MORPHED, FaceDownMode.MEGAMORPHED):
        from .alternate_costs import morph_cost
        return True  # morph_cost may be {} (free)
    if mode == FaceDownMode.MANIFESTED:
        saved: dict = getattr(card, "_face_up_data", {})  # type: ignore
        return "Creature" in (saved.get("type_line") or "")
    return False
