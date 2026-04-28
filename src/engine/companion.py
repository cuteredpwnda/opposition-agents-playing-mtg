"""Companion (CR 702.139) — outside-the-game start, then command zone.

Companion is a deckbuilding restriction + game-start ability. At the
beginning of the game, before any player draws their opening hand, a
player whose deck satisfies a companion's restriction may *reveal* that
card from their sideboard. Once revealed, the companion lives logically
in the **outside-the-game** zone (which we represent here with a
:class:`~src.engine.game_state.CommandZoneObject` of ``kind="companion"``).
Once during the game, that player may pay an extra ``{3}`` to put the
companion into their hand from outside the game; from then on it behaves
like a normal card in hand.

This module is **operational** only — it provides:

* :func:`reveal_companion` — register a companion at game start.
* :func:`pay_companion_tax` — move the companion from the outside-the-game
  marker into the player's hand once they pay the extra ``{3}``.

The deck-construction restriction itself (e.g. Lurrus = "permanents have
mana value 2 or less") is enforced at deck-load time elsewhere.
"""

from __future__ import annotations

import uuid
from typing import Optional

from src.engine.command_zone import (
    add_command_zone_object,
    command_zone_objects_for,
)
from src.engine.game_state import CardInstance, GameState, Zone


COMPANION_TAX = 3  # CR 702.139c — generic mana surcharge


def reveal_companion(
    state: GameState,
    controller_id: str,
    companion_card_data: dict,
) -> CardInstance:
    """Register a companion for ``controller_id`` (CR 702.139b).

    Creates a :class:`CardInstance` for the companion, places it in
    ``Zone.EXILE`` with metadata flag ``card_data['companion'] = True`` to
    represent the "outside-the-game" zone, and adds a tracking
    :class:`CommandZoneObject` so the priority loop / UI can surface the
    companion as a known game object.
    """
    instance_id = f"{controller_id}_companion_{uuid.uuid4().hex[:8]}"
    card = CardInstance(
        instance_id=instance_id,
        card_data=dict(companion_card_data, companion=True),
        zone=Zone.EXILE,  # outside-the-game proxy; not in any normal zone
        owner_id=controller_id,
        controller_id=controller_id,
    )
    state.cards.append(card)
    add_command_zone_object(
        state,
        controller_id=controller_id,
        kind="companion",
        name=card.name,
        card_data={"instance_id": instance_id, "tax_paid": False},
    )
    state.log(f"  ⊕ Companion revealed: {card.name} for {controller_id}")
    return card


def get_companion(state: GameState, controller_id: str) -> Optional[CardInstance]:
    objs = command_zone_objects_for(state, controller_id, kind="companion")
    if not objs:
        return None
    iid = objs[0].card_data.get("instance_id")
    return next((c for c in state.cards if c.instance_id == iid), None)


def pay_companion_tax(state: GameState, controller_id: str) -> Optional[CardInstance]:
    """Move the companion into ``controller_id``'s hand (CR 702.139c).

    The mana payment itself is handled by the caller (rules engine /
    spell-cast path) — this helper just performs the zone change and
    flips the ``tax_paid`` marker so the ability can't be used twice.

    Returns the companion :class:`CardInstance` if the move happened, or
    ``None`` if the player has no companion / already paid the tax.
    """
    objs = command_zone_objects_for(state, controller_id, kind="companion")
    if not objs:
        return None
    obj = objs[0]
    if obj.card_data.get("tax_paid"):
        return None
    iid = obj.card_data.get("instance_id")
    card = next((c for c in state.cards if c.instance_id == iid), None)
    if card is None:
        return None
    card.zone = Zone.HAND
    obj.card_data["tax_paid"] = True
    state.log(f"  ⊕ {controller_id} pays {{3}}: companion {card.name} → hand")
    return card
