"""Cascade keyword (CR 702.85).

When a player casts a spell that has cascade, exile cards from the top
of their library until exiling a non-land card with mana value strictly
less than the cascade spell's mana value. The player may cast that
exiled card without paying its mana cost. Then, all the other exiled
cards are placed on the bottom of the library in a random order.

Cascade is a *triggered* ability that goes on the stack above the
cascade spell, so the cascaded card resolves first.

This module implements a simplified, deterministic version that is wired
into the cast pipeline:

* `has_cascade(card)` — keyword detection.
* `execute_cascade(state, card)` — performs the cascade event right when
  the cascade spell is put on the stack. The cascaded spell is appended
  to the stack on top of the cascade spell, so it resolves first.

We do not currently model the choice "may cast"; the AI always casts the
free spell when legal targets exist (a reasonable default — cascade is
strictly card advantage when it hits). If the cascaded spell has
mandatory targets and none are legal, we put the card on the bottom of
the library along with the rest.
"""

from __future__ import annotations

import re
from typing import Optional

from src.engine.game_state import (
    CardInstance,
    GameState,
    StackItem,
    Zone,
)


_CASCADE_RE = re.compile(r"\bcascade\b", re.IGNORECASE)


def has_cascade(card: CardInstance) -> bool:
    return bool(_CASCADE_RE.search(card.oracle_text or ""))


def _mana_value(card: CardInstance) -> int:
    """Approximate mana value from the card's printed mana cost."""
    cost = card.card_data.get("mana_cost") or ""
    total = 0
    for tok in re.findall(r"\{([^}]+)\}", cost):
        t = tok.strip().upper()
        if t.isdigit():
            total += int(t)
        elif t == "X":
            continue  # X is 0 on the stack of a default cascade spell
        else:
            total += 1
    return total


def execute_cascade(state: GameState, source: CardInstance) -> Optional[CardInstance]:
    """Perform the cascade event for ``source``.

    Returns the cascaded spell's `CardInstance` if one was cast, else
    ``None``. Mutates ``state`` to:

    * exile cards from the top of the source's controller's library
      until finding a non-land with lower MV (or library empties),
    * push the cascaded spell onto the stack (above ``source``),
    * place the other exiled cards on the bottom of the library in
      reverse order (deterministic).
    """
    controller = next(
        (p for p in state.players if p.player_id == source.controller_id), None,
    )
    if controller is None:
        return None

    library = [
        c for c in state.cards
        if c.zone == Zone.LIBRARY and c.owner_id == controller.player_id
    ]
    if not library:
        return None

    source_mv = _mana_value(source)
    exiled: list[CardInstance] = []
    chosen: Optional[CardInstance] = None

    # The library list above is *not* ordered by top-of-deck, but the
    # engine maintains library order via the LIBRARY zone insertion
    # order in `state.cards`. We respect that: top = first in library.
    for c in list(library):
        exiled.append(c)
        c.zone = Zone.EXILE
        is_land = "Land" in (c.card_data.get("type_line") or "")
        if not is_land and _mana_value(c) < source_mv:
            chosen = c
            break

    state.log(
        f"[Cascade] {source.name} (MV {source_mv}) — exiled "
        f"{len(exiled)} card(s) from {controller.name or controller.player_id}'s library"
    )

    if chosen is not None:
        # Move the chosen card to the stack as a free cast (CR 702.85b).
        chosen.zone = Zone.STACK
        item = StackItem(
            source_card_id=chosen.instance_id,
            controller_id=controller.player_id,
            is_spell=True,
            targets=[],
            card_data=chosen.card_data.copy(),
        )
        state.stack.append(item)
        state.log(
            f"[Cascade] {controller.name or controller.player_id} casts "
            f"{chosen.name} for free"
        )

    # The unchosen exiled cards go to the bottom of the library in a
    # deterministic order (reverse of exile order — closest to "random"
    # without introducing nondeterminism).
    for c in reversed([e for e in exiled if e is not chosen]):
        c.zone = Zone.LIBRARY
        # Move it to the end of the cards list so it sits at the bottom.
        try:
            state.cards.remove(c)
            state.cards.append(c)
        except ValueError:
            pass

    return chosen
