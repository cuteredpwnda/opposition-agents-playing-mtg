"""Token factory (CR 111).

Creates token CardInstances directly on the battlefield. All tokens are
flagged with ``card_data['is_token'] = True`` so the SBA loop in
:mod:`rules_engine` removes them when they leave the battlefield.

Predefined helpers cover the common token archetypes referenced by
hundreds of cards in the pool: Treasure, Food, Clue, Blood, plus a
generic creature-token spawner.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .game_state import CardInstance, GameState, Zone


# ---------------------------------------------------------------------------
# Generic spawner
# ---------------------------------------------------------------------------


def create_token(
    state: GameState,
    controller_id: str,
    *,
    name: str,
    type_line: str,
    power: Optional[str] = None,
    toughness: Optional[str] = None,
    oracle_text: str = "",
    keywords: Optional[Iterable[str]] = None,
    colors: Optional[Iterable[str]] = None,
    enters_tapped: bool = False,
) -> CardInstance:
    """Spawn a token directly onto ``controller_id``'s battlefield."""
    color_list = list(colors or [])
    card_data = {
        "name": name,
        "type_line": type_line,
        "mana_cost": "",
        "cmc": 0,
        "oracle_text": oracle_text,
        "is_token": True,
        "keywords": list(keywords or []),
        "colors": color_list,
        # color_identity drives ETB filters like "another red creature
        # enters the battlefield" (Foundry Street Denizen, etc.).
        "color_identity": color_list,
    }
    if power is not None:
        card_data["power"] = power
    if toughness is not None:
        card_data["toughness"] = toughness

    tok = CardInstance(
        card_data=card_data,
        zone=Zone.BATTLEFIELD,
        owner_id=controller_id,
        controller_id=controller_id,
        tapped=enters_tapped,
        summoning_sick=("Creature" in type_line),
        turn_entered=state.turn_number,
    )
    state.cards.append(tok)
    # Fire ETB triggers (CR 603.6a) — tokens enter the battlefield like any
    # other permanent and trigger "whenever (another) X enters" abilities.
    _fire_token_etb_triggers(state, tok)
    return tok


def _fire_token_etb_triggers(state: GameState, tok: CardInstance) -> None:
    """Push ETB triggers caused by ``tok`` entering the battlefield onto
    the stack. Lazy-imported to avoid cycles with ``triggers``."""
    try:
        from .triggers import check_enters_battlefield_triggers
        from .game_state import StackItem
    except Exception:
        return
    etb = check_enters_battlefield_triggers(state, tok)
    for trig in etb:
        item = StackItem(
            source_card_id=trig.source_card_id,
            controller_id=trig.controller_id,
            is_spell=False,
            card_data={
                "name": f"[Trigger] {trig.description}",
                "type_line": "Ability",
            },
        )
        state.stack.append(item)
        state.triggered_abilities.append(trig)
        state.log(f"[TRIGGER (ETB)] {trig.description} added to stack")


# ---------------------------------------------------------------------------
# Predefined token archetypes
# ---------------------------------------------------------------------------


def create_treasure_token(state: GameState, controller_id: str) -> CardInstance:
    """Treasure: Artifact — Treasure. ``{T}, Sacrifice this: Add one mana of any color.``"""
    return create_token(
        state, controller_id,
        name="Treasure Token",
        type_line="Token Artifact \u2014 Treasure",
        oracle_text="{T}, Sacrifice this artifact: Add one mana of any color.",
    )


def create_food_token(state: GameState, controller_id: str) -> CardInstance:
    """Food: Artifact — Food. ``{2}, {T}, Sacrifice this: You gain 3 life.``"""
    return create_token(
        state, controller_id,
        name="Food Token",
        type_line="Token Artifact \u2014 Food",
        oracle_text="{2}, {T}, Sacrifice this artifact: You gain 3 life.",
    )


def create_clue_token(state: GameState, controller_id: str) -> CardInstance:
    """Clue: Artifact — Clue. ``{2}, Sacrifice this: Draw a card.``"""
    return create_token(
        state, controller_id,
        name="Clue Token",
        type_line="Token Artifact \u2014 Clue",
        oracle_text="{2}, Sacrifice this artifact: Draw a card.",
    )


def create_blood_token(state: GameState, controller_id: str) -> CardInstance:
    """Blood: Artifact — Blood. ``{1}, {T}, Discard a card, Sacrifice: Draw a card.``"""
    return create_token(
        state, controller_id,
        name="Blood Token",
        type_line="Token Artifact \u2014 Blood",
        oracle_text="{1}, {T}, Discard a card, Sacrifice this artifact: Draw a card.",
    )


def create_creature_token(
    state: GameState,
    controller_id: str,
    *,
    power: int,
    toughness: int,
    subtypes: Iterable[str] = ("Spirit",),
    keywords: Optional[Iterable[str]] = None,
    colors: Optional[Iterable[str]] = None,
) -> CardInstance:
    sub = " ".join(subtypes)
    return create_token(
        state, controller_id,
        name=f"{sub} Token",
        type_line=f"Token Creature \u2014 {sub}",
        power=str(power),
        toughness=str(toughness),
        keywords=keywords,
        colors=colors,
    )


def create_treasure_tokens(state: GameState, controller_id: str, count: int) -> list[CardInstance]:
    return [create_treasure_token(state, controller_id) for _ in range(count)]


def sacrifice_for_mana(
    state: GameState, treasure: CardInstance, color: str = "C",
) -> bool:
    """Activate Treasure: tap, sacrifice, add 1 mana of chosen color.

    Returns True on success.
    """
    if treasure.zone != Zone.BATTLEFIELD or treasure.tapped:
        return False
    if "Treasure" not in treasure.type_line:
        return False
    player = next((p for p in state.players if p.player_id == treasure.controller_id), None)
    if player is None:
        return False
    color = color.upper()
    if color not in {"W", "U", "B", "R", "G", "C"}:
        color = "C"
    treasure.tapped = True
    treasure.zone = Zone.GRAVEYARD
    player.mana_pool[color] = player.mana_pool.get(color, 0) + 1
    state.log(f"{player.name} sacrifices Treasure for {{{color}}}")
    return True
