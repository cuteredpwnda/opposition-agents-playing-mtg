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


def create_gold_token(state: GameState, controller_id: str) -> CardInstance:
    """Gold: Artifact — Gold. ``Sacrifice this: Add one mana of any color.`` (Theros-era)"""
    return create_token(
        state, controller_id,
        name="Gold Token",
        type_line="Token Artifact \u2014 Gold",
        oracle_text="Sacrifice this artifact: Add one mana of any color.",
    )


def create_map_token(state: GameState, controller_id: str) -> CardInstance:
    """Map: Artifact — Map. ``{1}, {T}, Sacrifice this: Look at the top card of your library.``"""
    return create_token(
        state, controller_id,
        name="Map Token",
        type_line="Token Artifact \u2014 Map",
        oracle_text="{1}, {T}, Sacrifice this artifact: Look at the top card of your library. You may put it on the bottom.",
    )


def create_powerstone_token(state: GameState, controller_id: str) -> CardInstance:
    """Powerstone: Artifact. ``{T}: Add {C}. This mana can't be spent to cast nonartifact spells.``"""
    return create_token(
        state, controller_id,
        name="Powerstone Token",
        type_line="Token Artifact \u2014 Powerstone",
        oracle_text="{T}: Add {C}. This mana can't be spent to cast nonartifact spells.",
    )


def create_shard_token(state: GameState, controller_id: str) -> CardInstance:
    """Shard: Enchantment. ``{2}, Sacrifice this: Scry 1, then draw a card.``"""
    return create_token(
        state, controller_id,
        name="Shard Token",
        type_line="Token Enchantment \u2014 Shard",
        oracle_text="{2}, Sacrifice this enchantment: Scry 1, then draw a card.",
    )


def create_incubator_token(
    state: GameState, controller_id: str, charge: int = 0
) -> CardInstance:
    """Incubator: Artifact. ``{2}: Transform.`` Transforms into a 0/0 Phyrexian creature."""
    token = create_token(
        state, controller_id,
        name="Incubator Token",
        type_line="Token Artifact \u2014 Incubator",
        oracle_text="{2}: Transform this artifact.",
    )
    if charge > 0:
        token.counters["+1/+1"] = charge
    return token


def create_walker_emblem(
    state: GameState, controller_id: str, planeswalker_name: str, effect_text: str
) -> CardInstance:
    """Create an emblem for a planeswalker ultimate (stored in exile zone as token)."""
    from .game_state import CardInstance as CI
    import uuid
    emblem = CI(
        instance_id=f"emblem_{planeswalker_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}",
        card_data={
            "name": f"{planeswalker_name} Emblem",
            "type_line": "Emblem",
            "oracle_text": effect_text,
            "is_token": True,
        },
        zone=Zone.COMMAND_ZONE,
        owner_id=controller_id,
        controller_id=controller_id,
    )
    emblem.is_token = True
    state.cards.append(emblem)
    state.log(f"{controller_id} gets an emblem: {effect_text[:60]}")
    return emblem


def create_copy_token(
    state: GameState, controller_id: str, original: CardInstance
) -> CardInstance:
    """Create a token that is a copy of ``original`` (CR 707.2)."""
    copy = create_token(
        state, controller_id,
        name=original.name,
        type_line=original.type_line,
        power=original.card_data.get("power"),
        toughness=original.card_data.get("toughness"),
        oracle_text=original.oracle_text,
        keywords=[k for k in (original.card_data.get("keywords") or [])],
    )
    # Copy counters from original (tokens don't normally enter with these, but explicit copies do)
    copy.counters = dict(original.counters)
    return copy


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
