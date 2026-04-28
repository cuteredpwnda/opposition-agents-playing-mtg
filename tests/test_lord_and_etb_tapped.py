"""Lord effects: subtype-restricted P/T and keyword grants."""

from __future__ import annotations

from src.engine.game_state import (
    CardInstance,
    GameState,
    PlayerState,
    Phase,
    Zone,
)
from src.engine.static_abilities import (
    has_keyword,
    get_effective_power_toughness,
    parse_static_abilities,
)


def _state_with(*cards: CardInstance) -> GameState:
    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    state.cards = list(cards)
    return state


def _goblin(name: str = "Goblin Soldier", *, owner: str = "A") -> CardInstance:
    return CardInstance(
        instance_id=f"{owner}_{name}".replace(" ", "_"),
        card_data={
            "name": name,
            "type_line": "Creature — Goblin",
            "power": "1",
            "toughness": "1",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id=owner,
        controller_id=owner,
    )


def test_subtype_pt_lord():
    lord = CardInstance(
        instance_id="A_lord",
        card_data={
            "name": "Goblin King",
            "type_line": "Creature — Goblin",
            "power": "2",
            "toughness": "2",
            "oracle_text": "Goblin creatures you control get +1/+1.",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    grunt = _goblin()
    state = _state_with(lord, grunt)
    p, t = get_effective_power_toughness(grunt, state)
    assert (p, t) == (2, 2)


def test_subtype_keyword_grant():
    """`Goblins you control have haste` grants haste to all goblins."""
    krenko = CardInstance(
        instance_id="A_krenko_warlord",
        card_data={
            "name": "Krenko, Tin Street Kingpin",
            "type_line": "Creature — Goblin",
            "power": "1",
            "toughness": "2",
            "oracle_text": "Goblins you control have haste.",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    grunt = _goblin()
    grunt.summoning_sick = True
    state = _state_with(krenko, grunt)
    assert has_keyword(grunt, state, "haste")


def test_subtype_keyword_does_not_apply_to_other_types():
    krenko = CardInstance(
        instance_id="A_krenko",
        card_data={
            "name": "Krenko",
            "type_line": "Creature — Goblin",
            "power": "1",
            "toughness": "2",
            "oracle_text": "Goblins you control have haste.",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    elf = CardInstance(
        instance_id="A_elf",
        card_data={
            "name": "Llanowar Elves",
            "type_line": "Creature — Elf",
            "power": "1",
            "toughness": "1",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    state = _state_with(krenko, elf)
    assert not has_keyword(elf, state, "haste")


def test_subtype_keyword_does_not_apply_to_opponents():
    krenko = CardInstance(
        instance_id="A_krenko",
        card_data={
            "name": "Krenko",
            "type_line": "Creature — Goblin",
            "power": "1",
            "toughness": "2",
            "oracle_text": "Goblins you control have haste.",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    foe = _goblin(name="Foe Goblin", owner="B")
    state = _state_with(krenko, foe)
    assert not has_keyword(foe, state, "haste")


def test_parse_keyword_subtype_grant():
    krenko = CardInstance(
        instance_id="A_krenko",
        card_data={
            "name": "Krenko",
            "type_line": "Creature — Goblin",
            "oracle_text": "Goblins you control have haste.",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="A",
        controller_id="A",
    )
    abilities = parse_static_abilities(krenko)
    assert any(
        a.effect_type == "keyword"
        and "haste" in (a.keywords or [])
        and a.scope == "subtype_creatures_you_control"
        and a.subtype_filter == "goblin"
        for a in abilities
    ), abilities


def test_enters_tapped_creature_resolves_tapped():
    """Non-land permanent with 'enters tapped' should resolve tapped."""
    from src.engine.rules_engine import RulesEngine
    from src.engine.game_state import StackItem
    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    creature = CardInstance(
        instance_id="A_thopter",
        card_data={
            "name": "Mistmeadow Witch",
            "type_line": "Creature — Kithkin Wizard",
            "power": "2",
            "toughness": "2",
            "oracle_text": "Mistmeadow Witch enters the battlefield tapped.",
        },
        zone=Zone.STACK,
        owner_id="A",
        controller_id="A",
    )
    state.cards.append(creature)
    state.stack.append(StackItem(
        source_card_id="A_thopter",
        controller_id="A",
        is_spell=True,
        card_data=creature.card_data,
    ))
    engine = RulesEngine()
    state = engine.resolve_stack_item(state)
    refreshed = next(c for c in state.cards if c.instance_id == "A_thopter")
    assert refreshed.zone == Zone.BATTLEFIELD
    assert refreshed.tapped is True
