"""Tests for H1 — continuous effects layer system (CR 613)."""

import pytest
from src.engine.game_state import CardInstance, GameState, PlayerState, Zone
from src.engine.continuous_effects import (
    ContinuousEffect,
    Layer,
    get_registry,
    next_timestamp,
    make_anthem_effect,
    make_keyword_grant_effect,
    make_eot_pump_effect,
    effective_power,
    effective_toughness,
    expire_end_of_turn,
    expire_for_card,
)


def _state_with_players() -> GameState:
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
        PlayerState(player_id="Bob", name="Bob", life_total=20),
    ]
    return GameState(players=players, cards=[])


def _creature(instance_id: str, name: str, power: str, toughness: str,
              controller: str = "Alice") -> CardInstance:
    return CardInstance(
        instance_id=instance_id,
        card_data={
            "name": name,
            "type_line": "Creature — Human",
            "oracle_text": "",
            "power": power,
            "toughness": toughness,
        },
        zone=Zone.BATTLEFIELD,
        owner_id=controller,
        controller_id=controller,
    )


# ---------------------------------------------------------------------------
# Basic registry operations
# ---------------------------------------------------------------------------

def test_get_registry_lazy_init():
    state = _state_with_players()
    reg = get_registry(state)
    assert reg is not None
    assert get_registry(state) is reg  # same instance


def test_next_timestamp_monotonic():
    state = _state_with_players()
    t1 = next_timestamp(state)
    t2 = next_timestamp(state)
    t3 = next_timestamp(state)
    assert t1 < t2 < t3


# ---------------------------------------------------------------------------
# Anthem effect
# ---------------------------------------------------------------------------

def test_anthem_stacks_with_two_lords():
    state = _state_with_players()
    creature = _creature("c1", "Soldier Token", "1", "1")
    state.cards.append(creature)

    # Two lord effects each granting +1/+1
    ts = next_timestamp(state)
    eff1 = make_anthem_effect(
        source_id="lord1", controller_id="Alice", timestamp=ts,
        power_bonus=1, toughness_bonus=1,
        filter_fn=lambda c, _s: True,
    )
    ts2 = next_timestamp(state)
    eff2 = make_anthem_effect(
        source_id="lord2", controller_id="Alice", timestamp=ts2,
        power_bonus=1, toughness_bonus=1,
        filter_fn=lambda c, _s: True,
    )
    get_registry(state).register(eff1)
    get_registry(state).register(eff2)

    assert effective_power(creature, state) == 3   # 1 base + 1 + 1
    assert effective_toughness(creature, state) == 3


def test_anthem_filter_by_type():
    """An anthem only boosting Soldiers must not affect non-Soldiers."""
    state = _state_with_players()
    soldier = _creature("sol1", "Soldier", "1", "1")
    soldier.card_data["subtypes"] = ["Soldier"]
    wizard = _creature("wiz1", "Wizard", "1", "1")
    wizard.card_data["subtypes"] = ["Wizard"]
    state.cards.extend([soldier, wizard])

    def is_soldier(c: CardInstance, _s: GameState) -> bool:
        return "Soldier" in (c.card_data.get("subtypes") or [])

    eff = make_anthem_effect(
        source_id="sol_lord", controller_id="Alice", timestamp=next_timestamp(state),
        power_bonus=2, toughness_bonus=0,
        filter_fn=is_soldier,
    )
    get_registry(state).register(eff)

    assert effective_power(soldier, state) == 3
    assert effective_power(wizard, state) == 1


# ---------------------------------------------------------------------------
# Keyword grant
# ---------------------------------------------------------------------------

def test_keyword_grant():
    state = _state_with_players()
    creature = _creature("c1", "Bear", "2", "2")
    state.cards.append(creature)

    eff = make_keyword_grant_effect(
        source_id="haste_lord", controller_id="Alice", timestamp=next_timestamp(state),
        keyword="haste",
        filter_fn=lambda c, _s: True,
    )
    get_registry(state).register(eff)

    snap = get_registry(state).apply_to(creature, state)
    kws = snap.card_data.get("keywords") or []
    assert "haste" in kws


# ---------------------------------------------------------------------------
# EOT pump expires
# ---------------------------------------------------------------------------

def test_eot_pump_expires_at_cleanup():
    state = _state_with_players()
    creature = _creature("c1", "Bear", "2", "2")
    state.cards.append(creature)

    ts = next_timestamp(state)
    eff = make_eot_pump_effect(
        source_id="pump_source", controller_id="Alice", timestamp=ts,
        target_id="c1", power_bonus=2, toughness_bonus=2,
    )
    get_registry(state).register(eff)

    assert effective_power(creature, state) == 4
    assert effective_toughness(creature, state) == 4

    expire_end_of_turn(state)

    assert effective_power(creature, state) == 2
    assert effective_toughness(creature, state) == 2


# ---------------------------------------------------------------------------
# LTB removes effect
# ---------------------------------------------------------------------------

def test_ltb_removes_source_effect():
    state = _state_with_players()
    creature = _creature("c1", "Bear", "2", "2")
    state.cards.append(creature)

    eff = make_anthem_effect(
        source_id="anthem_source", controller_id="Alice", timestamp=next_timestamp(state),
        power_bonus=1, toughness_bonus=1,
        filter_fn=lambda c, _s: True,
    )
    get_registry(state).register(eff)

    assert effective_power(creature, state) == 3

    expire_for_card(state, "anthem_source")

    assert effective_power(creature, state) == 2
