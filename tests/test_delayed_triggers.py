"""Tests for H5 — delayed triggered abilities (CR 603.7)."""

import pytest
from src.engine.game_state import GameState, PlayerState
from src.engine.delayed_triggers import (
    DelayedTrigger,
    DelayedTriggerRegistry,
    TriggerPoint,
    get_delayed_registry,
    schedule,
    fire,
)


def _state() -> GameState:
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
        PlayerState(player_id="Bob", name="Bob", life_total=20),
    ]
    return GameState(players=players, cards=[])


# ---------------------------------------------------------------------------
# Registry lazy-init
# ---------------------------------------------------------------------------

def test_get_delayed_registry_lazy():
    state = _state()
    reg = get_delayed_registry(state)
    assert reg is not None
    assert get_delayed_registry(state) is reg


def test_schedule_and_fire():
    state = _state()
    calls: list[str] = []

    def _effect(s: GameState) -> None:
        calls.append("fired")

    schedule(state, TriggerPoint.END_OF_TURN, _effect, "Alice", "test effect")

    assert len(get_delayed_registry(state).pending()) == 1

    fire(state, TriggerPoint.END_OF_TURN)

    assert calls == ["fired"]
    # one-shot: should be gone after firing
    assert len(get_delayed_registry(state).pending()) == 0


def test_fire_only_matching_point():
    state = _state()
    calls: list[str] = []

    schedule(state, TriggerPoint.END_OF_COMBAT,
             lambda s: calls.append("eoc"), "Alice", "eoc trigger")
    schedule(state, TriggerPoint.END_OF_TURN,
             lambda s: calls.append("eot"), "Alice", "eot trigger")

    fire(state, TriggerPoint.END_OF_COMBAT)

    assert calls == ["eoc"]
    assert len(get_delayed_registry(state).pending()) == 1  # EOT trigger remains


def test_fire_with_condition_true():
    state = _state()
    results: list[int] = []

    schedule(
        state, TriggerPoint.NEXT_UPKEEP,
        lambda s: results.append(1),
        "Alice",
        condition=lambda s: True,
    )

    fire(state, TriggerPoint.NEXT_UPKEEP)
    assert results == [1]


def test_fire_with_condition_false():
    state = _state()
    results: list[int] = []

    schedule(
        state, TriggerPoint.NEXT_UPKEEP,
        lambda s: results.append(1),
        "Alice",
        condition=lambda s: False,
    )

    fire(state, TriggerPoint.NEXT_UPKEEP)
    # Condition was False, nothing fired
    assert results == []
    # only_this_turn=True (default): trigger should be gone
    assert len(get_delayed_registry(state).pending()) == 0


def test_persistent_trigger_not_removed_after_firing():
    state = _state()
    calls: list[int] = []

    schedule(
        state, TriggerPoint.END_OF_TURN,
        lambda s: calls.append(1),
        "Alice",
        once=False,
        only_this_turn=False,
    )

    fire(state, TriggerPoint.END_OF_TURN)
    fire(state, TriggerPoint.END_OF_TURN)

    assert calls == [1, 1]
    assert len(get_delayed_registry(state).pending()) == 1


def test_expire_end_of_turn_removes_only_this_turn():
    state = _state()

    schedule(state, TriggerPoint.END_OF_TURN,
             lambda s: None, "Alice", only_this_turn=True)
    schedule(state, TriggerPoint.NEXT_UPKEEP,
             lambda s: None, "Alice", only_this_turn=False)

    get_delayed_registry(state).expire_end_of_turn()

    remaining = get_delayed_registry(state).pending()
    assert len(remaining) == 1
    assert remaining[0].trigger_point == TriggerPoint.NEXT_UPKEEP


def test_cancel_for_card():
    state = _state()

    schedule(state, TriggerPoint.END_OF_TURN,
             lambda s: None, "Alice", description="token_exile abc123")
    schedule(state, TriggerPoint.END_OF_TURN,
             lambda s: None, "Alice", description="unrelated trigger")

    get_delayed_registry(state).cancel_for_card("abc123")

    remaining = get_delayed_registry(state).pending()
    assert len(remaining) == 1
    assert "unrelated" in remaining[0].description


def test_eot_trigger_fires_once_for_myriad():
    """Simulate myriad-style token exile — schedule fires at END_OF_COMBAT."""
    state = _state()
    exiled: list[str] = []

    token_ids = ["tok1", "tok2"]

    def _exile(s: GameState, ids: list[str] = token_ids) -> None:
        exiled.extend(ids)

    schedule(state, TriggerPoint.END_OF_COMBAT, _exile, "Alice",
             description="Myriad exile tokens")

    fire(state, TriggerPoint.END_OF_COMBAT)

    assert exiled == ["tok1", "tok2"]
    assert len(get_delayed_registry(state).pending()) == 0
