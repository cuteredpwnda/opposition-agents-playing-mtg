"""Tests for H3 — Watcher event-observer system."""

import pytest
from src.engine.game_state import CardInstance, GameState, PlayerState, Zone
from src.engine.watchers import (
    GameEvent,
    GameEventKind,
    WatcherRegistry,
    SpellsCastThisTurnWatcher,
    LifeLostThisTurnWatcher,
    LifeGainedThisTurnWatcher,
    PlayerAttackedThisTurnWatcher,
    LandPlayedThisTurnWatcher,
    DamageThisTurnWatcher,
    get_watcher_registry,
    fire,
    on_spell_cast,
    on_life_lost,
    on_life_gained,
    on_creature_attacked,
    on_land_played,
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

def test_get_watcher_registry_lazy():
    state = _state()
    reg = get_watcher_registry(state)
    assert reg is not None
    assert get_watcher_registry(state) is reg


def test_default_watchers_registered():
    state = _state()
    reg = get_watcher_registry(state)
    assert reg.get(SpellsCastThisTurnWatcher) is not None
    assert reg.get(LifeLostThisTurnWatcher) is not None


# ---------------------------------------------------------------------------
# SpellsCastThisTurnWatcher
# ---------------------------------------------------------------------------

def test_spells_cast_watcher_counts():
    state = _state()
    on_spell_cast(state, "Alice", "spell1")
    on_spell_cast(state, "Alice", "spell2")
    on_spell_cast(state, "Bob", "spell3")

    reg = get_watcher_registry(state)
    w = reg.get(SpellsCastThisTurnWatcher)
    assert w.count_for("Alice") == 2
    assert w.count_for("Bob") == 1
    assert w.total() == 3


def test_spells_cast_watcher_resets():
    state = _state()
    on_spell_cast(state, "Alice", "s1")
    get_watcher_registry(state).reset_turn()

    w = get_watcher_registry(state).get(SpellsCastThisTurnWatcher)
    assert w.count_for("Alice") == 0
    assert w.total() == 0


# ---------------------------------------------------------------------------
# LifeLostThisTurnWatcher
# ---------------------------------------------------------------------------

def test_life_lost_watcher():
    state = _state()
    on_life_lost(state, "Alice", 3, "source1")
    on_life_lost(state, "Alice", 2, "source2")
    on_life_lost(state, "Bob", 5, "source1")

    w = get_watcher_registry(state).get(LifeLostThisTurnWatcher)
    assert w.lost_by("Alice") == 5
    assert w.lost_by("Bob") == 5


def test_life_lost_watcher_resets():
    state = _state()
    on_life_lost(state, "Alice", 10, "x")
    get_watcher_registry(state).reset_turn()
    w = get_watcher_registry(state).get(LifeLostThisTurnWatcher)
    assert w.lost_by("Alice") == 0


# ---------------------------------------------------------------------------
# LifeGainedThisTurnWatcher
# ---------------------------------------------------------------------------

def test_life_gained_watcher():
    state = _state()
    on_life_gained(state, "Bob", 4)
    on_life_gained(state, "Bob", 6)

    w = get_watcher_registry(state).get(LifeGainedThisTurnWatcher)
    assert w.gained_by("Bob") == 10


# ---------------------------------------------------------------------------
# PlayerAttackedThisTurnWatcher
# ---------------------------------------------------------------------------

def test_attacked_watcher():
    state = _state()
    on_creature_attacked(state, "Alice", "creature1")
    on_creature_attacked(state, "Alice", "creature2")

    w = get_watcher_registry(state).get(PlayerAttackedThisTurnWatcher)
    assert w.did_attack("Alice") is True
    assert w.did_attack("Bob") is False
    assert "creature1" in w.attacker_ids
    assert "creature2" in w.attacker_ids


# ---------------------------------------------------------------------------
# LandPlayedThisTurnWatcher
# ---------------------------------------------------------------------------

def test_land_played_watcher():
    state = _state()
    on_land_played(state, "Alice", "land1")

    w = get_watcher_registry(state).get(LandPlayedThisTurnWatcher)
    assert w.count_for("Alice") == 1
    assert w.count_for("Bob") == 0


# ---------------------------------------------------------------------------
# Event routing — only matching watchers receive events
# ---------------------------------------------------------------------------

def test_event_routing_spell_only_reaches_spell_watcher():
    state = _state()
    on_spell_cast(state, "Alice", "s1")

    land_w = get_watcher_registry(state).get(LandPlayedThisTurnWatcher)
    assert land_w.count_for("Alice") == 0
