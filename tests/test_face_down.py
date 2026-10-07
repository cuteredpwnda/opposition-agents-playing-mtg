"""Tests for H4 — face-down mechanics (morph, megamorph, manifest)."""

import pytest
from src.engine.game_state import CardInstance, GameState, PlayerState, Zone
from src.engine.face_down import (
    FaceDownMode,
    is_face_down,
    face_down_mode,
    can_be_turned_face_up,
    manifest,
    _apply_face_down_stub,
    _restore_face_up_data,
)


def _state() -> GameState:
    players = [
        PlayerState(player_id="Alice", name="Alice", life_total=20),
        PlayerState(player_id="Bob", name="Bob", life_total=20),
    ]
    return GameState(players=players, cards=[])


def _morph_creature(instance_id: str, controller: str = "Alice") -> CardInstance:
    return CardInstance(
        instance_id=instance_id,
        card_data={
            "name": "Fathom Seer",
            "mana_cost": "{1}{U}{U}",
            "type_line": "Creature — Illusion",
            "oracle_text": "Morph {U}{U}\nWhen Fathom Seer is turned face up, draw two cards.",
            "power": "1",
            "toughness": "3",
            "colors": ["U"],
        },
        zone=Zone.BATTLEFIELD,
        owner_id=controller,
        controller_id=controller,
    )


def _non_creature_card(instance_id: str, controller: str = "Alice") -> CardInstance:
    return CardInstance(
        instance_id=instance_id,
        card_data={
            "name": "Lightning Bolt",
            "mana_cost": "{R}",
            "type_line": "Instant",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "colors": ["R"],
        },
        zone=Zone.BATTLEFIELD,
        owner_id=controller,
        controller_id=controller,
    )


# ---------------------------------------------------------------------------
# _apply_face_down_stub
# ---------------------------------------------------------------------------

def test_face_down_stub_makes_22_colorless():
    card = _morph_creature("c1")
    _apply_face_down_stub(card, FaceDownMode.MORPHED)

    assert card.card_data["power"] == "2"
    assert card.card_data["toughness"] == "2"
    assert card.card_data.get("oracle_text") == ""
    assert card.card_data.get("colors") == []
    assert card.card_data.get("name") == ""


def test_face_down_preserves_original_data():
    card = _morph_creature("c1")
    original_name = card.card_data["name"]
    original_oracle = card.card_data["oracle_text"]
    _apply_face_down_stub(card, FaceDownMode.MORPHED)

    assert hasattr(card, "_face_up_data")
    assert card._face_up_data["name"] == original_name
    assert card._face_up_data["oracle_text"] == original_oracle


# ---------------------------------------------------------------------------
# _restore_face_up_data
# ---------------------------------------------------------------------------

def test_restore_face_up_data():
    card = _morph_creature("c1")
    _apply_face_down_stub(card, FaceDownMode.MORPHED)
    _restore_face_up_data(card)

    assert card.card_data["name"] == "Fathom Seer"
    assert card.card_data["power"] == "1"
    assert card.card_data["toughness"] == "3"
    assert not card.face_down
    assert not hasattr(card, "_face_up_data")


# ---------------------------------------------------------------------------
# is_face_down / face_down_mode
# ---------------------------------------------------------------------------

def test_is_face_down():
    card = _morph_creature("c1")
    assert not is_face_down(card)
    _apply_face_down_stub(card, FaceDownMode.MORPHED)
    assert is_face_down(card)


def test_face_down_mode():
    card = _morph_creature("c1")
    _apply_face_down_stub(card, FaceDownMode.MEGAMORPHED)
    assert face_down_mode(card) == FaceDownMode.MEGAMORPHED


# ---------------------------------------------------------------------------
# can_be_turned_face_up
# ---------------------------------------------------------------------------

def test_can_be_turned_face_up_morph():
    card = _morph_creature("c1")
    _apply_face_down_stub(card, FaceDownMode.MORPHED)
    assert can_be_turned_face_up(card) is True


def test_can_be_turned_face_up_manifest_creature():
    card = _morph_creature("c1")  # is a creature
    _apply_face_down_stub(card, FaceDownMode.MANIFESTED)
    assert can_be_turned_face_up(card) is True


def test_cannot_turn_face_up_manifest_non_creature():
    card = _non_creature_card("nc1")
    _apply_face_down_stub(card, FaceDownMode.MANIFESTED)
    assert can_be_turned_face_up(card) is False


def test_cannot_turn_face_up_without_mode():
    card = _morph_creature("c1")
    # Not face down at all
    assert can_be_turned_face_up(card) is False


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_manifest_makes_face_down_22():
    state = _state()
    card = _morph_creature("c1")
    card.zone = Zone.BATTLEFIELD
    state.cards.append(card)

    manifest(state, "Alice", card)

    assert is_face_down(card)
    assert face_down_mode(card) == FaceDownMode.MANIFESTED
    assert card.card_data["power"] == "2"
    assert card.card_data["toughness"] == "2"
    assert card.summoning_sick is True
