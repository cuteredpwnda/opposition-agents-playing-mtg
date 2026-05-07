"""Tests for the deck-builder constraint engine (G1 / G10).

All tests are pure-Python — no network, no Neo4j, no card DB file.
"""
from __future__ import annotations

import pytest

from src.agents.deck_builder.constraints import (
    ConstraintResult,
    ConstraintSet,
    card_color_identity,
    commander_color_identity,
    is_basic_land,
    validate_full_deck,
)


# ---------------------------------------------------------------------------
# Minimal card-dict helpers
# ---------------------------------------------------------------------------

def _card(
    name: str,
    ci: list[str] | None = None,
    type_line: str = "Instant",
    legalities: dict | None = None,
    cmc: float = 2.0,
) -> dict:
    return {
        "name": name,
        "color_identity": ci or [],
        "type_line": type_line,
        "legalities": legalities or {"commander": "legal"},
        "cmc": cmc,
    }


def _basic(name: str) -> dict:
    return _card(name, ci=[], type_line="Basic Land — Forest")


# ---------------------------------------------------------------------------
# commander_color_identity / card_color_identity
# ---------------------------------------------------------------------------


def test_commander_ci_single():
    krenko = _card("Krenko, Mob Boss", ci=["R"])
    assert commander_color_identity([krenko]) == {"R"}


def test_commander_ci_partners():
    a = _card("A", ci=["W", "U"])
    b = _card("B", ci=["B"])
    assert commander_color_identity([a, b]) == {"W", "U", "B"}


def test_commander_ci_colorless():
    kozilek = _card("Kozilek, Butcher of Truth", ci=[])
    assert commander_color_identity([kozilek]) == set()


def test_card_color_identity_case():
    card = _card("Foo", ci=["w", "u"])
    assert card_color_identity(card) == {"W", "U"}


# ---------------------------------------------------------------------------
# is_basic_land
# ---------------------------------------------------------------------------


def test_basic_land_recognised():
    for name in ("Plains", "Island", "Swamp", "Mountain", "Forest",
                 "Wastes", "Snow-Covered Forest"):
        assert is_basic_land({"name": name})


def test_non_basic_not_basic():
    assert not is_basic_land(_card("Stomping Ground", ci=["R", "G"]))


# ---------------------------------------------------------------------------
# ConstraintSet.accept — colour identity
# ---------------------------------------------------------------------------


def test_accept_in_ci():
    cs = ConstraintSet(commander_ci={"R"})
    result = cs.accept(_card("Lightning Bolt", ci=["R"]))
    assert result.ok


def test_reject_off_ci():
    cs = ConstraintSet(commander_ci={"R"})
    result = cs.accept(_card("Counterspell", ci=["U"]))
    assert not result.ok
    assert "colour identity" in result.reason


def test_basic_land_always_accepted_regardless_of_ci():
    cs = ConstraintSet(commander_ci={"R"})
    # Forest is green but a basic — must always be legal.
    forest = {"name": "Forest", "color_identity": ["G"],
              "type_line": "Basic Land — Forest",
              "legalities": {"commander": "legal"}, "cmc": 0.0}
    assert cs.accept(forest).ok


def test_colorless_card_always_accepted():
    cs = ConstraintSet(commander_ci={"R"})
    sol_ring = _card("Sol Ring", ci=[])
    assert cs.accept(sol_ring).ok


# ---------------------------------------------------------------------------
# ConstraintSet.accept — singleton
# ---------------------------------------------------------------------------


def test_singleton_duplicate_rejected():
    cs = ConstraintSet(commander_ci={"R"})
    card = _card("Lightning Bolt", ci=["R"])
    cs.add(card)
    result = cs.accept(card)
    assert not result.ok
    assert "singleton" in result.reason


def test_allow_non_singleton_flag():
    cs = ConstraintSet(commander_ci={"R"}, allow_non_singleton=True)
    card = _card("Lightning Bolt", ci=["R"])
    cs.add(card)
    assert cs.accept(card).ok


# ---------------------------------------------------------------------------
# ConstraintSet.accept — legality
# ---------------------------------------------------------------------------


def test_banned_card_rejected():
    cs = ConstraintSet(commander_ci={"U", "W"})
    sol_ring_banned = _card(
        "Sol Ring", ci=[], legalities={"commander": "banned"}
    )
    result = cs.accept(sol_ring_banned)
    assert not result.ok
    assert "banned" in result.reason


def test_not_legal_card_rejected():
    cs = ConstraintSet(commander_ci={"R"})
    standard_only = _card(
        "SomeCard", ci=["R"], legalities={"commander": "not_legal"}
    )
    assert not cs.accept(standard_only).ok


def test_legal_card_passes():
    cs = ConstraintSet(commander_ci={"R"})
    assert cs.accept(_card("Shock", ci=["R"])).ok


def test_unknown_format_key_still_passes():
    cs = ConstraintSet(commander_ci={"R"}, format="commander")
    # Card has no "commander" key in legalities → empty string → not banned
    card = _card("Unknown", ci=["R"], legalities={})
    assert cs.accept(card).ok


# ---------------------------------------------------------------------------
# ConstraintSet state tracking
# ---------------------------------------------------------------------------


def test_needs_more_and_current_count():
    cs = ConstraintSet(commander_ci={"R"}, target_size=3)
    assert cs.needs_more()
    for i in range(3):
        card = _card(f"Card{i}", ci=["R"])
        cs.add(card)
    assert not cs.needs_more()
    assert cs.current_count() == 3


def test_current_names():
    cs = ConstraintSet(commander_ci={"G"})
    cs.add(_card("Llanowar Elves", ci=["G"]))
    cs.add(_card("Birds of Paradise", ci=["G"]))
    assert cs.current_names() == {"Llanowar Elves", "Birds of Paradise"}


def test_reset_clears_mainboard():
    cs = ConstraintSet(commander_ci={"R"})
    cs.add(_card("Shock", ci=["R"]))
    cs.reset()
    assert cs.current_count() == 0


# ---------------------------------------------------------------------------
# mana_curve
# ---------------------------------------------------------------------------


def test_mana_curve_basic():
    cards = [
        _card("A", cmc=1.0),
        _card("B", cmc=2.0),
        _card("C", cmc=4.0),
        _card("Land", ci=[], type_line="Basic Land — Forest", cmc=0.0),
    ]
    cs = ConstraintSet(commander_ci=set())
    curve = cs.mana_curve(cards)
    # Land excluded; A<=1, B<=2, C<=4.
    assert curve["<=1"] == 1
    assert curve["<=2"] == 1
    assert curve["<=4"] == 1


# ---------------------------------------------------------------------------
# validate_full_deck
# ---------------------------------------------------------------------------


def make_99_cards(ci: list[str], start: int = 0) -> list[dict]:
    return [_card(f"Card{i+start}", ci=ci) for i in range(99)]


def test_validate_clean_deck():
    commander = [_card("Krenko", ci=["R"], type_line="Legendary Creature")]
    mainboard = make_99_cards(["R"])
    errors = validate_full_deck(commander, mainboard)
    assert errors == []


def test_validate_wrong_size():
    commander = [_card("Krenko", ci=["R"])]
    mainboard = make_99_cards(["R"])[:50]
    errors = validate_full_deck(commander, mainboard)
    assert any("51 cards" in e for e in errors)


def test_validate_ci_violation():
    commander = [_card("Krenko", ci=["R"])]
    mainboard = make_99_cards(["R"])
    # Sneak in a blue card
    mainboard[5] = _card("Counterspell", ci=["U"])
    errors = validate_full_deck(commander, mainboard)
    assert any("Counterspell" in e for e in errors)
