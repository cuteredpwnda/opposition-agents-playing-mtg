"""Tests for Commander Bracket classification + legality checks."""

from __future__ import annotations

import pytest

from src.integrations.commander_brackets import (
    BracketReport,
    can_play_together,
    check_legality,
    classify_bracket,
)
from src.integrations.decklist_loader import Decklist
from src.integrations.offline_card_db import OfflineCardDB


# ---------------------------------------------------------------------------
# Fake DB — avoids loading the 165 MB Scryfall cache during tests
# ---------------------------------------------------------------------------


def _card(name, *, type_line="Creature", ci=("R",), oracle="", power=2, toughness=2):
    return {
        "name": name, "type_line": type_line,
        "color_identity": list(ci), "oracle_text": oracle,
        "mana_cost": "{1}{R}", "cmc": 2,
        "power": str(power), "toughness": str(toughness),
    }


def _basic(name, ci):
    return {"name": name, "type_line": f"Basic Land — {name}",
            "color_identity": [ci], "oracle_text": f"({{T}}: Add {{{ci}}})",
            "mana_cost": "", "cmc": 0}


def _build_db():
    db = OfflineCardDB()
    cards = {
        "Krenko, Mob Boss": _card(
            "Krenko, Mob Boss",
            type_line="Legendary Creature — Goblin Warrior",
            oracle="Tap: Create X 1/1 red Goblin tokens",
            power=3, toughness=3,
        ),
        "Goblin Guide": _card("Goblin Guide", type_line="Creature — Goblin Scout"),
        "Lightning Bolt": _card("Lightning Bolt", type_line="Instant", oracle="3 damage"),
        "Sol Ring": _card("Sol Ring", type_line="Artifact", ci=(), oracle="add 2 mana"),
        "Demonic Tutor": _card("Demonic Tutor", type_line="Sorcery", ci=("B",),
                                oracle="Search your library for a card"),
        "Mana Crypt": _card("Mana Crypt", type_line="Artifact", ci=()),
        "Rhystic Study": _card("Rhystic Study", type_line="Enchantment", ci=("U",)),
        "Armageddon": _card("Armageddon", type_line="Sorcery", ci=("W",),
                             oracle="Destroy all lands."),
        "Time Warp": _card("Time Warp", type_line="Sorcery", ci=("U",),
                            oracle="Take an extra turn."),
        "Counterspell": _card("Counterspell", type_line="Instant", ci=("U",)),
        "Mountain": _basic("Mountain", "R"),
        "Island": _basic("Island", "U"),
        "Plains": _basic("Plains", "W"),
    }
    for n, c in cards.items():
        db.by_name[n.lower()] = c
    return db


def _mono_red_skeleton(commander="Krenko, Mob Boss") -> Decklist:
    """Build a 100-card mono-red deck with Krenko as commander."""
    d = Decklist()
    d.commander = [commander]
    # 1 commander + 99 others. Use Mountain x60 + Goblin Guide x4
    # + Lightning Bolt x4 + 31 Sol Ring proxies? No, singleton — must vary.
    # Easiest: 60 Mountain + 39 unique singleton cards.
    d.mainboard["Mountain"] = 60
    # Need 39 unique non-basic cards. We'll use Goblin Guide x1 + 38 fakes.
    # But fake cards aren't in the test DB → unknown-card warning, no violation.
    # So we'll just add one of each known red card and pad with Mountain.
    d.mainboard["Goblin Guide"] = 1
    d.mainboard["Lightning Bolt"] = 1
    d.mainboard["Sol Ring"] = 1
    # That's 60+1+1+1+1(commander) = 64. Pad.
    d.mainboard["Mountain"] += 36   # → 96 Mountains. Total = 96+1+1+1+1 = 100.
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_mono_red_is_bracket_1_or_2():
    db = _build_db()
    deck = _mono_red_skeleton()
    rep = classify_bracket(deck, db)
    assert rep.bracket in (1, 2)
    assert rep.is_legal, f"unexpected violations: {rep.violations}"
    assert not rep.game_changers
    assert not rep.mld


def test_game_changers_push_to_bracket_3_or_4():
    db = _build_db()
    deck = _mono_red_skeleton()
    # Add 2 Game Changers (still mono-red so no CI violation; Mana Crypt is colorless)
    deck.mainboard["Mana Crypt"] = 1
    deck.mainboard["Mountain"] -= 1
    rep = classify_bracket(deck, db)
    assert rep.bracket >= 3
    assert "Mana Crypt" in rep.game_changers


def test_mld_pushes_above_bracket_3():
    db = _build_db()
    deck = _mono_red_skeleton()
    # Swap commander to a multi-color so Armageddon (W) is legal
    deck.commander = ["Krenko, Mob Boss"]
    # Override CI for the test: pretend Krenko is 5-color so Armageddon is legal
    db.by_name["krenko, mob boss"]["color_identity"] = ["W", "U", "B", "R", "G"]
    deck.mainboard["Armageddon"] = 1
    deck.mainboard["Mountain"] -= 1
    rep = classify_bracket(deck, db)
    assert rep.bracket >= 4
    assert "Armageddon" in rep.mld


def test_singleton_violation_detected():
    db = _build_db()
    deck = _mono_red_skeleton()
    # Two copies of a non-basic
    deck.mainboard["Goblin Guide"] = 2
    deck.mainboard["Mountain"] -= 1
    violations = check_legality(deck, db)
    assert any("singleton" in v.lower() for v in violations)


def test_color_identity_violation_detected():
    db = _build_db()
    deck = _mono_red_skeleton()  # Krenko = mono-red
    deck.mainboard["Counterspell"] = 1     # blue card
    deck.mainboard["Mountain"] -= 1
    violations = check_legality(deck, db)
    assert any("color identity" in v.lower() for v in violations)


def test_deck_size_violation_detected():
    db = _build_db()
    deck = _mono_red_skeleton()
    deck.mainboard["Mountain"] -= 5
    violations = check_legality(deck, db)
    assert any("100" in v for v in violations)


def test_pod_compatibility():
    r1 = BracketReport(bracket=2, label="", score=0)
    r2 = BracketReport(bracket=3, label="", score=0)
    r3 = BracketReport(bracket=5, label="", score=0)
    assert can_play_together([r1, r2])
    assert not can_play_together([r1, r3])
    assert can_play_together([r1, r2], tolerance=2)


def test_cedh_hint_forces_bracket_5():
    db = _build_db()
    deck = _mono_red_skeleton()
    rep = classify_bracket(deck, db, cedh_hint=True)
    assert rep.bracket == 5
