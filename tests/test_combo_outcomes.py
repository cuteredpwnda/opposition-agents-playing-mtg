"""Tests for combo outcome categorisation + display naming."""
from __future__ import annotations

from src.knowledge.combo_outcomes import (
    Outcome,
    classify_feature,
    classify_features,
    combo_display_name,
)


def test_infinite_mana_classified_as_mana_infinite() -> None:
    oc = classify_feature("Infinite colorless mana")
    assert oc.category == "mana"
    assert oc.magnitude == "infinite"
    assert oc.outcome_id == "mana:infinite"
    assert "Mana" in oc.display_name


def test_near_infinite_damage_bucketed_as_damage_near_infinite() -> None:
    oc = classify_feature("Near-infinite damage")
    assert oc.category == "damage"
    assert oc.magnitude == "near_infinite"


def test_arbitrary_storm_classified_with_magnitude() -> None:
    oc = classify_feature("Arbitrary storm count")
    assert oc.category == "storm"
    assert oc.magnitude == "arbitrary"


def test_win_the_game_takes_priority_over_generic_buckets() -> None:
    oc = classify_feature("Win the game")
    assert oc.category == "win_game"


def test_unknown_feature_falls_back_to_other_finite() -> None:
    oc = classify_feature("Sproingboingify everything")
    assert oc == Outcome(feature="Sproingboingify everything",
                         category="other", magnitude="finite")


def test_classify_features_dedupes_on_category_magnitude() -> None:
    out = classify_features([
        "Infinite colorless mana",
        "Infinite blue mana",  # same bucket as above
        "Infinite damage",
        "",
    ])
    assert {o.outcome_id for o in out} == {"mana:infinite", "damage:infinite"}


def test_combo_display_name_joins_all_when_short() -> None:
    name = combo_display_name(["Heliod, Sun-Crowned", "Walking Ballista"])
    assert name == "Heliod, Sun-Crowned + Walking Ballista"


def test_combo_display_name_truncates_long_lists() -> None:
    name = combo_display_name(["A", "B", "C", "D", "E"])
    assert name == "A + B + C + 2 more"


def test_combo_display_name_handles_empty() -> None:
    assert combo_display_name([]) == "(unnamed combo)"
