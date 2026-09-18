"""Tests for the type-line parser and the ontology-shaped ABox projection.

The parser is the component that turns "Scryfall gave us a string" into
"we know which family each subtype belongs to", so these tests carry the
cases that motivated the type-system rebuild: Vehicle is an artifact
subtype, Mount a creature subtype, Planet a land subtype, and none of them
is a card type.

All tests run offline against the generated TTL; none needs Neo4j.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge.type_line import (
    CARD_TYPE_TO_FAMILY,
    load_type_system,
    parse_type_line,
    report_over_cards,
    split_faces,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVED = REPO_ROOT / "data" / "ontology" / "mtg-cr-types.ttl"

pytestmark = pytest.mark.skipif(
    not DERIVED.exists(),
    reason="run scripts/build_ontology_from_cr.py to generate the type system",
)


@pytest.fixture(scope="module")
def ts():
    return load_type_system()


# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------


def test_type_system_sizes(ts):
    assert len(ts.card_types) == 15
    assert len(ts.supertypes) == 5
    assert len(ts.all_subtypes) > 500


def test_supertypes_are_exactly_the_five(ts):
    assert ts.supertypes == {"Basic", "Legendary", "Ongoing", "Snow", "World"}


@pytest.mark.parametrize(
    "name,family",
    [
        ("Vehicle", "ArtifactType"),
        ("Spacecraft", "ArtifactType"),
        ("Planet", "LandType"),
        ("Mount", "CreatureType"),
        ("Time Lord", "CreatureType"),
        ("Aura", "EnchantmentType"),
        ("Siege", "BattleType"),
    ],
)
def test_subtype_resolves_to_family(ts, name, family):
    assert family in ts.family_of(name)


def test_subtypes_are_not_card_types(ts):
    for wrong in ("Vehicle", "Mount", "Planet", "Spacecraft", "Equipment", "Aura"):
        assert wrong not in ts.card_types


def test_spacecraft_belongs_to_two_families(ts):
    """CR reuses the word; the parser must not pick one arbitrarily."""
    assert ts.family_of("Spacecraft") == {"ArtifactType", "PlanarType"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_simple_type_line(ts):
    p = parse_type_line("Instant", ts)
    assert p.card_types == ["Instant"]
    assert p.subtypes == []
    assert p.is_well_formed


def test_vehicle_is_parsed_as_an_artifact_subtype(ts):
    p = parse_type_line("Artifact — Vehicle", ts)
    assert p.card_types == ["Artifact"]
    assert p.subtypes == ["Vehicle"]
    assert p.subtype_families["Vehicle"] == {"ArtifactType"}
    assert p.is_well_formed


def test_multiple_supertypes(ts):
    p = parse_type_line("Basic Snow Land — Mountain", ts)
    assert p.supertypes == ["Basic", "Snow"]
    assert p.card_types == ["Land"]
    assert p.subtypes == ["Mountain"]


def test_dual_type_card_correlates_each_subtype(ts):
    """Dryad Arbor — the example CR 205.3c itself uses."""
    p = parse_type_line("Land Creature — Forest Dryad", ts)
    assert set(p.card_types) == {"Land", "Creature"}
    assert p.subtype_families["Forest"] == {"LandType"}
    assert p.subtype_families["Dryad"] == {"CreatureType"}


def test_two_word_creature_type_is_not_split(ts):
    p = parse_type_line("Legendary Creature — Time Lord Doctor", ts)
    assert p.subtypes == ["Time Lord", "Doctor"]


def test_kindred_licenses_creature_types(ts):
    p = parse_type_line("Kindred Enchantment — Merfolk", ts)
    assert set(p.card_types) == {"Kindred", "Enchantment"}
    assert p.subtype_families["Merfolk"] == {"CreatureType"}


def test_plane_subtype_is_the_whole_remainder(ts):
    """CR 205.3b: all words after the dash are one planar subtype."""
    p = parse_type_line("Plane — Bolas's Meditation Realm", ts)
    assert p.subtypes == ["Bolas's Meditation Realm"]


def test_spell_types_span_instant_and_sorcery(ts):
    for line in ("Instant — Arcane", "Sorcery — Arcane"):
        p = parse_type_line(line, ts)
        assert p.subtype_families["Arcane"] == {"SpellType"}


# ---------------------------------------------------------------------------
# CR 205.3d violation detection
# ---------------------------------------------------------------------------


def test_unlicensed_subtype_is_flagged(ts):
    """An enchantment cannot carry the artifact subtype Vehicle."""
    p = parse_type_line("Enchantment — Vehicle", ts)
    assert p.violations == ["Vehicle"]
    assert not p.is_well_formed


def test_licensed_subtype_is_not_flagged(ts):
    p = parse_type_line("Artifact Creature — Vehicle Golem", ts)
    assert p.violations == []


def test_nonsense_token_is_reported_not_silently_dropped(ts):
    p = parse_type_line("Creature — Truck", ts)
    assert "Truck" in p.unknown_tokens
    assert not p.is_well_formed


def test_every_card_type_with_subtypes_has_a_family_mapping(ts):
    """Guards against a new card type arriving with no family wired up."""
    typed_families = set(CARD_TYPE_TO_FAMILY.values())
    assert typed_families <= set(ts.subtypes.keys())


# ---------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------


def test_split_faces():
    assert split_faces("Instant // Sorcery") == ["Instant", "Sorcery"]
    assert split_faces("Creature — Human") == ["Creature — Human"]


def test_report_over_cards_counts_violations(ts):
    cards = [
        {"name": "Good", "type_line": "Artifact — Vehicle"},
        {"name": "Bad", "type_line": "Enchantment — Vehicle"},
        {"name": "Weird", "type_line": "Creature — Truck"},
    ]
    report = report_over_cards(cards, ts)
    assert report.total == 3
    assert report.well_formed == 1
    assert report.with_violations == 1
    assert report.with_unknown == 1
    assert "Truck" in report.unknown_counts


# ---------------------------------------------------------------------------
# ABox projection
# ---------------------------------------------------------------------------


def test_shape_card_emits_ontology_shape(ts):
    from src.knowledge.abox_builder import shape_card

    row = shape_card(
        {
            "name": "Smuggler's Copter",
            "id": "abc-123",
            "type_line": "Artifact — Vehicle",
            "oracle_text": "Flying",
            "mana_cost": "{2}",
            "cmc": 2,
            "power": "3",
            "toughness": "3",
            "set": "kld",
            "rarity": "rare",
            "keywords": ["Flying", "Crew"],
        },
        ts,
    )
    assert row["cardName"] == "Smuggler's Copter"
    assert row["cardTypes"] == ["Artifact"]
    assert {"name": "Vehicle", "family": "ArtifactType"} in row["subtypes"]
    assert row["printing"]["scryfallId"] == "abc-123"
    assert row["design"]["epistemicStatus"] == "curated"
    kinds = {q["kind"]: q for q in row["qualities"]}
    assert kinds["Power"]["numeric"] == 3
    assert row["violations"] == []


def test_characteristic_defining_power_keeps_lexical_drops_numeric(ts):
    """The case that forced the quality/region split."""
    from src.knowledge.abox_builder import shape_card

    row = shape_card(
        {"name": "Tarmogoyf", "type_line": "Creature — Lhurgoyf",
         "power": "*", "toughness": "1+*", "cmc": 2},
        ts,
    )
    kinds = {q["kind"]: q for q in row["qualities"]}
    assert kinds["Power"]["lexical"] == "*"
    assert kinds["Power"]["numeric"] is None
    assert kinds["Toughness"]["lexical"] == "1+*"
    assert kinds["Toughness"]["numeric"] is None


def test_shape_card_reports_violations(ts):
    from src.knowledge.abox_builder import shape_card

    row = shape_card(
        {"name": "Impossible", "type_line": "Enchantment — Vehicle", "cmc": 1}, ts
    )
    assert len(row["violations"]) == 1
    assert "Vehicle" in row["violations"][0]


def test_shape_card_handles_double_faced(ts):
    from src.knowledge.abox_builder import shape_card

    row = shape_card(
        {"name": "Split", "type_line": "Instant // Sorcery", "cmc": 1}, ts
    )
    assert set(row["cardTypes"]) == {"Instant", "Sorcery"}


def test_shape_card_without_id_has_no_printing(ts):
    from src.knowledge.abox_builder import shape_card

    row = shape_card({"name": "NoId", "type_line": "Instant", "cmc": 1}, ts)
    assert row["printing"] is None
