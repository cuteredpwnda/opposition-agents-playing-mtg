"""Tests for the Comprehensive-Rules-derived ontology.

These assert that the generated ABox actually contains the game content,
and that the card-type / subtype distinction is modelled correctly. They
are the regression net for "a new set added a type and nobody noticed":
regenerate from a fresh CR and these will move.

Counts are asserted as lower bounds, since Wizards adds types over time
but effectively never removes them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

rdflib = pytest.importorskip("rdflib")

from rdflib import RDF, RDFS, Graph, Literal, URIRef  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "data" / "ontology" / "mtg-ontology-v2.0.ttl"
TYPES = REPO_ROOT / "data" / "ontology" / "mtg-cr-types.ttl"
# Rule text is generated locally and git-ignored; tests that need it skip.
RULES = REPO_ROOT / "data" / "ontology" / "mtg-cr-rules.ttl"

MTG = "http://purl.org/mtg/ontology#"
SUBTYPE = "http://purl.org/mtg/ontology/subtype#"
KEYWORD = "http://purl.org/mtg/ontology/keyword#"
ACTION = "http://purl.org/mtg/ontology/keyword-action#"


@pytest.fixture(scope="module")
def graph() -> Graph:
    if not TYPES.exists():
        pytest.skip(
            "mtg-cr-types.ttl missing — run scripts/build_ontology_from_cr.py"
        )
    g = Graph()
    g.parse(str(SCHEMA), format="turtle")
    g.parse(str(TYPES), format="turtle")
    if RULES.exists():
        g.parse(str(RULES), format="turtle")
    return g


def labels_of_type(g: Graph, local: str) -> set[str]:
    cls = URIRef(MTG + local)
    return {
        str(lbl)
        for s in g.subjects(RDF.type, cls)
        for lbl in g.objects(s, RDFS.label)
    }


# --------------------------------------------------------------------------
# Card types — the complete CR 205.2a list
# --------------------------------------------------------------------------

CARD_TYPES = {
    "Artifact", "Battle", "Conspiracy", "Creature", "Dungeon", "Enchantment",
    "Instant", "Kindred", "Land", "Phenomenon", "Plane", "Planeswalker",
    "Scheme", "Sorcery", "Vanguard",
}


def test_all_fifteen_card_types_present(graph):
    found = {s.lower() for s in labels_of_type(graph, "CardType")}
    missing = {t for t in CARD_TYPES if t.lower() not in found}
    assert not missing, f"missing card types: {sorted(missing)}"


def test_no_subtype_is_declared_as_a_card_type(graph):
    """The error this whole section exists to prevent."""
    card_types = {s.lower() for s in labels_of_type(graph, "CardType")}
    for wrong in ("vehicle", "mount", "planet", "spacecraft", "equipment", "aura"):
        assert wrong not in card_types, f"{wrong!r} is a subtype, not a card type"


# --------------------------------------------------------------------------
# Subtypes — the cases that prompted this work
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,family",
    [
        ("Vehicle", "ArtifactType"),
        ("Spacecraft", "ArtifactType"),
        ("Equipment", "ArtifactType"),
        ("Fortification", "ArtifactType"),
        ("Lander", "ArtifactType"),
        ("Planet", "LandType"),
        ("Forest", "LandType"),
        ("Cave", "LandType"),
        ("Mount", "CreatureType"),
        ("Time Lord", "CreatureType"),
        ("Aura", "EnchantmentType"),
        ("Saga", "EnchantmentType"),
        ("Room", "EnchantmentType"),
        ("Adventure", "SpellType"),
        ("Siege", "BattleType"),
        ("Undercity", "DungeonType"),
    ],
)
def test_subtype_is_in_the_right_family(graph, name, family):
    assert name in labels_of_type(graph, family), f"{name} missing from {family}"


def test_basic_land_types(graph):
    basics = labels_of_type(graph, "BasicLandType")
    assert basics == {"Forest", "Island", "Mountain", "Plains", "Swamp"}


def test_basic_land_types_are_also_land_types(graph):
    assert labels_of_type(graph, "BasicLandType") <= labels_of_type(graph, "LandType")


@pytest.mark.parametrize(
    "family,minimum",
    [
        ("ArtifactType", 22),
        ("EnchantmentType", 13),
        ("LandType", 17),
        ("PlaneswalkerType", 80),
        ("SpellType", 5),
        ("CreatureType", 320),
        ("PlanarType", 80),
    ],
)
def test_subtype_family_sizes(graph, family, minimum):
    """Lower bounds — Wizards adds subtypes but does not remove them."""
    assert len(labels_of_type(graph, family)) >= minimum


def test_cross_family_reuse_is_represented_not_lost(graph):
    """Spacecraft is both an artifact type and a planar type (CR 205.3g/n)."""
    spacecraft = URIRef(SUBTYPE + "Spacecraft")
    types = set(graph.objects(spacecraft, RDF.type))
    assert URIRef(MTG + "ArtifactType") in types
    assert URIRef(MTG + "PlanarType") in types


# --------------------------------------------------------------------------
# Keywords and keyword actions
# --------------------------------------------------------------------------


def test_keyword_abilities_extracted(graph):
    keywords = labels_of_type(graph, "Keyword")
    assert len(keywords) >= 190
    for expected in ("Flying", "Trample", "Deathtouch", "Ward", "Crew", "Saddle"):
        assert expected in keywords


def test_keyword_actions_extracted(graph):
    actions = labels_of_type(graph, "KeywordAction")
    assert len(actions) >= 65
    for expected in ("Scry", "Mill", "Exile", "Attach", "Discover"):
        assert expected in actions


def test_keyword_action_exile_does_not_collide_with_the_exile_zone(graph):
    """The collision that forced per-family namespaces."""
    zone = URIRef(MTG + "Exile")
    action = URIRef(ACTION + "Exile")
    assert zone != action
    assert (zone, RDF.type, URIRef(MTG + "Zone")) in graph
    assert (action, RDF.type, URIRef(MTG + "KeywordAction")) in graph
    assert (zone, RDF.type, URIRef(MTG + "KeywordAction")) not in graph


def test_keyword_counters_are_keywords(graph):
    counters = labels_of_type(graph, "KeywordCounter")
    assert "Flying" in counters and "Deathtouch" in counters
    # CR 122.1b: a keyword counter grants that keyword, so it is the same
    # individual as the CR 702 ability.
    flying = URIRef(KEYWORD + "Flying")
    types = set(graph.objects(flying, RDF.type))
    assert URIRef(MTG + "Keyword") in types
    assert URIRef(MTG + "KeywordCounter") in types


# --------------------------------------------------------------------------
# Rule tree
# --------------------------------------------------------------------------


def test_rule_tree_is_populated(graph):
    if not RULES.exists():
        pytest.skip("rule-text layer is generated locally; not present")
    rules = list(graph.subjects(RDF.type, URIRef(MTG + "ComprehensiveRule")))
    assert len(rules) >= 3000


def test_a_known_rule_has_its_text(graph):
    if not RULES.exists():
        pytest.skip("rule-text layer is generated locally; not present")
    rule = URIRef(MTG + "CR205_2a")
    text = next(graph.objects(rule, URIRef(MTG + "ruleText")), None)
    assert text is not None
    assert "card types are" in str(text)


def test_sub_rule_links_exist(graph):
    if not RULES.exists():
        pytest.skip("rule-text layer is generated locally; not present")
    sub = URIRef(MTG + "subRuleOf")
    assert (URIRef(MTG + "CR205_3m"), sub, URIRef(MTG + "CR205_3")) in graph


def test_every_generated_individual_is_grounded_in_a_rule(graph):
    grounded = URIRef(MTG + "groundedIn")
    ungrounded = []
    for family in ("ArtifactType", "LandType", "CreatureType", "Keyword", "KeywordAction"):
        for s in graph.subjects(RDF.type, URIRef(MTG + family)):
            if (s, grounded, None) not in graph:
                ungrounded.append(str(s))
    assert not ungrounded, f"ungrounded individuals: {sorted(ungrounded)[:10]}"


def test_generated_content_is_marked_curated(graph):
    status = URIRef(MTG + "epistemicStatus")
    for s in graph.subjects(RDF.type, URIRef(MTG + "ArtifactType")):
        assert Literal("curated") in set(graph.objects(s, status))


def test_provenance_points_at_the_rules_document(graph):
    doc = URIRef(MTG + "CRDocument")
    assert (doc, RDF.type, URIRef("http://www.w3.org/ns/prov#Entity")) in graph
    derived = URIRef("http://www.w3.org/ns/prov#wasDerivedFrom")
    assert any(True for _ in graph.subjects(derived, doc))
