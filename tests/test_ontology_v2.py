"""Tests for the OWL 2 DL ontology.

These are contract tests on the artefact, not on any reasoner: they assert
that the modelling commitments described in the ontology header actually
hold in the file, so a future edit cannot silently drop them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

rdflib = pytest.importorskip("rdflib")

from rdflib import OWL, RDF, RDFS, Graph, URIRef  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = REPO_ROOT / "data" / "ontology" / "mtg-ontology-v2.0.ttl"

MTG = "http://purl.org/mtg/ontology#"
MTGD = "http://purl.org/mtg/ontology/decision#"
DUL = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"
PROV = "http://www.w3.org/ns/prov#"


@pytest.fixture(scope="module")
def graph() -> Graph:
    g = Graph()
    g.parse(str(ONTOLOGY), format="turtle")
    return g


def mtg_(local: str) -> URIRef:
    return URIRef(MTG + local)


def dul_(local: str) -> URIRef:
    return URIRef(DUL + local)


def test_ontology_parses(graph):
    assert len(graph) > 500


@pytest.mark.parametrize(
    "term,parent",
    [
        ("CardDesign", "InformationObject"),
        ("CardPrinting", "InformationRealization"),
        ("GameObject", "Object"),
        ("Player", "Agent"),
        ("Zone", "Place"),
        ("Turn", "Event"),
        ("GameAction", "Action"),
        ("GameSituation", "Situation"),
        ("GameQuality", "Quality"),
        ("ValueRegion", "Region"),
        ("GameRole", "Role"),
        ("Combo", "Plan"),
        ("Strategy", "Plan"),
        ("Archetype", "Description"),
        ("Keyword", "Concept"),
        ("CardType", "Concept"),
        ("Deck", "Collection"),
    ],
)
def test_dolce_alignment(graph, term, parent):
    assert (mtg_(term), RDFS.subClassOf, dul_(parent)) in graph


def test_no_punning_between_classes_and_individuals(graph):
    """v1.x used CardType and Zone as both classes and individuals."""
    classes = set(graph.subjects(RDF.type, OWL.Class))
    structural = {
        OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty,
        OWL.Ontology, OWL.Restriction, OWL.AllDifferent, OWL.AllDisjointClasses,
        OWL.NamedIndividual, OWL.FunctionalProperty, OWL.InverseFunctionalProperty,
        OWL.TransitiveProperty, OWL.SymmetricProperty, OWL.AsymmetricProperty,
        OWL.IrreflexiveProperty,
    }
    for s, _, o in graph.triples((None, RDF.type, None)):
        if s in classes and isinstance(o, URIRef) and o not in structural:
            pytest.fail(f"{s} is punned as both owl:Class and instance of {o}")


def test_zone_membership_is_functional(graph):
    """CR 400.5 — an object exists in exactly one zone."""
    assert (mtg_("inZone"), RDF.type, OWL.FunctionalProperty) in graph


def test_turn_ordering_is_a_strict_partial_order(graph):
    precedes = mtg_("precedes")
    assert (precedes, RDF.type, OWL.TransitiveProperty) in graph
    assert (precedes, RDF.type, OWL.AsymmetricProperty) in graph
    assert (precedes, RDF.type, OWL.IrreflexiveProperty) in graph


def test_counters_is_asymmetric_but_synergy_is_symmetric(graph):
    assert (mtg_("counters"), RDF.type, OWL.AsymmetricProperty) in graph
    assert (mtg_("synergisesWith"), RDF.type, OWL.SymmetricProperty) in graph


def test_ability_partition_is_a_disjoint_union(graph):
    """CR 113.3 partitions abilities into exactly four kinds."""
    assert (mtg_("Ability"), OWL.disjointUnionOf, None) in graph


def test_keys_are_declared(graph):
    for cls in ("CardDesign", "Combo", "CardPrinting"):
        assert (mtg_(cls), OWL.hasKey, None) in graph, f"{cls} has no owl:hasKey"


@pytest.mark.parametrize(
    "prop", ["involvedIn", "deckEmploysStrategy", "answersCombo"]
)
def test_property_chains_are_declared(graph, prop):
    assert (mtg_(prop), OWL.propertyChainAxiom, None) in graph


def test_combo_requires_at_least_two_pieces(graph):
    cardinalities = [
        int(o) for o in graph.objects(None, OWL.minQualifiedCardinality)
    ]
    assert 2 in cardinalities


def test_token_cannot_realise_a_card_design(graph):
    assert 0 in [int(o) for o in graph.objects(None, OWL.maxQualifiedCardinality)]


def test_every_declared_term_carries_an_epistemic_status(graph):
    status = mtg_("epistemicStatus")
    missing = []
    for t in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty):
        for s in graph.subjects(RDF.type, t):
            if not isinstance(s, URIRef):
                continue
            if not (str(s).startswith(MTG) or str(s).startswith(MTGD)):
                continue
            if (s, status, None) not in graph:
                missing.append(str(s))
    assert not missing, f"terms without epistemicStatus: {sorted(missing)}"


def test_induced_assertions_require_provenance(graph):
    """No learned claim may exist without the run that produced it."""
    assert (mtg_("InducedAssertion"), RDFS.subClassOf, URIRef(PROV + "Entity")) in graph
    restrictions = list(graph.objects(mtg_("InducedAssertion"), RDFS.subClassOf))
    generated_by = URIRef(PROV + "wasGeneratedBy")
    assert any(
        (r, OWL.onProperty, generated_by) in graph for r in restrictions
    ), "InducedAssertion must be restricted on prov:wasGeneratedBy"


def test_decision_module_is_provenance_enabled(graph):
    decision = URIRef(MTGD + "Decision")
    assert (decision, RDFS.subClassOf, URIRef(PROV + "Activity")) in graph


def test_competency_questions_reference_declared_terms():
    yaml = pytest.importorskip("yaml")
    cq_path = REPO_ROOT / "data" / "competency_questions.yaml"
    spec = yaml.safe_load(cq_path.read_text(encoding="utf-8"))
    g = Graph()
    g.parse(str(ONTOLOGY), format="turtle")
    ns = spec["namespaces"]
    known = set(g.subjects())

    missing = []
    for q in spec["questions"]:
        for bucket in ("explicit", "implicit", "derived"):
            for term in (q.get("terms", {}) or {}).get(bucket, []) or []:
                prefix, local = term.split(":", 1)
                uri = URIRef(ns[prefix] + local)
                if str(uri).startswith((MTG, MTGD)) and uri not in known:
                    missing.append(f"{q['id']} -> {term}")
    assert not missing, f"CQs reference undeclared terms: {missing}"
