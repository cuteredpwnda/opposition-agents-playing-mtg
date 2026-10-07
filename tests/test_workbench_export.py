from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from rdflib import RDF, BNode, Graph, Literal, URIRef
from rdflib.compare import isomorphic

from scripts.export_ontology_workbench import (
    MTG,
    MTGD,
    ROOT,
    export_workbench,
    rdf_json,
)

SCHEMA = ROOT / "data" / "ontology" / "mtg-ontology-v2.0.ttl"
VOCABULARY = ROOT / "data" / "ontology" / "mtg-cr-types.ttl"
CQS = ROOT / "data" / "competency_questions.yaml"


@pytest.fixture(scope="module")
def exported():
    return export_workbench(SCHEMA, VOCABULARY, CQS)


def test_pinned_workbench_contract(exported):
    contract = Path(__file__).parent / "fixtures" / "kg_workbench_import.schema.json"
    Draft202012Validator(json.loads(contract.read_text(encoding="utf-8"))).validate(exported[0])


def test_review_modules_and_references(exported):
    payload, report = exported
    classes = {c["id"]: c for c in payload["classes"]}
    modules = {m["id"] for m in payload["modules"]}
    assert classes[str(MTG.ComprehensiveRule)]["moduleId"] == "rules"
    assert classes[str(MTG.CardDesign)]["moduleId"] == "cards"
    assert classes[str(MTG.GameSituation)]["moduleId"] == "gameplay"
    assert classes[str(MTG.Combo)]["moduleId"] == "strategy"
    assert classes[str(MTGD.Decision)]["moduleId"] == "decisions"
    assert classes[str(MTG.InducedSynergy)]["moduleId"] == "evidence"
    for c in classes.values():
        assert c["moduleId"] in modules
        assert c["parentClassId"] is None or c["parentClassId"] in classes
    ids = {r["id"] for r in payload["relations"]}
    assert len(ids) == len(payload["relations"])
    for relation in payload["relations"]:
        assert relation["domainClassId"] in classes
        assert relation["rangeClassId"] in classes
    assert report["counts"]["schema_named_classes"] == 62
    assert len(payload["competencyQuestions"]) == 32
    for cq in payload["competencyQuestions"]:
        assert cq["subjectClassId"] is None or cq["subjectClassId"] in classes
        assert cq["predicateRelationId"] is None or cq["predicateRelationId"] in ids
        assert all(module["id"] in modules for module in cq["modules"])


def test_union_inverse_and_examples_are_explicit_projections(exported):
    payload, report = exported
    pieces = [r for r in payload["relations"] if r["name"] == "mtg:partOfCombo"]
    assert len(pieces) == 1
    assert pieces[0]["domainClassId"] == str(MTG.CardDesign)
    assert pieces[0]["rangeClassId"] == str(MTG.Combo)
    ability = [r for r in payload["relations"] if r["name"] == "mtg:hasAbility"]
    assert {r["domainClassId"] for r in ability} == {str(MTG.CardDesign), str(MTG.GameObject)}
    assert "not an OWL round-trip" in payload["usecase"]
    assert report["warnings"]
    assert any("Flying" in example["value"] for example in payload["examples"])
    assert all(not e["isInstanceCandidate"] for e in payload["examples"])
    assert not any(e["targetId"] == str(MTG.ComprehensiveRule)
                   for e in payload["examples"])


def test_export_is_reproducible_and_inputs_unchanged(exported):
    original = [path.read_bytes() for path in (SCHEMA, VOCABULARY, CQS)]
    assert export_workbench(SCHEMA, VOCABULARY, CQS) == exported
    assert original == [path.read_bytes() for path in (SCHEMA, VOCABULARY, CQS)]


def test_persistent_workbench_files_match_current_sources(exported):
    output = ROOT / "data" / "ontology" / "mtg-workbench.json"
    assert json.loads(output.read_text(encoding="utf-8")) == exported[0]
    assert json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8")) == exported[1]
    assert json.loads(output.with_suffix(".rdf.json").read_text(encoding="utf-8")) == rdf_json(
        SCHEMA, VOCABULARY,
    )


def test_companion_rdf_json_is_lossless():
    encoded = rdf_json(SCHEMA, VOCABULARY)
    decoded = Graph()

    def resource(value):
        return BNode(value[2:]) if value.startswith("_:") else URIRef(value)

    for subject, predicates in encoded.items():
        for predicate, values in predicates.items():
            for value in values:
                obj = (
                    Literal(value["value"], lang=value.get("lang"),
                            datatype=value.get("datatype"))
                    if value["type"] == "literal" else resource(value["value"])
                )
                decoded.add((resource(subject), URIRef(predicate), obj))
    source = Graph().parse(SCHEMA) + Graph().parse(VOCABULARY)
    assert len(decoded) == len(source)
    assert isomorphic(source, decoded)
    assert (MTG.Combo, RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class")) in decoded
    assert rdf_json(SCHEMA, VOCABULARY) == encoded
