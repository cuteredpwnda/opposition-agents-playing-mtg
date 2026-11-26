from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from rdflib import OWL, RDF, Graph, URIRef

from scripts.validate_ontology import stage_profile
from src.knowledge.ontology_reasoning import import_closure


def import_fixture(tmp_path):
    root = tmp_path / "root.ttl"
    vocabulary = tmp_path / "vocabulary.ttl"
    imported = tmp_path / "dependency.ttl"
    root.write_text(
        "<urn:root> a <http://www.w3.org/2002/07/owl#Ontology> ; "
        "<http://www.w3.org/2002/07/owl#imports> <urn:external> .",
    )
    vocabulary.write_text(
        "<urn:vocabulary> <http://www.w3.org/2002/07/owl#imports> <urn:root> .",
    )
    imported.write_text("<urn:Class> a <http://www.w3.org/2002/07/owl#Class> .")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"imports": [{
        "iri": "urn:external", "file": imported.name, "format": "turtle",
        "sha256": hashlib.sha256(imported.read_bytes()).hexdigest(),
    }]}))
    return root, vocabulary, imported, lock


def test_import_closure_resolves_supplied_ontology_without_network(tmp_path):
    root, vocabulary, _, lock = import_fixture(tmp_path)
    graph, hashes = import_closure([root, vocabulary], lock, tmp_path)
    assert (URIRef("urn:Class"), RDF.type, OWL.Class) in graph
    assert list(graph.triples((None, OWL.imports, None))) == []
    assert len(hashes) == 3


def test_changed_or_missing_imports_fail_explicitly(tmp_path):
    root, vocabulary, imported, lock = import_fixture(tmp_path)
    imported.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        import_closure([root, vocabulary], lock, tmp_path)
    lock.write_text('{"imports": []}')
    with pytest.raises(ValueError, match="Unpinned"):
        import_closure([root], lock, tmp_path)


def test_nested_unpinned_import_fails(tmp_path):
    root, _, imported, lock = import_fixture(tmp_path)
    imported.write_text(
        "<urn:external> <http://www.w3.org/2002/07/owl#imports> <urn:unpinned> .",
    )
    entries = json.loads(lock.read_text())
    entries["imports"][0]["sha256"] = hashlib.sha256(imported.read_bytes()).hexdigest()
    lock.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="Unpinned.*urn:unpinned"):
        import_closure([root], lock, tmp_path)


def test_strict_profile_gate_does_not_confuse_consistency_with_profile(monkeypatch):
    pytest.importorskip("owlready2")
    from scripts.validate_ontology import stage_reasoner

    monkeypatch.setattr("src.knowledge.ontology_reasoning.qualify_ontology", lambda *_: {
        "reasoner": {"consistent": True, "unsatisfiable_classes": []},
        "profile": {"in_profile": False, "violations": ["external profile violation"]},
        "closure_triples": 1,
    })
    assert stage_reasoner(Path("unused")).ok
    assert not stage_reasoner(Path("unused"), require_dl_profile=True).ok


@pytest.mark.parametrize("skip_flag", ["--no-imports", "--skip-reasoner"])
def test_strict_profile_gate_cannot_be_bypassed_by_skip_flag(monkeypatch, skip_flag):
    from scripts import validate_ontology

    monkeypatch.setattr("sys.argv", ["validate_ontology.py", "--require-dl-profile", skip_flag])
    with pytest.raises(SystemExit) as exc:
        validate_ontology.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("property_type", [OWL.AsymmetricProperty, OWL.IrreflexiveProperty])
def test_local_sanity_rejects_non_simple_ordering_axioms(property_type):
    graph = Graph()
    prop = URIRef("urn:precedes")
    graph.add((prop, RDF.type, OWL.ObjectProperty))
    graph.add((prop, RDF.type, OWL.TransitiveProperty))
    graph.add((prop, RDF.type, property_type))
    assert not stage_profile(graph).ok


def test_local_sanity_rejects_inverse_functional_datatype():
    graph = Graph()
    prop = URIRef("urn:id")
    graph.add((prop, RDF.type, OWL.DatatypeProperty))
    graph.add((prop, RDF.type, OWL.InverseFunctionalProperty))
    assert not stage_profile(graph).ok


@pytest.mark.parametrize("edges,conforms", [
    ([("a", "b"), ("b", "c")], True),
    ([("a", "b"), ("b", "a")], False),
    ([("a", "a")], False),
])
def test_temporal_shacl_checks_cycles(edges, conforms):
    pyshacl = pytest.importorskip("pyshacl")
    graph = Graph()
    for a, b in edges:
        graph.add((URIRef("urn:" + a), URIRef("http://purl.org/mtg/ontology#precedes"),
                   URIRef("urn:" + b)))
    shapes = Path(__file__).resolve().parents[1] / "data" / "ontology" / "mtg-shapes.ttl"
    assert pyshacl.validate(graph, shacl_graph=str(shapes))[0] is conforms


def test_reasoning_bridge_uses_file_serialization_not_owlready_file_uri(monkeypatch):
    pytest.importorskip("owlready2")
    from src.knowledge.ontology_reasoning import qualify_graph

    monkeypatch.setattr("src.knowledge.ontology_reasoning.java_executable", lambda _: "java")
    response = Mock(returncode=0, stderr="", stdout=(
        'PROFILE={"in_profile":true,"violations":[]}\n'
        'REASONER={"consistent":true,"unsatisfiable_classes":[]}\n'
    ))
    run = Mock(return_value=response)
    monkeypatch.setattr("src.knowledge.ontology_reasoning.subprocess.run", run)
    result = qualify_graph(Graph())
    assert result["reasoner"]["consistent"]
    args = run.call_args[0][0]
    assert args[-1].endswith("closure.rdf")
    assert not args[-1].startswith("file:")


@pytest.mark.parametrize("failure,exception,match", [
    (Mock(returncode=1, stderr="failed", stdout=""), RuntimeError, "failed"),
    (Mock(returncode=0, stderr="", stdout=""), RuntimeError, "Incomplete"),
    (subprocess.TimeoutExpired("java", 1, output=b"partial"), TimeoutError, "partial"),
])
def test_reasoning_failures_do_not_become_passes(monkeypatch, failure, exception, match):
    pytest.importorskip("owlready2")
    from src.knowledge.ontology_reasoning import qualify_graph

    monkeypatch.setattr("src.knowledge.ontology_reasoning.java_executable", lambda _: "java")
    run = (
        Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    )
    monkeypatch.setattr("src.knowledge.ontology_reasoning.subprocess.run", run)
    with pytest.raises(exception, match=match):
        qualify_graph(Graph())


@pytest.mark.parametrize("invalid", [False, True])
def test_real_hermit_detects_disjoint_abox_probe(invalid):
    pytest.importorskip("owlready2")
    from src.knowledge.ontology_reasoning import java_executable, qualify_graph

    try:
        java_executable()
    except FileNotFoundError:
        pytest.skip("Java is required for the real reasoner probes")
    root = Path(__file__).resolve().parents[1]
    graph = Graph().parse(root / "data" / "ontology" / "mtg-ontology-v2.0.ttl")
    graph.remove((None, OWL.imports, None))
    individual = URIRef("urn:mtg:probe:object")
    graph.add((individual, RDF.type, URIRef("http://purl.org/mtg/ontology#GameObject")))
    if invalid:
        graph.add((individual, RDF.type, URIRef("http://purl.org/mtg/ontology#GameEvent")))
    result = qualify_graph(graph, timeout=60)
    assert result["reasoner"]["consistent"] is not invalid


def test_fuseki_snapshot_preserves_graph_boundaries(tmp_path, monkeypatch):
    from rdflib import Dataset

    from scripts import serve_ontology

    root, vocabulary, _, lock = import_fixture(tmp_path)
    monkeypatch.setattr(serve_ontology, "SCHEMA", root)
    monkeypatch.setattr(serve_ontology, "VOCABULARY", vocabulary)
    monkeypatch.setattr(serve_ontology, "IMPORT_CACHE", tmp_path)
    monkeypatch.setattr(serve_ontology, "IMPORT_LOCK", lock)
    monkeypatch.setattr(
        serve_ontology, "import_closure", lambda sources: import_closure(sources, lock, tmp_path),
    )
    snapshot, manifest = serve_ontology.build_dataset(tmp_path)
    dataset = Dataset().parse(snapshot, format="trig")
    assert manifest["named_graph_triples"]["http://purl.org/mtg/graph/schema"] == 2
    assert len(dataset.graph(URIRef("http://purl.org/mtg/graph/vocabulary"))) == 1
    assert len(dataset.graph(URIRef("urn:external"))) == 1
    assert len(dataset.default_graph) == 3
    assert (URIRef("urn:Class"), RDF.type, OWL.Class) not in dataset.default_graph
    assert manifest["snapshot_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    config = serve_ontology.write_config(tmp_path, snapshot).read_text()
    assert "fuseki:query" in config
    assert "fuseki:gsp_r" in config
    assert "fuseki:update" not in config
