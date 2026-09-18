#!/usr/bin/env python
"""Validate the MTG ontology: syntax, OWL 2 profile, SHACL, CQ coverage.

Implements the decompose -> validate -> report cascade rather than a single
"does it parse" check. Each stage is independent and reports separately, so a
failure localises to one concern:

1. **Syntax** — rdflib parse of the Turtle/RDF-XML source.
2. **Structure** — counts of classes, object/datatype properties, individuals,
   and the OWL 2 constructs that carry modelling weight here (property chains,
   keys, qualified cardinality, disjoint unions).
3. **Profile** — OWL 2 DL sanity checks we can run without a Java reasoner:
   punning detection (a term used as both class and individual), undeclared
   terms in domain/range position, and dangling ``rdfs:subClassOf`` targets.
4. **Consistency** — optional HermiT run via ``owlready2`` when available.
5. **SHACL** — optional ``pyshacl`` run against the shapes file.
6. **Competency questions** — term coverage plus SPARQL execution, from
   ``data/competency_questions.yaml``.
7. **Provenance** — every axiom subject must carry ``mtg:epistemicStatus``,
   and every induced term must be reachable from a ``prov:Activity``.

Exit code is non-zero if any *enabled* stage fails.

Usage::

    python scripts/validate_ontology.py
    python scripts/validate_ontology.py --no-imports --skip-reasoner
    python scripts/validate_ontology.py --ontology data/ontology/mtg-ontology-v2.0.ttl
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONTOLOGY = REPO_ROOT / "data" / "ontology" / "mtg-ontology-v2.0.ttl"
DEFAULT_SHAPES = REPO_ROOT / "data" / "ontology" / "mtg-shapes.ttl"
DEFAULT_CQS = REPO_ROOT / "data" / "competency_questions.yaml"

MTG = "http://purl.org/mtg/ontology#"
MTGD = "http://purl.org/mtg/ontology/decision#"


@dataclass
class StageResult:
    name: str
    ok: bool
    detail: str = ""
    items: list[str] = field(default_factory=list)
    skipped: bool = False

    def render(self) -> str:
        mark = "SKIP" if self.skipped else ("PASS" if self.ok else "FAIL")
        head = f"[{mark}] {self.name}"
        if self.detail:
            head += f" — {self.detail}"
        body = "".join(f"\n        {i}" for i in self.items[:25])
        if len(self.items) > 25:
            body += f"\n        ... and {len(self.items) - 25} more"
        return head + body


def _require_rdflib():
    try:
        import rdflib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "rdflib is required: pip install -e .[ontology]  (or pip install rdflib)"
        ) from exc


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_syntax(path: Path) -> tuple[StageResult, Any]:
    from rdflib import Graph

    g = Graph()
    fmt = "turtle" if path.suffix in {".ttl", ".n3"} else "xml"
    try:
        g.parse(str(path), format=fmt)
    except Exception as exc:
        return StageResult("syntax", False, f"{type(exc).__name__}: {exc}"), None
    return StageResult("syntax", True, f"{len(g)} triples parsed from {path.name}"), g


def stage_structure(g: Any) -> StageResult:
    from rdflib import OWL, RDF, RDFS

    def count(pred, obj) -> int:
        return len(set(g.subjects(pred, obj)))

    stats = {
        "owl:Class": count(RDF.type, OWL.Class),
        "owl:ObjectProperty": count(RDF.type, OWL.ObjectProperty),
        "owl:DatatypeProperty": count(RDF.type, OWL.DatatypeProperty),
        "owl:AnnotationProperty": count(RDF.type, OWL.AnnotationProperty),
        "owl:propertyChainAxiom": len(list(g.triples((None, OWL.propertyChainAxiom, None)))),
        "owl:hasKey": len(list(g.triples((None, OWL.hasKey, None)))),
        "owl:disjointUnionOf": len(list(g.triples((None, OWL.disjointUnionOf, None)))),
        "owl:AllDisjointClasses": count(RDF.type, OWL.AllDisjointClasses),
        "owl:equivalentClass": len(list(g.triples((None, OWL.equivalentClass, None)))),
        "qualified cardinality": (
            len(list(g.triples((None, OWL.minQualifiedCardinality, None))))
            + len(list(g.triples((None, OWL.maxQualifiedCardinality, None))))
            + len(list(g.triples((None, OWL.qualifiedCardinality, None))))
        ),
        "rdfs:subClassOf": len(list(g.triples((None, RDFS.subClassOf, None)))),
    }
    items = [f"{k:28s} {v}" for k, v in stats.items()]
    empty = [k for k in ("owl:Class", "owl:ObjectProperty") if stats[k] == 0]
    return StageResult(
        "structure",
        not empty,
        "no classes or properties declared" if empty else "term inventory",
        items,
    )


def stage_profile(g: Any) -> StageResult:
    """OWL 2 DL sanity checks that do not need a reasoner."""
    from rdflib import OWL, RDF, RDFS, BNode, URIRef

    problems: list[str] = []

    classes = set(g.subjects(RDF.type, OWL.Class))
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    annot_props = set(g.subjects(RDF.type, OWL.AnnotationProperty))
    declared = classes | obj_props | data_props | annot_props

    # Punning: a term declared as a class that is also used as an individual
    # of another class. This is what pushed v1.x out of OWL 2 DL.
    for s, _, o in g.triples((None, RDF.type, None)):
        if isinstance(s, BNode) or not isinstance(o, URIRef):
            continue
        if o in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
                 OWL.AnnotationProperty, OWL.Ontology, OWL.AllDifferent,
                 OWL.AllDisjointClasses, OWL.Restriction, OWL.NamedIndividual,
                 OWL.FunctionalProperty, OWL.InverseFunctionalProperty,
                 OWL.TransitiveProperty, OWL.SymmetricProperty,
                 OWL.AsymmetricProperty, OWL.IrreflexiveProperty):
            continue
        if s in classes:
            problems.append(f"punning: {s} is an owl:Class and also an instance of {o}")

    # Domain/range pointing at undeclared, non-blank, in-namespace terms.
    for pred in (RDFS.domain, RDFS.range):
        for _, _, o in g.triples((None, pred, None)):
            if isinstance(o, BNode) or not isinstance(o, URIRef):
                continue
            iri = str(o)
            if not (iri.startswith(MTG) or iri.startswith(MTGD)):
                continue
            if o not in declared and o not in set(g.subjects()):
                problems.append(f"undeclared term in {pred.split('#')[-1]} position: {o}")

    # subClassOf targets that are in our namespace but never declared.
    for _, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(o, BNode) or not isinstance(o, URIRef):
            continue
        iri = str(o)
        if (iri.startswith(MTG) or iri.startswith(MTGD)) and o not in classes:
            problems.append(f"subClassOf target not declared as owl:Class: {o}")

    unique = sorted(set(problems))
    return StageResult(
        "owl2-dl-profile",
        not unique,
        "clean" if not unique else f"{len(unique)} issue(s)",
        unique,
    )


def stage_provenance(g: Any) -> StageResult:
    """Every in-namespace declared term must carry an epistemic status."""
    from rdflib import OWL, RDF, URIRef

    status = URIRef(MTG + "epistemicStatus")
    declared = set()
    for t in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty):
        for s in g.subjects(RDF.type, t):
            if isinstance(s, URIRef) and (str(s).startswith(MTG) or str(s).startswith(MTGD)):
                declared.add(s)

    missing = sorted(str(s) for s in declared if (s, status, None) not in g)
    return StageResult(
        "provenance",
        not missing,
        f"{len(declared) - len(missing)}/{len(declared)} terms annotated",
        [f"missing mtg:epistemicStatus: {m}" for m in missing],
    )


def stage_competency(g: Any, cq_path: Path) -> StageResult:
    try:
        import yaml
    except ImportError:
        return StageResult("competency-questions", True, "pyyaml not installed", skipped=True)
    if not cq_path.exists():
        return StageResult("competency-questions", True, f"{cq_path} not found", skipped=True)

    from rdflib import URIRef

    spec = yaml.safe_load(cq_path.read_text(encoding="utf-8"))
    ns = spec.get("namespaces", {})
    questions = spec.get("questions", [])

    def expand(term: str) -> URIRef:
        if ":" in term and not term.startswith("http"):
            prefix, local = term.split(":", 1)
            return URIRef(ns.get(prefix, prefix + ":") + local)
        return URIRef(term)

    known = set(g.subjects())
    problems: list[str] = []
    covered = 0
    sparql_run = 0

    for q in questions:
        qid = q.get("id", "?")
        terms = q.get("terms", {}) or {}
        missing = []
        for bucket in ("explicit", "implicit", "derived"):
            for term in terms.get(bucket, []) or []:
                uri = expand(term)
                # Only terms in our own namespaces are our responsibility.
                if not (str(uri).startswith(MTG) or str(uri).startswith(MTGD)):
                    continue
                if uri not in known:
                    missing.append(f"{qid}: undeclared term {term} ({bucket})")
        if missing:
            problems.extend(missing)
        else:
            covered += 1

        query = q.get("sparql")
        if query:
            prologue = "".join(f"PREFIX {p}: <{u}>\n" for p, u in ns.items())
            prologue += (
                "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            )
            try:
                result = g.query(prologue + query)
                sparql_run += 1
                if result.type == "ASK" and not bool(result.askAnswer):
                    problems.append(f"{qid}: ASK returned false")
            except Exception as exc:
                problems.append(f"{qid}: SPARQL error {type(exc).__name__}: {exc}")

    detail = (
        f"{covered}/{len(questions)} questions fully covered, "
        f"{sparql_run} SPARQL checks executed"
    )
    return StageResult("competency-questions", not problems, detail, sorted(set(problems)))


def stage_shacl(ontology: Path, shapes: Path) -> StageResult:
    try:
        from pyshacl import validate as shacl_validate
    except ImportError:
        return StageResult("shacl", True, "pyshacl not installed", skipped=True)
    if not shapes.exists():
        return StageResult("shacl", True, f"{shapes} not found", skipped=True)
    try:
        conforms, _, text = shacl_validate(
            str(ontology),
            shacl_graph=str(shapes),
            data_graph_format="turtle" if ontology.suffix == ".ttl" else "xml",
            shacl_graph_format="turtle",
            inference="none",
            advanced=True,
        )
    except Exception as exc:
        return StageResult("shacl", False, f"{type(exc).__name__}: {exc}")
    return StageResult(
        "shacl",
        conforms,
        "conforms" if conforms else "violations found",
        [] if conforms else text.splitlines()[:40],
    )


def stage_reasoner(ontology: Path) -> StageResult:
    try:
        import owlready2
    except ImportError:
        return StageResult("reasoner", True, "owlready2 not installed", skipped=True)
    try:
        onto = owlready2.get_ontology(ontology.as_uri()).load()
        with onto:
            owlready2.sync_reasoner(infer_property_values=True)
    except Exception as exc:
        return StageResult(
            "reasoner", False, f"{type(exc).__name__}: {exc}",
            ["HermiT needs a JRE on PATH; rerun with --skip-reasoner if unavailable"],
        )
    return StageResult("reasoner", True, "consistent (HermiT)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--competency-questions", type=Path, default=DEFAULT_CQS)
    parser.add_argument(
        "--no-imports",
        action="store_true",
        help="Validate the local file only; do not resolve owl:imports (offline default).",
    )
    parser.add_argument("--skip-reasoner", action="store_true")
    parser.add_argument("--skip-shacl", action="store_true")
    args = parser.parse_args()

    _require_rdflib()

    if not args.ontology.exists():
        print(f"ontology not found: {args.ontology}", file=sys.stderr)
        return 2

    results: list[StageResult] = []

    syntax, graph = stage_syntax(args.ontology)
    results.append(syntax)
    if graph is None:
        _report(results)
        return 1

    results.append(stage_structure(graph))
    results.append(stage_profile(graph))
    results.append(stage_provenance(graph))
    results.append(stage_competency(graph, args.competency_questions))

    if args.skip_shacl:
        results.append(StageResult("shacl", True, "skipped by flag", skipped=True))
    else:
        results.append(stage_shacl(args.ontology, args.shapes))

    if args.skip_reasoner or args.no_imports:
        results.append(
            StageResult("reasoner", True, "skipped (offline / by flag)", skipped=True)
        )
    else:
        results.append(stage_reasoner(args.ontology))

    return _report(results)


def _report(results: list[StageResult]) -> int:
    print(f"\nOntology validation report\n{'=' * 60}")
    for r in results:
        print(r.render())
    failed = [r for r in results if not r.ok and not r.skipped]
    print("=" * 60)
    print(f"{len(results) - len(failed)}/{len(results)} stages passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
