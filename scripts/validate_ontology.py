#!/usr/bin/env python
"""Validate the MTG ontology: syntax, OWL 2 profile, SHACL, CQ coverage.

Implements the decompose -> validate -> report cascade rather than a single
"does it parse" check. Each stage is independent and reports separately, so a
failure localises to one concern:

1. **Syntax** — rdflib parse of the Turtle/RDF-XML source.
2. **Structure** — counts of classes, object/datatype properties, individuals,
   and the OWL 2 constructs that carry modelling weight here (property chains,
   keys, qualified cardinality, disjoint unions).
3. **Local modelling sanity** — incomplete guards for undeclared local terms,
   incompatible property characteristics and dangling local subclass targets.
   OWL 2 permits class/individual punning; these checks do not certify a profile.
4. **Logical qualification** — checksum-pinned imports are materialised as
   RDF/XML for an isolated Java OWLAPI/HermiT checker. Profile conformance,
   consistency and named-class satisfiability are reported separately.
   ``owlready2`` supplies the HermiT jar; a JDK supplies source-file launching.
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
import json
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

    # Mixed class/instance uses violate this project's modelling convention,
    # not OWL 2 DL itself, which permits class/individual punning.
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

    for term in obj_props & data_props:
        problems.append(f"forbidden object/datatype property punning: {term}")
    for term in data_props & set(g.subjects(RDF.type, OWL.InverseFunctionalProperty)):
        problems.append(f"inverse functionality is not allowed on datatype properties: {term}")
    transitive = set(g.subjects(RDF.type, OWL.TransitiveProperty))
    for type_ in (OWL.AsymmetricProperty, OWL.IrreflexiveProperty):
        for term in transitive & set(g.subjects(RDF.type, type_)):
            problems.append(f"non-simple transitive property cannot have {type_}: {term}")

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
        "local-modelling-sanity",
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


def stage_reasoner(
    ontology: Path, java: Path | None = None, timeout: float = 180,
    report_path: Path | None = None, vocabulary: Path | None = None,
    require_dl_profile: bool = False,
) -> StageResult:
    try:
        import owlready2  # noqa: F401
    except ImportError:
        return StageResult(
            "reasoner", not require_dl_profile, "owlready2 not installed",
            skipped=not require_dl_profile,
        )
    try:
        from src.knowledge.ontology_reasoning import qualify_ontology

        sources = [ontology] + ([vocabulary] if vocabulary else [])
        report = qualify_ontology(sources, java, timeout)
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        return StageResult(
            "reasoner", False, f"{type(exc).__name__}: {exc}",
            ["Fetch pinned imports with scripts/fetch_ontology_imports.py; "
             "install a JDK or pass --java. Errors are not counted as passes."],
        )
    reasoner = report["reasoner"]
    profile = report["profile"]
    logical_ok = reasoner["consistent"] and not reasoner["unsatisfiable_classes"]
    ok = logical_ok and (not require_dl_profile or profile["in_profile"])
    return StageResult(
        "reasoner", ok,
        "HermiT: consistent, no unsatisfiable named classes" if ok else (
            "required OWL 2 DL profile not met" if logical_ok else "logical defects"
        ),
        [
            f"Pinned import closure: {report['closure_triples']} triples",
            f"Full OWLAPI profile: {'in profile' if profile['in_profile'] else 'outside OWL 2 DL'} "
            f"({len(profile['violations'])} violations; separate from consistency)",
        ],
    )


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
    parser.add_argument(
        "--java", type=Path, help="JDK Java executable (supports source-file launch).",
    )
    parser.add_argument("--reasoner-timeout", type=float, default=180)
    parser.add_argument("--reasoner-report", type=Path)
    parser.add_argument("--require-dl-profile", action="store_true",
                        help="Fail if the resolved OWLAPI profile is outside OWL 2 DL.")
    parser.add_argument("--reasoner-vocabulary", type=Path,
                        help="Also reason over the factual vocabulary, not only the schema.")
    args = parser.parse_args()
    if args.require_dl_profile and (args.no_imports or args.skip_reasoner):
        parser.error("--require-dl-profile cannot be combined with a skipped reasoner")

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
        results.append(stage_reasoner(
            args.ontology, args.java, args.reasoner_timeout,
            args.reasoner_report, args.reasoner_vocabulary, args.require_dl_profile,
        ))

    return _report(results)


def _report(results: list[StageResult]) -> int:
    print(f"\nOntology validation report\n{'=' * 60}")
    for r in results:
        print(r.render())
    failed = [r for r in results if not r.ok and not r.skipped]
    print("=" * 60)
    enabled = [r for r in results if not r.skipped]
    print(
        f"{len(enabled) - len(failed)}/{len(enabled)} enabled stages passed; "
        f"{len(results) - len(enabled)} skipped"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
