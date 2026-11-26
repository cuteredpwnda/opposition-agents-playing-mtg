"""Export a review projection of the MTG ontology for KG Workbench.

The Turtle remains authoritative: the editor cannot express arbitrary OWL axioms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.compare import to_canonical_graph

ROOT = Path(__file__).resolve().parents[1]
MTG = Namespace("http://purl.org/mtg/ontology#")
MTGD = Namespace("http://purl.org/mtg/ontology/decision#")
WORKBENCH_REVISION = "4d61c37f4e52c22c344aab004ada7622bec9fead"
MODULES = {
    "rules": ("Rules and vocabulary", "Normative rule references, taxonomy and keywords."),
    "cards": (
        "Card descriptions", "Immutable designs, printings, abilities and printed qualities.",
    ),
    "gameplay": (
        "Gameplay situations", "Objects, players, events, roles and zone/state relations.",
    ),
    "strategy": ("Strategies and combos", "Curated plans, decks, combos, answers and outcomes."),
    "decisions": ("Agent decisions", "Beliefs, policies, objectives and decision provenance."),
    "evidence": ("Learned evidence", "Reified induced claims and their self-play provenance."),
    "dependencies": (
        "External dependencies", "Referenced external classes; imports are not resolved.",
    ),
}
ROOT_MODULES = {
    "rules": {"ComprehensiveRule", "CardType", "Subtype", "Supertype", "Keyword",
              "KeywordAction", "Format"},
    "cards": {"CardDesign", "CardPrinting", "Ability", "GameQuality", "ValueRegion"},
    "gameplay": {"GameObject", "Player", "Zone", "Game", "Turn", "Phase", "Step",
                 "GameEvent", "GameAction", "GameSituation", "GameRole"},
    "strategy": {"Strategy", "Archetype", "Combo", "Outcome", "Deck"},
    "evidence": {"InducedAssertion", "SelfPlayRun"},
}
PROJECTION_NOTICE = (
    "Review projection, not an OWL round-trip or rules engine. Modules are editor groupings, "
    "not independent owl:imports. Named classes and one named parent per class are shown. "
    "Union endpoints expand into separate display relations, not additional RDF axioms. "
    "Missing relation endpoints use owl:Thing, not an invented domain/range assertion. "
    "Restrictions, multiple inheritance, keys, property chains, disjointness, annotations "
    "and imports remain authoritative in Turtle and the companion RDF JSON. Vocabulary "
    "individuals are class examples, not a populated card/combo database. External class "
    "stubs do not imply that their ontologies were loaded or validated."
)


def named_classes(graph: Graph, node: Any) -> list[URIRef]:
    if isinstance(node, URIRef):
        return [node]
    union = graph.value(node, OWL.unionOf)
    if union is not None:
        return sorted(
            {member for member in graph.items(union) if isinstance(member, URIRef)}, key=str,
        )
    return []


def class_module(graph: Graph, term: URIRef, visited: frozenset = frozenset()) -> str:
    if str(term).startswith(str(MTGD)):
        return "decisions"
    if not str(term).startswith(str(MTG)):
        return "dependencies"
    for module, roots in ROOT_MODULES.items():
        if str(term).removeprefix(str(MTG)) in roots:
            return module
    if term in visited:
        raise ValueError(f"Cyclic class hierarchy at {term}")
    parents = sorted(
        (p for p in graph.objects(term, RDFS.subClassOf)
         if isinstance(p, URIRef) and str(p).startswith(str(MTG))), key=str,
    )
    if parents:
        return class_module(graph, parents[0], visited | {term})
    raise ValueError(f"No review module assigned to {term}")


def label(graph: Graph, term: URIRef) -> str:
    labels = sorted(str(value) for value in graph.objects(term, RDFS.label)
                    if isinstance(value, Literal) and value.language in (None, "en"))
    return labels[0] if labels else str(term).rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def description(graph: Graph, term: URIRef) -> str:
    comments = sorted(str(value) for value in graph.objects(term, RDFS.comment))
    annotations = [
        f"{graph.namespace_manager.normalizeUri(predicate)}: {value}"
        for predicate in sorted(set(graph.subjects(RDF.type, OWL.AnnotationProperty)), key=str)
        for value in sorted(graph.objects(term, predicate), key=str)
    ]
    return "\n\n".join([f"Source IRI: {term}", *comments, *annotations])


def endpoints(graph: Graph, prop: URIRef, predicate: URIRef) -> list[URIRef]:
    result = {c for node in graph.objects(prop, predicate) for c in named_classes(graph, node)}
    if not result:
        inverse_predicate = RDFS.range if predicate == RDFS.domain else RDFS.domain
        inverses = set(graph.objects(prop, OWL.inverseOf)) | set(
            graph.subjects(OWL.inverseOf, prop),
        )
        result = {c for inverse in inverses for node in graph.objects(inverse, inverse_predicate)
                  for c in named_classes(graph, node)}
    return sorted(result, key=str)


def export_workbench(schema: Path, vocabulary: Path, cqs: Path) -> tuple[dict, dict]:
    graph = Graph().parse(schema, format="turtle")
    factual = Graph().parse(vocabulary, format="turtle")
    cq_data = yaml.safe_load(cqs.read_text(encoding="utf-8"))
    classes = {c for c in graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
    local_classes = set(classes)
    object_properties = sorted(set(graph.subjects(RDF.type, OWL.ObjectProperty)), key=str)
    datatype_properties = sorted(set(graph.subjects(RDF.type, OWL.DatatypeProperty)), key=str)
    class_rows: dict[URIRef, dict] = {}
    warnings: list[str] = []
    relation_ids: dict[URIRef, list[str]] = {}
    property_domains: dict[URIRef, list[URIRef]] = {}

    for term in local_classes:
        classes.update(p for p in graph.objects(term, RDFS.subClassOf) if isinstance(p, URIRef))
    relations = []
    for prop in object_properties:
        domains = endpoints(graph, prop, RDFS.domain)
        ranges = endpoints(graph, prop, RDFS.range)
        if not domains or not ranges:
            warnings.append(f"{prop}: unspecified endpoints displayed as owl:Thing")
        domains, ranges = domains or [OWL.Thing], ranges or [OWL.Thing]
        property_domains[prop] = domains
        classes.update(domains + ranges)
        relation_ids[prop] = []
        for domain in domains:
            for range_ in ranges:
                relation_id = f"{prop}|{domain}|{range_}"
                relation_ids[prop].append(relation_id)
                inverse = sorted(set(graph.objects(prop, OWL.inverseOf)) | set(
                    graph.subjects(OWL.inverseOf, prop),
                ), key=str)
                relations.append({
                    "id": relation_id, "name": graph.namespace_manager.normalizeUri(prop),
                    "domainClassId": str(domain), "rangeClassId": str(range_),
                    "description": description(graph, prop) + "\nDisplay endpoint projection only.",
                    "inverseName": (
                        graph.namespace_manager.normalizeUri(inverse[0]) if inverse else None
                    ),
                })
    for prop in datatype_properties:
        domains = endpoints(graph, prop, RDFS.domain) or [OWL.Thing]
        if domains == [OWL.Thing]:
            warnings.append(f"{prop}: unspecified attribute domain displayed as owl:Thing")
        property_domains[prop] = domains
        classes.update(domains)
    for term in sorted(classes, key=str):
        parents = sorted((p for p in graph.objects(term, RDFS.subClassOf)
                          if isinstance(p, URIRef)), key=str)
        if len(parents) > 1:
            warnings.append(f"{term}: only one of {len(parents)} named parents displayed")
        class_rows[term] = {
            "id": str(term), "name": label(graph, term),
            "moduleId": class_module(graph, term),
            "description": description(graph, term),
            "parentClassId": str(parents[0]) if parents else None, "attributes": [],
        }
    for prop in datatype_properties:
        ranges = endpoints(graph, prop, RDFS.range)
        if len(ranges) != 1:
            raise ValueError(f"Expected one datatype for {prop}: {ranges}")
        datatype = graph.namespace_manager.normalizeUri(ranges[0])
        for domain in property_domains[prop]:
            class_rows[domain]["attributes"].append({
                "id": f"{prop}|{domain}", "name": graph.namespace_manager.normalizeUri(prop),
                "dataType": datatype, "required": False, "description": description(graph, prop),
            })
    modules = [{"id": key, "name": name, "description": text}
               for key, (name, text) in MODULES.items()]
    questions, notes = [], []
    for index, cq in enumerate(cq_data["questions"]):
        terms = [
            URIRef(cq_data["namespaces"][prefix] + local)
            for value in cq["terms"].values() for name in value
            for prefix, local in [name.split(":", 1)]
        ]
        linked_classes = [term for term in terms if term in class_rows]
        linked_properties = [term for term in terms if term in relation_ids]
        cq_modules = {class_rows[c]["moduleId"] for c in linked_classes}
        for term in terms:
            cq_modules.update(class_rows[c]["moduleId"] for c in property_domains.get(term, []))
        questions.append({
            "id": cq["id"], "question": cq["question"], "sortOrder": index,
            "subjectClassId": str(linked_classes[0]) if linked_classes else None,
            # A CQ can name several predicates. Never arbitrarily pin a single displayed pair.
            "predicateRelationId": (
                relation_ids[linked_properties[0]][0]
                if len(linked_properties) == 1 and len(relation_ids[linked_properties[0]]) == 1
                else None
            ),
            "modules": [{"id": key, "name": MODULES[key][0]} for key in sorted(cq_modules)],
        })
        notes.append({
            "targetType": "cq", "targetId": cq["id"], "authorName": "MTG exporter",
            "body": yaml.safe_dump(cq, allow_unicode=True, sort_keys=False),
        })
    examples = []
    merged = graph + factual
    for term in sorted(class_rows, key=str):
        for subject in sorted(set(merged.subjects(RDF.type, term)), key=str):
            if not isinstance(subject, URIRef):
                continue
            examples.append({
                "id": f"{term}|{subject}", "targetType": "class", "targetId": str(term),
                "value": label(merged, subject) + "\nSource IRI: " + str(subject),
                "isInstanceCandidate": False,
            })
    payload = {
        "name": "Magic: The Gathering ontology — modular review",
        "version": "2.0.0", "defaultLanguage": "en",
        "usecase": PROJECTION_NOTICE, "modules": modules,
        "classes": list(class_rows.values()), "relations": relations,
        "competencyQuestions": questions, "notes": notes, "examples": examples,
    }
    report = {
        "workbench_revision": WORKBENCH_REVISION,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (schema, vocabulary, cqs)},
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "projection_notice": PROJECTION_NOTICE, "warnings": warnings,
        "counts": {"schema_named_classes": len(local_classes),
                   "display_classes": len(class_rows), "display_relations": len(relations),
                   "object_properties": len(object_properties),
                   "datatype_properties": len(datatype_properties),
                   "competency_questions": len(questions), "examples": len(examples)},
    }
    return payload, report


def rdf_json(schema: Path, vocabulary: Path) -> dict:
    """Standard RDF/JSON, with canonical blank nodes for reproducible output."""
    graph = to_canonical_graph(
        Graph().parse(schema, format="turtle") + Graph().parse(vocabulary, format="turtle"),
    )
    result: dict[str, dict[str, list[dict]]] = {}
    for subject, predicate, obj in sorted(
        graph, key=lambda triple: tuple(term.n3() for term in triple),
    ):
        key = "_:" + str(subject) if isinstance(subject, BNode) else str(subject)
        if isinstance(obj, Literal):
            value = {"type": "literal", "value": str(obj)}
            if obj.language:
                value["lang"] = obj.language
            if obj.datatype:
                value["datatype"] = str(obj.datatype)
        else:
            value = {"type": "bnode" if isinstance(obj, BNode) else "uri",
                     "value": "_:" + str(obj) if isinstance(obj, BNode) else str(obj)}
        result.setdefault(key, {}).setdefault(str(predicate), []).append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=ROOT / "data/ontology/mtg-ontology-v2.0.ttl")
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "data/ontology/mtg-cr-types.ttl")
    parser.add_argument("--cqs", type=Path, default=ROOT / "data/competency_questions.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ontology/mtg-workbench.json")
    args = parser.parse_args()
    payload, report = export_workbench(args.schema, args.vocabulary, args.cqs)
    outputs = {
        args.output: payload,
        args.output.with_suffix(".report.json"): report,
        args.output.with_suffix(".rdf.json"): rdf_json(args.schema, args.vocabulary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, data in outputs.items():
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
