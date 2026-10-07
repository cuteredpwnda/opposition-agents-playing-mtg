"""Build a versioned RDF dataset and serve it read-only with local Apache Fuseki."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from rdflib import Dataset, Graph, URIRef
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID

from src.knowledge.ontology_reasoning import (
    IMPORT_CACHE,
    IMPORT_LOCK,
    ROOT,
    import_closure,
    java_executable,
)

SCHEMA = ROOT / "data" / "ontology" / "mtg-ontology-v2.0.ttl"
VOCABULARY = ROOT / "data" / "ontology" / "mtg-cr-types.ttl"


def build_dataset(run_dir: Path) -> tuple[Path, dict]:
    _, hashes = import_closure([SCHEMA, VOCABULARY])
    dataset = Dataset()
    default = dataset.graph(DATASET_DEFAULT_GRAPH_ID)
    counts = {}
    sources = [
        ("http://purl.org/mtg/graph/schema", SCHEMA, "turtle"),
        ("http://purl.org/mtg/graph/vocabulary", VOCABULARY, "turtle"),
    ]
    imports = json.loads(IMPORT_LOCK.read_text(encoding="utf-8"))["imports"]
    sources += [(entry["iri"], IMPORT_CACHE / entry["file"], entry["format"])
                for entry in imports]
    for iri, path, format_ in sources:
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes.get(str(path), hashes.get(iri)):
            raise ValueError(f"Ontology source changed while building the snapshot: {path}")
        graph = Graph().parse(data=content, format=format_, publicID=iri)
        target = dataset.graph(URIRef(iri))
        for triple in graph:
            target.add(triple)
            if path in (SCHEMA, VOCABULARY):
                default.add(triple)
        counts[iri] = len(target)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "snapshot.trig"
    dataset.serialize(destination=snapshot, format="trig")
    manifest = {
        "source_sha256": hashes, "named_graph_triples": counts,
        "default_graph": "asserted local schema + factual vocabulary; no implicit entailment",
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return snapshot, manifest


def write_config(run_dir: Path, snapshot: Path) -> Path:
    config = run_dir / "config.ttl"
    config.write_text(
        "@prefix fuseki: <http://jena.apache.org/fuseki#> .\n"
        "@prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .\n"
        "<#service> a fuseki:Service ; fuseki:name \"mtg\" ;\n"
        "  fuseki:endpoint [ fuseki:operation fuseki:query ; fuseki:name \"query\" ] ;\n"
        "  fuseki:endpoint [ fuseki:operation fuseki:gsp_r ; fuseki:name \"data\" ] ;\n"
        "  fuseki:dataset <#dataset> .\n"
        "<#dataset> a ja:MemoryDataset ;\n"
        f"  ja:data <{snapshot.resolve().as_uri()}> .\n",
        encoding="utf-8",
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3030)
    parser.add_argument("--java", type=Path)
    parser.add_argument("--fuseki-jar", type=Path,
                        default=ROOT / ".tools" / "fuseki" / "fuseki-server.jar")
    parser.add_argument("--run-dir", type=Path, default=ROOT / ".tools" / "fuseki-run")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    snapshot, manifest = build_dataset(args.run_dir)
    config = write_config(args.run_dir, snapshot)
    print(json.dumps(manifest["named_graph_triples"]), flush=True)
    if args.build_only:
        return
    if not args.fuseki_jar.is_file():
        raise FileNotFoundError(f"Apache Fuseki jar not found: {args.fuseki_jar}")
    result = subprocess.run([
        java_executable(args.java), "-Xmx1024M", "-jar", str(args.fuseki_jar.resolve()),
        "--localhost", f"--port={args.port}", "--timeout=10000", f"--config={config.resolve()}",
    ], env={**os.environ, "FUSEKI_BASE": str(args.run_dir.resolve())},
        cwd=args.run_dir.resolve())
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
