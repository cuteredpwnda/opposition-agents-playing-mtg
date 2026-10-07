"""Fetch and checksum-lock the external ontology documents used for reasoning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
from rdflib import OWL, Graph

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "ontology" / "imports.lock.json"
CACHE = ROOT / "data" / "cache" / "ontology_imports"
SOURCES = [
    ("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl",
     "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl", "dul.ttl", "turtle"),
    ("http://www.w3.org/ns/prov-o#", "https://www.w3.org/ns/prov-o.ttl", "prov.ttl", "turtle"),
    ("http://www.w3.org/2004/02/skos/core",
     "https://www.w3.org/2004/02/skos/core.rdf", "skos.rdf", "xml"),
    ("http://www.w3.org/2006/time#", "https://www.w3.org/2006/time.ttl", "time.ttl", "turtle"),
]


def fetch_imports(lock: Path = LOCK, cache: Path = CACHE, update_lock: bool = False) -> dict:
    if update_lock:
        entries = [
            {"iri": iri, "url": url, "file": filename, "format": format_}
            for iri, url, filename, format_ in SOURCES
        ]
    else:
        entries = json.loads(lock.read_text(encoding="utf-8"))["imports"]
    cache.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for entry in entries:
        response = requests.get(entry["url"], timeout=60)
        response.raise_for_status()
        data = response.content
        digest = hashlib.sha256(data).hexdigest()
        if not update_lock and digest != entry["sha256"]:
            raise ValueError(f"Import changed: {entry['iri']}; cached file was not replaced")
        graph = Graph().parse(data=data, format=entry["format"], publicID=entry["iri"])
        nested = sorted(str(iri) for iri in graph.objects(None, OWL.imports))
        known = {item["iri"] for item in entries}
        if not set(nested) <= known:
            raise ValueError(f"Unpinned recursive imports from {entry['iri']}: {nested}")
        downloaded.append((cache / entry["file"], data))
        entry.update(sha256=digest, triples=len(graph), imports=nested)
    for path, data in downloaded:
        path.write_bytes(data)
    manifest = {"schema_version": 1, "imports": entries}
    if update_lock:
        lock.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument(
        "--update-lock", action="store_true",
        help="Explicitly freeze new publisher snapshots instead of enforcing hashes.",
    )
    args = parser.parse_args()
    result = fetch_imports(args.lock, args.cache, args.update_lock)
    print(f"Verified {len(result['imports'])} pinned external ontology documents")


if __name__ == "__main__":
    main()
