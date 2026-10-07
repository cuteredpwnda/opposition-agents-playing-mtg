"""Pinned import closure and isolated OWLAPI/HermiT qualification."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rdflib import OWL, RDF, Graph

ROOT = Path(__file__).resolve().parents[2]
IMPORT_LOCK = ROOT / "data" / "ontology" / "imports.lock.json"
IMPORT_CACHE = ROOT / "data" / "cache" / "ontology_imports"


def import_closure(
    sources: list[Path], lock: Path = IMPORT_LOCK, cache: Path = IMPORT_CACHE,
) -> tuple[Graph, dict[str, str]]:
    entries = {entry["iri"]: entry for entry in json.loads(
        lock.read_text(encoding="utf-8"),
    )["imports"]}
    combined = Graph()
    hashes = {}
    pending = []
    loaded = set()
    for path in sources:
        content = path.read_bytes()
        graph = Graph().parse(data=content, format="turtle", publicID=path.resolve().as_uri())
        combined += graph
        loaded.update(str(iri) for iri in graph.subjects(RDF.type, OWL.Ontology))
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        pending.extend(str(iri) for iri in graph.objects(None, OWL.imports))
    while pending:
        iri = pending.pop()
        if iri in loaded:
            continue
        if iri not in entries:
            raise ValueError(f"Unpinned owl:imports IRI: {iri}")
        entry = entries[iri]
        path = cache / entry["file"]
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Cached ontology hash mismatch: {iri}")
        graph = Graph().parse(data=content, format=entry["format"], publicID=iri)
        combined += graph
        hashes[iri] = digest
        loaded.add(iri)
        pending.extend(str(value) for value in graph.objects(None, OWL.imports))
    # All imported axioms are now materialised; prohibit Java-side network loading.
    combined.remove((None, OWL.imports, None))
    return combined, hashes


def java_executable(explicit: Path | None = None) -> str:
    if explicit:
        if not explicit.is_file():
            raise FileNotFoundError(f"Java executable not found: {explicit}")
        return str(explicit.resolve())
    installed = shutil.which("java")
    if installed:
        return installed
    local = ROOT / ".tools" / "java" / "bin" / "java.exe"
    if local.is_file():
        return str(local)
    raise FileNotFoundError("Java is unavailable; install a JDK or pass --java")


def qualify_graph(graph: Graph, java: Path | None = None, timeout: float = 120) -> dict[str, Any]:
    import owlready2

    jar = Path(owlready2.__file__).parent / "hermit" / "HermiT.jar"
    if not jar.is_file():
        raise FileNotFoundError(f"Bundled HermiT jar not found: {jar}")
    executable = java_executable(java)
    checker = ROOT / "scripts" / "ontology" / "ReasonerCheck.java"
    checker_source = checker.read_bytes()
    jar_hash = hashlib.sha256(jar.read_bytes()).hexdigest()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="mtg-reasoner-") as temp:
        ontology = Path(temp) / "closure.rdf"
        frozen_checker = Path(temp) / checker.name
        frozen_checker.write_bytes(checker_source)
        graph.serialize(destination=ontology, format="xml")
        try:
            result = subprocess.run(
                [executable, "-Xmx1024M", "-cp", str(jar), str(frozen_checker), str(ontology)],
                capture_output=True, text=True, encoding="utf-8", timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8")
            raise TimeoutError(f"HermiT exceeded {timeout}s. Partial output: {stdout}") from exc
    output: dict[str, Any] = {
        "owlready2_version": version("owlready2"),
        "hermit_jar_sha256": jar_hash,
        "checker_sha256": hashlib.sha256(checker_source).hexdigest(),
        "returncode": result.returncode,
        "stderr": result.stderr,
        "elapsed_seconds": time.monotonic() - started,
    }
    for line in result.stdout.splitlines():
        for prefix in ("PROFILE", "REASONER"):
            if line.startswith(prefix + "="):
                output[prefix.lower()] = json.loads(line.partition("=")[2])
    if result.returncode != 0:
        raise RuntimeError(json.dumps(output, ensure_ascii=False))
    if "profile" not in output or "reasoner" not in output:
        raise RuntimeError(f"Incomplete Java qualification output: {result.stdout!r}")
    return output


def qualify_ontology(
    sources: list[Path], java: Path | None = None, timeout: float = 180,
) -> dict[str, Any]:
    graph, hashes = import_closure(sources)
    result = qualify_graph(graph, java, timeout)
    result.update(source_sha256=hashes, closure_triples=len(graph))
    return result
