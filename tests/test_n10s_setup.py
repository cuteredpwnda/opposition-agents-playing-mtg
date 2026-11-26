from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.config import settings
from src.knowledge.n10s_setup import GRAPH_CONFIG, N10sSetup


class Result:
    def __init__(self, rows):
        self.rows = rows
        self.consume = AsyncMock()

    async def single(self):
        return self.rows[0] if self.rows else None

    def __aiter__(self):
        async def iterate():
            for row in self.rows:
                yield row
        return iterate()


class Driver:
    def __init__(self, config=None, stats=None):
        self.config = config or {}
        self.stats = stats
        self.calls = []
        self.results = []
        self.verify_connectivity = AsyncMock()
        self.close = AsyncMock()

    def session(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def run(self, query, **params):
        self.calls.append((query, params))
        if "graphconfig.show" in query:
            rows = [{"param": key, "value": value} for key, value in self.config.items()]
        elif "import.fetch" in query:
            rows = [self.stats] if self.stats is not None else []
        else:
            rows = []
        result = Result(rows)
        self.results.append(result)
        return result


def setup_with_driver(monkeypatch, driver):
    monkeypatch.setattr(
        "src.knowledge.n10s_setup.AsyncGraphDatabase.driver", lambda *_args, **_kwargs: driver,
    )
    return N10sSetup()


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [None, GRAPH_CONFIG])
async def test_setup_preserves_compatible_config_and_consumes_writes(monkeypatch, config):
    driver = Driver(config)
    setup = setup_with_driver(monkeypatch, driver)
    await setup.init_graph_config()
    driver.verify_connectivity.assert_awaited_once()
    queries = [query for query, _ in driver.calls]
    assert not any("drop" in query for query in queries)
    assert any("Resource" in query for query in queries)
    assert any("graphconfig.init" in query for query in queries) == (config is None)
    for (query, _), result in zip(driver.calls, driver.results):
        if "show" not in query:
            result.consume.assert_awaited_once()


@pytest.mark.asyncio
async def test_incompatible_config_and_connectivity_failure_prevent_writes(monkeypatch):
    driver = Driver({**GRAPH_CONFIG, "handleVocabUris": "KEEP"})
    setup = setup_with_driver(monkeypatch, driver)
    with pytest.raises(RuntimeError, match="Existing config was not changed"):
        await setup.init_graph_config()
    assert len(driver.calls) == 1
    driver.calls.clear()
    driver.verify_connectivity.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError, match="offline"):
        await setup.init_graph_config()
    assert driver.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path,format_", [
    (r"data\ontology\mtg-ontology-v2.0.ttl", "Turtle"),
    (r"data\ontology\mtg-ontology-v1.0.owl", "RDF/XML"),
])
async def test_import_uses_correct_format_and_returns_counts(monkeypatch, path, format_):
    stats = {"terminationStatus": "OK", "triplesLoaded": 100, "triplesParsed": 200}
    driver = Driver(stats=stats)
    setup = setup_with_driver(monkeypatch, driver)
    assert await setup.import_ontology(path) == stats
    assert driver.calls[0][1]["format"] == format_
    assert driver.calls[0][1]["uri"].startswith("file:///import/ontology/")


@pytest.mark.asyncio
@pytest.mark.parametrize("stats", [
    None, {}, {"terminationStatus": "KO", "extraInfo": "bad input"},
    {"terminationStatus": "OK", "triplesLoaded": 0, "triplesParsed": 10},
    {"terminationStatus": "OK", "triplesLoaded": 10},
    {"terminationStatus": "OK", "triplesLoaded": True, "triplesParsed": 10},
])
async def test_failed_or_unaccounted_import_is_not_success(monkeypatch, stats):
    setup = setup_with_driver(monkeypatch, Driver(stats=stats))
    with pytest.raises(RuntimeError):
        await setup.import_instances()


@pytest.mark.asyncio
async def test_full_setup_loads_schema_and_factual_vocabulary(monkeypatch):
    stats = {"terminationStatus": "OK", "triplesLoaded": 100, "triplesParsed": 100}
    driver = Driver(stats=stats)
    setup = setup_with_driver(monkeypatch, driver)
    reports = await setup.full_setup()
    assert set(reports) == {"schema", "schema_rdf", "vocabulary"}
    imports = [params for query, params in driver.calls if "import.fetch" in query]
    assert len(imports) == 3
    assert imports[-1]["uri"].endswith("mtg-cr-types.ttl")
    assert all(p["format"] == "Turtle" for p in imports)
    assert any("CardDesign" in query for query, _ in driver.calls)
    assert settings.ontology_path.endswith("mtg-ontology-v2.0.ttl")


def test_format_and_server_uri_guards():
    assert N10sSetup._file_uri(r"data\ontology\file with space.ttl") == (
        "file:///import/ontology/file%20with%20space.ttl"
    )
    with pytest.raises(ValueError, match="Unsupported RDF"):
        N10sSetup._rdf_format("bad.csv")


@pytest.mark.asyncio
async def test_training_stage_stops_after_setup_failure_and_closes_driver(monkeypatch):
    source = Path("scripts") / "train_pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    stage = next(node for node in tree.body
                 if isinstance(node, ast.AsyncFunctionDef) and node.name == "stage_2_build_kg")
    context = {"logger": logging.getLogger("pipeline-test")}
    exec(compile(ast.Module(body=[stage], type_ignores=[]), str(source), "exec"), context)
    setup_with_driver(monkeypatch, Driver())
    full_setup = AsyncMock(side_effect=ConnectionError("offline"))
    close = AsyncMock()
    monkeypatch.setattr(N10sSetup, "full_setup", full_setup)
    monkeypatch.setattr(N10sSetup, "close", close)
    assert await context["stage_2_build_kg"]() is False
    close.assert_awaited_once()
