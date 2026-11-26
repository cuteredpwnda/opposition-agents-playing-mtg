from __future__ import annotations

import gzip
import json
from argparse import Namespace

import httpx
import pytest

from scripts import fetch_card_data
from src.integrations.scryfall_bulk import bulk_download_info, download_bulk


@pytest.mark.parametrize("jsonl", [False, True])
async def test_bulk_formats_preserve_card_records_and_write_provenance(tmp_path, jsonl):
    rows = [{"name": "Test Card", "oracle_id": "test", "oracle_text": "Keep immutable."}]
    data = (
        gzip.compress(("\n".join(json.dumps(row) for row in rows) + "\n").encode())
        if jsonl else json.dumps(rows).encode()
    )
    field = "jsonl_download_uri" if jsonl else "download_uri"
    size = "compressed_size" if jsonl else "size"
    item = {"type": "oracle_cards", field: "https://data.scryfall.io/test",
            size: len(data), "updated_at": "2026-10-05"}

    def respond(request):
        return httpx.Response(200, stream=httpx.ByteStream(data))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        destination = tmp_path / "cards.json"
        provenance = await download_bulk(client, item, destination)
    assert json.loads(destination.read_text()) == rows
    assert provenance["record_count"] == 1
    assert provenance["source_updated_at"] == "2026-10-05"
    assert len(provenance["sha256"]) == 64
    assert provenance["bytes"] == destination.stat().st_size
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("data", [
    b"not gzip", gzip.compress(b"invalid json\n"), gzip.compress(b"[]\n"), gzip.compress(b""),
])
async def test_bad_download_preserves_old_cache_and_cleans_staging(tmp_path, data):
    path = tmp_path / "cards.json"
    path.write_text('[{"name":"Old"}]')
    item = {"type": "oracle_cards", "jsonl_download_uri": "https://data.scryfall.io/test"}
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, stream=httpx.ByteStream(data))
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises((ValueError, OSError)):
            await download_bulk(client, item, path)
    assert path.read_text() == '[{"name":"Old"}]'
    assert list(tmp_path.iterdir()) == [path]


async def test_size_mismatch_does_not_publish(tmp_path):
    item = {"type": "rulings", "download_uri": "https://data.scryfall.io/test", "size": 100}
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, stream=httpx.ByteStream(b"[]"))
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="Incomplete"):
            await download_bulk(client, item, tmp_path / "rulings.json")
    assert not list(tmp_path.iterdir())


def test_unknown_bulk_representation_is_explicit():
    with pytest.raises(ValueError, match="No supported"):
        bulk_download_info({"type": "future"})


async def test_refresh_failure_preserves_whole_previous_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_card_data, "DATA_DIR", tmp_path)
    for name, constant in [
        ("oracle-cards.json", "ORACLE_PATH"), ("rulings.json", "RULINGS_PATH"),
        ("by_name.json", "INDEX_PATH"),
    ]:
        path = tmp_path / name
        path.write_text("old " + name)
        monkeypatch.setattr(fetch_card_data, constant, path)

    async def metadata(client, kind):
        return {"type": kind}

    async def download(client, item, destination):
        if item["type"] == "rulings":
            raise ValueError("Broken rulings")
        destination.write_text('[{"name":"New","oracle_id":"new"}]')
        return {"sha256": "new"}

    monkeypatch.setattr(fetch_card_data, "fetch_bulk_metadata", metadata)
    monkeypatch.setattr(
        fetch_card_data, "bulk_download_info", lambda _: ("https://data.scryfall.io/test", 1, True)
    )
    monkeypatch.setattr(fetch_card_data, "download_bulk", download)
    with pytest.raises(ValueError, match="Broken rulings"):
        await fetch_card_data.main(Namespace(refresh=True, skip_rulings=False))
    assert (tmp_path / "oracle-cards.json").read_text() == "old oracle-cards.json"
    assert (tmp_path / "rulings.json").read_text() == "old rulings.json"
    assert (tmp_path / "by_name.json").read_text() == "old by_name.json"
    assert len(list(tmp_path.iterdir())) == 3
