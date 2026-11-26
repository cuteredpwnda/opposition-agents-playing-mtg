"""Validated, atomic downloads of Scryfall JSON or compressed JSONL snapshots."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HEADERS = {"User-Agent": "OppositionAgentsMTG/1.0 (research)", "Accept": "application/json"}


def bulk_download_info(item: dict[str, Any]) -> tuple[str, int, bool]:
    if item.get("download_uri"):
        return item["download_uri"], int(item.get("size", 0)), False
    if item.get("jsonl_download_uri"):
        return item["jsonl_download_uri"], int(item.get("compressed_size", 0)), True
    raise ValueError(f"No supported bulk download URI for {item.get('type')!r}")


async def fetch_bulk_metadata(client: httpx.AsyncClient, bulk_type: str) -> dict[str, Any]:
    response = await client.get("https://api.scryfall.com/bulk-data")
    response.raise_for_status()
    for item in response.json()["data"]:
        if item["type"] == bulk_type:
            bulk_download_info(item)
            return item
    raise ValueError(f"Scryfall bulk type {bulk_type!r} not found")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def download_bulk(
    client: httpx.AsyncClient, item: dict[str, Any], destination: Path,
) -> dict[str, Any]:
    """Publish a JSON array only after complete download, decoding and validation."""
    url, expected_size, jsonl = bulk_download_info(item)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scryfall-", dir=destination.parent) as directory:
        raw = Path(directory) / "download"
        converted = Path(directory) / "snapshot.json"
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with raw.open("wb") as stream:
                async for chunk in response.aiter_raw():
                    stream.write(chunk)
        if expected_size and raw.stat().st_size != expected_size:
            raise ValueError(
                f"Incomplete {item['type']} download: expected {expected_size} bytes, "
                f"got {raw.stat().st_size}"
            )
        if jsonl:
            count = 0
            with gzip.open(raw, "rt", encoding="utf-8") as source, converted.open(
                "w", encoding="utf-8", newline="\n",
            ) as output:
                output.write("[")
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Bulk JSONL line {line_number} must be an object")
                    output.write((",\n" if count else "\n") + json.dumps(row, ensure_ascii=False))
                    count += 1
                output.write("\n]\n")
        else:
            rows = json.loads(raw.read_text("utf-8"))
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Bulk JSON must be an array of objects")
            count = len(rows)
            shutil.copyfile(raw, converted)
        if not count:
            raise ValueError(f"Empty {item['type']} bulk snapshot")
        provenance = {
            "type": item["type"],
            "source_updated_at": item.get("updated_at"),
            "download_uri": url,
            "source_format": "gzip-jsonl" if jsonl else "json-array",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 -- Python 3.10
            "download_bytes": raw.stat().st_size,
            "download_sha256": file_sha256(raw),
            "record_count": count,
            "bytes": converted.stat().st_size,
            "sha256": file_sha256(converted),
        }
        converted.replace(destination)
    return provenance
