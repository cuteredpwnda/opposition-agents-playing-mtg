"""Complete wire configs exported from the pinned engine's format registry."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.integrations.phase_rs.client import PROTOCOL_VERSION


@lru_cache(maxsize=1)
def _load_defaults() -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(__file__).with_name("format_defaults.json").read_text("utf-8"))
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("Format registry is stale; rerun scripts.export_phase_rs_formats")
    return payload["configs"]


def format_config(name: str) -> dict[str, Any]:
    configs = _load_defaults()
    if name not in configs:
        raise ValueError(f"Unsupported engine format {name!r}; choose from {sorted(configs)}")
    return copy.deepcopy(configs[name])
