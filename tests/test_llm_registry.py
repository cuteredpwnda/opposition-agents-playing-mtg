"""Tests for ``src/agents/llm_registry.py``."""

from __future__ import annotations

import pytest

from src.agents.llm_registry import LLM_REGISTRY, get_llm_spec, list_llms


def test_registry_non_empty():
    assert LLM_REGISTRY


@pytest.mark.parametrize("name", list(LLM_REGISTRY))
def test_every_entry_within_seven_billion_budget(name):
    spec = get_llm_spec(name)
    assert spec.fits_seven_billion_budget, (
        f"{name} declared {spec.params_billion}B params, exceeds 7B budget"
    )
    assert spec.ollama_tag
    assert spec.family
    assert spec.vram_gb_estimate > 0


def test_list_llms_filters_by_vram():
    small = list_llms(max_vram_gb=3.0)
    assert all(s.vram_gb_estimate <= 3.0 for s in small)
    assert any(s.name == "gemma2-2b" for s in small)


def test_get_llm_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_llm_spec("does-not-exist")
