"""Tests for ``src/world_model/presets.py``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.world_model.presets import (
    WORLD_MODEL_PRESETS,
    get_preset,
    list_presets,
)


@pytest.mark.parametrize("name", list(WORLD_MODEL_PRESETS))
def test_preset_builds_world_model(name):
    preset = get_preset(name)
    cfg = preset.build_config(use_jepa=False)
    # Lazy import keeps the test runnable on CPU-only setups.
    from src.world_model.world_model import WorldModel
    wm = WorldModel(cfg)
    n_params = sum(p.numel() for p in wm.parameters())
    assert n_params > 0
    # Allow generous tolerance: presets are *target* sizes, not exact.
    expected_million = preset.approx_params_million
    assert n_params / 1e6 < expected_million * 20 + 1.0, (
        f"preset {name}: got {n_params/1e6:.2f}M params, expected "
        f"~{expected_million}M"
    )


def test_list_presets_filters_by_vram():
    small = list_presets(max_vram_gb=2.0)
    names = [p.name for p in small]
    assert "tiny" in names
    assert "xl" not in names


def test_get_preset_unknown():
    with pytest.raises(KeyError):
        get_preset("nonexistent")
