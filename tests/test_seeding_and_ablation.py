"""Tests for ``src/utils/seeding.py`` and the ablation script entry point."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

from src.utils.seeding import derive_seed, set_global_seed


def test_set_global_seed_makes_random_reproducible():
    set_global_seed(123)
    a = [random.random() for _ in range(5)]
    set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_derive_seed_is_deterministic():
    s1 = derive_seed(42, "wm", "small", 0)
    s2 = derive_seed(42, "wm", "small", 0)
    s3 = derive_seed(42, "wm", "small", 1)
    assert s1 == s2
    assert s1 != s3


def test_ablation_sweep_help_exits_zero():
    """The CLI parser at least produces help without crashing."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "ablation_sweep.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--sweep" in result.stdout
