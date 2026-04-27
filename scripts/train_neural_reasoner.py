"""Thin CLI wrapper that delegates to :mod:`src.training.neural_reasoner_trainer`."""
from __future__ import annotations

from src.training.neural_reasoner_trainer import main

if __name__ == "__main__":
    raise SystemExit(main())
