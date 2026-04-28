"""Determinism helpers.

Call ``set_global_seed(seed)`` once at the top of any script, benchmark,
or pytest fixture that should be reproducible.  This seeds Python's
``random`` module, NumPy, PyTorch (CPU + CUDA), and the
``PYTHONHASHSEED`` environment variable.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def set_global_seed(seed: Optional[int]) -> None:
    """Seed every RNG we rely on.  ``None`` is a no-op (useful for CLIs)."""
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism for matmul kernels; safe to leave on.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def derive_seed(parent_seed: int, *parts: object) -> int:
    """Deterministically derive a child seed from a parent + arbitrary parts.

    Used by the ablation sweep to give each (preset, repeat) pair its own
    seed without requiring the caller to track a counter.
    """
    h = hash((parent_seed, *parts)) & 0xFFFFFFFF
    return int(h)
